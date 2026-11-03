// =============================================================================
// Module Name: simd_mac
// Description: 4-Way SIMD INT8 Multiply-Accumulate (MAC) Unit
// Standard:    Verilog-2001 synthesizable RTL
// Target:      RISC-V Custom-0 ISA Extension (Opcode 0x0B)
//
// Computes:
//   mac_out <= (clr_acc ? 32'sd0 : acc_in) + sum(a_i * b_i)  for i in [0..3]
//
// Little-Endian 32-bit Register Packing:
//   Byte 0: bits [ 7: 0] (int8_t signed)
//   Byte 1: bits [15: 8] (int8_t signed)
//   Byte 2: bits [23:16] (int8_t signed)
//   Byte 3: bits [31:24] (int8_t signed)
// =============================================================================

`timescale 1ns / 1ps

module simd_mac (
    input  wire        clk,           // System clock
    input  wire        rst_n,         // Active-low synchronous reset
    input  wire        enable,        // Clock enable / execution gate
    input  wire [31:0] rs1_data,      // 4 packed 8-bit signed operands (Vector A)
    input  wire [31:0] rs2_data,      // 4 packed 8-bit signed operands (Vector B)
    input  wire [31:0] acc_in,        // 32-bit accumulator operand input (from rd)
    input  wire        clr_acc,       // 1 = start fresh sum (0 + dot product); 0 = accumulate with acc_in
    output reg  [31:0] mac_out,       // 32-bit signed accumulated output (to rd)
    output reg         valid_out,     // High when mac_out holds valid result
    output reg         overflow_flag  // High if signed 32-bit accumulation overflow occurred
);

    // -------------------------------------------------------------------------
    // 1. Little-Endian Byte Unpacking into 8-bit Signed Operands
    // -------------------------------------------------------------------------
    wire signed [7:0] a0 = rs1_data[ 7: 0];
    wire signed [7:0] a1 = rs1_data[15: 8];
    wire signed [7:0] a2 = rs1_data[23:16];
    wire signed [7:0] a3 = rs1_data[31:24];

    wire signed [7:0] b0 = rs2_data[ 7: 0];
    wire signed [7:0] b1 = rs2_data[15: 8];
    wire signed [7:0] b2 = rs2_data[23:16];
    wire signed [7:0] b3 = rs2_data[31:24];

    // -------------------------------------------------------------------------
    // 2. Parallel 8x8 Signed Multipliers (Outputs: 16-bit signed products)
    // -------------------------------------------------------------------------
    wire signed [15:0] prod0 = a0 * b0;
    wire signed [15:0] prod1 = a1 * b1;
    wire signed [15:0] prod2 = a2 * b2;
    wire signed [15:0] prod3 = a3 * b3;

    // -------------------------------------------------------------------------
    // 3. Balanced 2-Level Adder Reduction Tree
    //    Level 1: 16-bit + 16-bit -> 17-bit signed sum
    //    Level 2: 17-bit + 17-bit -> 18-bit signed sum (Zero intermediate overflow)
    // -------------------------------------------------------------------------
    wire signed [16:0] sum_01  = prod0 + prod1;
    wire signed [16:0] sum_23  = prod2 + prod3;
    wire signed [17:0] sum_all = sum_01 + sum_23;

    // -------------------------------------------------------------------------
    // 4. Accumulator Base Muxing & 32-bit Summation
    // -------------------------------------------------------------------------
    wire signed [31:0] effective_acc = clr_acc ? 32'sd0 : $signed(acc_in);
    wire signed [31:0] sum_all_ext   = {{14{sum_all[17]}}, sum_all}; // 18-bit to 32-bit sign extension
    wire signed [31:0] next_mac_out  = effective_acc + sum_all_ext;

    // -------------------------------------------------------------------------
    // 5. Overflow Detection Logic
    //    Two's complement overflow occurs when operands have the same sign,
    //    but the resulting sum has a different sign.
    // -------------------------------------------------------------------------
    wire pos_overflow = (effective_acc[31] == 1'b0) && (sum_all_ext[31] == 1'b0) && (next_mac_out[31] == 1'b1);
    wire neg_overflow = (effective_acc[31] == 1'b1) && (sum_all_ext[31] == 1'b1) && (next_mac_out[31] == 1'b0);
    wire next_overflow = pos_overflow | neg_overflow;

    // -------------------------------------------------------------------------
    // 6. Synchronous Registered Output Stage
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            mac_out       <= 32'sd0;
            valid_out     <= 1'b0;
            overflow_flag <= 1'b0;
        end else if (enable) begin
            mac_out       <= next_mac_out;
            valid_out     <= 1'b1;
            overflow_flag <= next_overflow;
        end else begin
            valid_out     <= 1'b0;
        end
    end

endmodule
