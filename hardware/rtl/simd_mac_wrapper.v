// =============================================================================
// Module Name: simd_mac_wrapper
// Description: RISC-V Custom-0 Co-processor Wrapper for SIMD MAC Unit
// Standard:    Verilog-2001 synthesizable RTL
// Target:      RISC-V ISA Extension Opcode 0x0B (custom-0, 7'b0001011)
//
// Instruction Encoding (R-type):
//   [ 31:25] funct7  - 7'b0000000: simd_mac      rd = rd + dot4(rs1, rs2)   (reads rd: 3 source operands)
//                    - 7'b0000001: simd_mac_clr  rd = dot4(rs1, rs2)
//                    - 7'b0000010: simd_macc     acc = acc + dot4(rs1, rs2); rd = acc
//                    - 7'b0000011: simd_macz     acc = dot4(rs1, rs2);       rd = acc
//
//   simd_macc / simd_macz keep the running total in the unit's own result register, so they
//   read only rs1 and rs2. That fits cores with two register read ports and coprocessor
//   interfaces such as PicoRV32's PCPI, which pass two source operands. The accumulator holds
//   the result of the last MAC instruction and is kept while other instructions run.
//   [ 24:20] rs2     - Source register 2 (Vector B packed INT8)
//   [ 19:15] rs1     - Source register 1 (Vector A packed INT8)
//   [ 14:12] funct3  - 3'b000
//   [ 11: 7] rd      - Destination register (accumulator in/out)
//   [  6: 0] opcode  - 7'b0001011 (custom-0)
//
// Co-processor Handshake Protocol:
//   Request:  req_valid HIGH with insn, rs1_val, rs2_val, rd_val valid
//   Response: rsp_valid HIGH next cycle with rsp_data = mac result
// =============================================================================

`timescale 1ns / 1ps

module simd_mac_wrapper (
    input  wire        clk,         // System clock
    input  wire        rst_n,       // Active-low synchronous reset

    // CPU pipeline interface
    input  wire [31:0] insn,        // Full 32-bit R-type instruction word
    input  wire [31:0] rs1_val,     // Value of rs1 register (Vector A, 4x INT8 packed)
    input  wire [31:0] rs2_val,     // Value of rs2 register (Vector B, 4x INT8 packed)
    input  wire [31:0] rd_val,      // Current value of rd (accumulator input)
    input  wire        req_valid,   // High when instruction and register values are valid

    // Co-processor response interface
    output wire        req_ready,   // Always ready (1-cycle latency, single-stage pipeline)
    output wire        rsp_valid,   // High when rsp_data holds valid result
    output wire [31:0] rsp_data,    // MAC result to write back to rd
    output wire        rsp_overflow // Overflow flag for CSR or trap handler
);

    // -------------------------------------------------------------------------
    // 1. R-type Instruction Field Decode
    // -------------------------------------------------------------------------
    wire [6:0]  opcode = insn[ 6: 0];
    wire [4:0]  rd     = insn[11: 7];   // Destination register index (unused here, for decode)
    wire [2:0]  funct3 = insn[14:12];
    wire [4:0]  rs1    = insn[19:15];   // Source register 1 index (unused here, for decode)
    wire [4:0]  rs2    = insn[24:20];   // Source register 2 index (unused here, for decode)
    wire [6:0]  funct7 = insn[31:25];

    // -------------------------------------------------------------------------
    // 2. Opcode Validity Check
    //    custom-0 = 7'b0001011 = 7'h0B
    //    Only process when opcode matches and req_valid is asserted
    // -------------------------------------------------------------------------
    localparam [6:0] OPCODE_CUSTOM0 = 7'b0001011;
    localparam [6:0] FUNCT7_MAC     = 7'b0000000;  // simd_mac: accumulate
    localparam [6:0] FUNCT7_MAC_CLR = 7'b0000001;  // simd_mac_clr: clear then accumulate
    localparam [6:0] FUNCT7_MACC    = 7'b0000010;  // simd_macc: accumulate into the internal total
    localparam [6:0] FUNCT7_MACZ    = 7'b0000011;  // simd_macz: start a new internal total

    wire is_custom0  = (opcode == OPCODE_CUSTOM0) && (funct3 == 3'b000);
    wire is_mac_clr  = is_custom0 && (funct7 == FUNCT7_MAC_CLR);
    wire is_mac      = is_custom0 && (funct7 == FUNCT7_MAC);
    wire is_macc     = is_custom0 && (funct7 == FUNCT7_MACC);
    wire is_macz     = is_custom0 && (funct7 == FUNCT7_MACZ);
    wire insn_valid  = (is_mac || is_mac_clr || is_macc || is_macz) && req_valid;

    // -------------------------------------------------------------------------
    // 3. SIMD MAC Core Instantiation
    //    clr_acc = 1 for simd_mac_clr (funct7=0000001)
    //    clr_acc = 0 for simd_mac     (funct7=0000000)
    // -------------------------------------------------------------------------
    wire [31:0] mac_out_wire;
    wire        valid_out_wire;
    wire        overflow_wire;

    // Internal-accumulator forms feed the unit's own result register back into the adder,
    // so the register file's rd value is not needed.
    wire [31:0] acc_sel = (is_macc || is_macz) ? mac_out_wire : rd_val;
    wire        clr_sel = is_mac_clr || is_macz;

    simd_mac u_simd_mac (
        .clk          (clk),
        .rst_n        (rst_n),
        .enable       (insn_valid),
        .rs1_data     (rs1_val),
        .rs2_data     (rs2_val),
        .acc_in       (acc_sel),
        .clr_acc      (clr_sel),
        .mac_out      (mac_out_wire),
        .valid_out    (valid_out_wire),
        .overflow_flag(overflow_wire)
    );

    // -------------------------------------------------------------------------
    // 4. Co-processor Handshake Output
    //    Single-cycle latency: req arrives cycle N, rsp valid on cycle N+1
    //    req_ready is always 1 (no backpressure, pipeline never stalls)
    // -------------------------------------------------------------------------
    assign req_ready   = 1'b1;           // Always accept — no multi-cycle stall
    assign rsp_valid   = valid_out_wire; // Registered inside simd_mac
    assign rsp_data    = mac_out_wire;   // MAC result
    assign rsp_overflow = overflow_wire; // Overflow flag passthrough

endmodule
