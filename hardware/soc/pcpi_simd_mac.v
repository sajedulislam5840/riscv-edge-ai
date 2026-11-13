// Connects the SIMD MAC wrapper to PicoRV32's PCPI (pico co-processor interface).
//
// PicoRV32 raises pcpi_valid for every instruction it does not decode itself and waits for a
// co-processor to answer. This adapter accepts custom-0 instructions with funct7 = 1, 2 or 3
// (simd_mac_clr, simd_macc, simd_macz), starts the MAC once, holds pcpi_wait so the core does
// not time out, and returns the result one cycle later. funct7 = 0 (simd_mac) reads a third
// register, which PCPI cannot supply, so it is left unanswered and the core traps on it.
//
// Cost: 2 extra cycles per MAC instruction (the cycle that issues it and the cycle that
// returns the result).
`timescale 1ns / 1ps

module pcpi_simd_mac (
    input  wire        clk,
    input  wire        resetn,

    input  wire        pcpi_valid,
    input  wire [31:0] pcpi_insn,
    input  wire [31:0] pcpi_rs1,
    input  wire [31:0] pcpi_rs2,
    output wire        pcpi_wr,
    output wire [31:0] pcpi_rd,
    output wire        pcpi_wait,
    output wire        pcpi_ready
);
    wire mine = pcpi_valid
             && pcpi_insn[6:0]   == 7'b0001011      // custom-0
             && pcpi_insn[14:12] == 3'b000
             && pcpi_insn[31:27] == 5'b00000
             && pcpi_insn[26:25] != 2'b00;          // funct7 = 1, 2 or 3

    reg busy;                                        // MAC started, result due next cycle
    wire issue = mine && !busy;

    always @(posedge clk) begin
        if (!resetn) busy <= 1'b0;
        else         busy <= issue;
    end

    wire        mac_rsp_valid;
    wire [31:0] mac_rsp_data;
    wire        mac_req_ready, mac_overflow;

    simd_mac_wrapper u_mac (
        .clk          (clk),
        .rst_n        (resetn),
        .insn         (pcpi_insn),
        .rs1_val      (pcpi_rs1),
        .rs2_val      (pcpi_rs2),
        .rd_val       (32'd0),                       // the 2-operand forms never read it
        .req_valid    (issue),
        .req_ready    (mac_req_ready),
        .rsp_valid    (mac_rsp_valid),
        .rsp_data     (mac_rsp_data),
        .rsp_overflow (mac_overflow)
    );

    assign pcpi_wait  = issue;
    assign pcpi_ready = busy && mac_rsp_valid;
    assign pcpi_wr    = 1'b1;
    assign pcpi_rd    = mac_rsp_data;
endmodule
