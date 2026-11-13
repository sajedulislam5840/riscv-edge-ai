// PicoRV32 with the SIMD MAC attached through PCPI. Memory and the UART live in the testbench.
// WITH_MAC = 0 leaves the PCPI port unanswered: the same core without the unit.
`timescale 1ns / 1ps

module pico_mac_soc #(
    parameter WITH_MAC = 1
) (
    input  wire        clk,
    input  wire        resetn,
    output wire        trap,

    output wire        mem_valid,
    output wire        mem_instr,
    input  wire        mem_ready,
    output wire [31:0] mem_addr,
    output wire [31:0] mem_wdata,
    output wire [ 3:0] mem_wstrb,
    input  wire [31:0] mem_rdata
);
    wire        pcpi_valid, pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_insn, pcpi_rs1, pcpi_rs2, pcpi_rd;

    picorv32 #(
        .ENABLE_COUNTERS   (1),
        .ENABLE_COUNTERS64 (1),
        .ENABLE_REGS_16_31 (1),
        .ENABLE_REGS_DUALPORT (1),
        .BARREL_SHIFTER    (1),
        .ENABLE_MUL        (0),
        .ENABLE_FAST_MUL   (1),
        .ENABLE_DIV        (1),
        .ENABLE_PCPI       (1),
        .COMPRESSED_ISA    (0),
        .CATCH_MISALIGN    (1),
        .CATCH_ILLINSN     (1),
        .PROGADDR_RESET    (32'h0000_0000)
    ) cpu (
        .clk(clk), .resetn(resetn), .trap(trap),
        .mem_valid(mem_valid), .mem_instr(mem_instr), .mem_ready(mem_ready),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata), .mem_wstrb(mem_wstrb), .mem_rdata(mem_rdata),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn), .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd), .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .irq(32'b0)
    );

    generate
        if (WITH_MAC) begin : g_mac
            pcpi_simd_mac u_pcpi (
                .clk(clk), .resetn(resetn),
                .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
                .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
                .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
                .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready)
            );
        end else begin : g_nomac
            assign pcpi_wr = 1'b0;
            assign pcpi_rd = 32'd0;
            assign pcpi_wait = 1'b0;
            assign pcpi_ready = 1'b0;
        end
    endgenerate
endmodule
