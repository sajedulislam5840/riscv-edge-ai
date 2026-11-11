// Timing harness for simd_mac_wrapper (custom-0 decoder + MAC unit + accumulator feedback).
// Same idea as simd_mac_fmax_top.v: inputs come from a shift register fed by one pin and the
// result is folded into one output pin, so the measured path is the wrapper's own.
// Used by hardware/scripts/fmax.sh only.
module simd_mac_wrapper_fmax_top (
    input  wire clk,
    input  wire din,
    output reg  dout,
    output reg  ovf
);
    reg [128:0] sr = 129'd0;
    reg [31:0] insn_q = 32'd0, rs1_q = 32'd0, rs2_q = 32'd0, rd_q = 32'd0;
    reg        valid_q = 1'b0;
    always @(posedge clk) begin
        sr      <= {sr[127:0], din};
        insn_q  <= {sr[31:7], 7'b0001011};  // keep the opcode on custom-0 so the decoder is live
        rs1_q   <= sr[63:32];
        rs2_q   <= sr[95:64];
        rd_q    <= sr[127:96];
        valid_q <= sr[128];
    end

    wire        rsp_valid, rsp_overflow;
    wire [31:0] rsp_data;
    simd_mac_wrapper u_wrap (
        .clk(clk), .rst_n(1'b1), .insn(insn_q), .rs1_val(rs1_q), .rs2_val(rs2_q), .rd_val(rd_q),
        .req_valid(valid_q), .req_ready(), .rsp_valid(rsp_valid), .rsp_data(rsp_data), .rsp_overflow(rsp_overflow)
    );

    always @(posedge clk) begin
        dout <= ^rsp_data;
        ovf  <= rsp_overflow & rsp_valid;
    end
endmodule
