// Timing harness for simd_mac: measures the register-to-register clock speed of the unit.
//
// The unit has about 130 input/output bits, more than a small FPGA has pins, so the inputs
// come from a 97-bit shift register (one serial input pin) and the 32-bit result is folded
// into one output pin. Every input bit stays live, so synthesis cannot remove any logic, and
// the critical path is the unit's own: input registers -> 4 multipliers -> adder tree ->
// 32-bit accumulate -> output register. Used by hardware/scripts/fmax.sh only.
module simd_mac_fmax_top (
    input  wire clk,
    input  wire din,
    output reg  dout,
    output reg  ovf
);
    reg [96:0] sr = 97'd0;
    reg [31:0] rs1_q = 32'd0, rs2_q = 32'd0, acc_q = 32'd0;
    reg        clr_q = 1'b0;
    always @(posedge clk) begin
        sr    <= {sr[95:0], din};
        rs1_q <= sr[31:0];
        rs2_q <= sr[63:32];
        acc_q <= sr[95:64];
        clr_q <= sr[96];
    end

    wire [31:0] mac_out;
    wire        valid_out, overflow_flag;
    simd_mac u_mac (
        .clk(clk), .rst_n(1'b1), .enable(1'b1),
        .rs1_data(rs1_q), .rs2_data(rs2_q), .acc_in(acc_q), .clr_acc(clr_q),
        .mac_out(mac_out), .valid_out(valid_out), .overflow_flag(overflow_flag)
    );

    always @(posedge clk) begin
        dout <= ^mac_out;
        ovf  <= overflow_flag & valid_out;
    end
endmodule
