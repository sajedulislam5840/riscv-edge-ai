// =============================================================================
// Testbench: tb_mult_array
// Description: Unit testbench for 4-way 8x8 signed two's complement multiplier array
// Standard:    Verilog-2001
// =============================================================================

`timescale 1ns / 1ps

module tb_mult_array;

    reg  [31:0] rs1_data;
    reg  [31:0] rs2_data;
    integer     errors;
    integer     test_count;

    // Direct instantiation of 4 signed multipliers matching simd_mac datapath
    wire signed [7:0] a0 = $signed(rs1_data[ 7: 0]);
    wire signed [7:0] a1 = $signed(rs1_data[15: 8]);
    wire signed [7:0] a2 = $signed(rs1_data[23:16]);
    wire signed [7:0] a3 = $signed(rs1_data[31:24]);

    wire signed [7:0] b0 = $signed(rs2_data[ 7: 0]);
    wire signed [7:0] b1 = $signed(rs2_data[15: 8]);
    wire signed [7:0] b2 = $signed(rs2_data[23:16]);
    wire signed [7:0] b3 = $signed(rs2_data[31:24]);

    wire signed [15:0] prod0 = a0 * b0;
    wire signed [15:0] prod1 = a1 * b1;
    wire signed [15:0] prod2 = a2 * b2;
    wire signed [15:0] prod3 = a3 * b3;

    // Check task
    task check_vector(
        input [31:0] v_rs1,
        input [31:0] v_rs2,
        input signed [15:0] exp0,
        input signed [15:0] exp1,
        input signed [15:0] exp2,
        input signed [15:0] exp3,
        input [127:0] test_name
    );
        begin
            rs1_data = v_rs1;
            rs2_data = v_rs2;
            #10;
            test_count = test_count + 1;
            if (prod0 !== exp0 || prod1 !== exp1 || prod2 !== exp2 || prod3 !== exp3) begin
                $display("[FAIL] %0s: RS1=%h RS2=%h", test_name, v_rs1, v_rs2);
                $display("       Expected: [%d, %d, %d, %d]", exp0, exp1, exp2, exp3);
                $display("       Actual  : [%d, %d, %d, %d]", prod0, prod1, prod2, prod3);
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s: [%d, %d, %d, %d]", test_name, prod0, prod1, prod2, prod3);
            end
        end
    endtask

    initial begin
        $display("==================================================");
        $display("   Running 4-Way 8x8 Multiplier Array Testbench   ");
        $display("==================================================");

        errors = 0;
        test_count = 0;

        // 1. All Zero Test
        check_vector(32'h00000000, 32'h00000000, 16'sd0, 16'sd0, 16'sd0, 16'sd0, "All Zeros");

        // 2. Identity Test (all 1s * all 1s)
        check_vector(32'h01010101, 32'h01010101, 16'sd1, 16'sd1, 16'sd1, 16'sd1, "All Ones");

        // 3. Maximum Negative x Maximum Negative: -128 * -128 = +16384 (0x4000)
        // 0x80 = 8'b10000000 = -128 in signed 8-bit
        check_vector(32'h80808080, 32'h80808080, 16'sd16384, 16'sd16384, 16'sd16384, 16'sd16384, "MaxNeg x MaxNeg (-128*-128)");

        // 4. Maximum Negative x Maximum Positive: -128 * +127 = -16256 (-0x3F80)
        // 0x7F = +127, 0x80 = -128
        check_vector(32'h80808080, 32'h7F7F7F7F, -16'sd16256, -16'sd16256, -16'sd16256, -16'sd16256, "MaxNeg x MaxPos (-128*+127)");

        // 5. Maximum Positive x Maximum Positive: +127 * +127 = +16129 (0x3F01)
        check_vector(32'h7F7F7F7F, 32'h7F7F7F7F, 16'sd16129, 16'sd16129, 16'sd16129, 16'sd16129, "MaxPos x MaxPos (+127*+127)");

        // 6. Zero multiplication with extremes: 0 * -128 = 0
        check_vector(32'h00000000, 32'h80808080, 16'sd0, 16'sd0, 16'sd0, 16'sd0, "Zero x MaxNeg");

        // 7. Alternating Mixed Signs:
        // Byte 0: 10 * 20 = 200
        // Byte 1: -10 * 20 = -200
        // Byte 2: 50 * -50 = -2500
        // Byte 3: -1 * -1 = +1
        // rs1: [-1 (0xFF), 50 (0x32), -10 (0xF6), 10 (0x0A)] = 32'hFF32F60A
        // rs2: [-1 (0xFF), -50 (0xCE), 20 (0x14), 20 (0x14)] = 32'hFFCE1414
        check_vector(32'hFF32F60A, 32'hFFCE1414, 16'sd200, -16'sd200, -16'sd2500, 16'sd1, "Mixed Signs");

        $display("--------------------------------------------------");
        if (errors == 0) begin
            $display(">> [SUCCESS] All %0d Multiplier Tests PASSED! <<", test_count);
        end else begin
            $display(">> [FAILURE] %0d / %0d Multiplier Tests FAILED! <<", errors, test_count);
        end
        $display("==================================================");
        $finish;
    end

endmodule
