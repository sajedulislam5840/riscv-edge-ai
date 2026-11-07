// =============================================================================
// Testbench: tb_mult_array
// Description: Unit testbench for 4-way 8x8 signed multiplier array AND
//              2-level balanced adder reduction tree (Plan 03.2 + Plan 03.3)
// Standard:    Verilog-2001
// =============================================================================

`timescale 1ns / 1ps

module tb_mult_array;

    reg  [31:0] rs1_data;
    reg  [31:0] rs2_data;
    integer     errors;
    integer     test_count;

    // -------------------------------------------------------------------------
    // Multiplier Array (Plan 03.2): 4-way 8x8 signed two's complement
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Balanced Adder Reduction Tree (Plan 03.3): 2-level binary tree
    //   Level 1: 16-bit + 16-bit -> 17-bit signed intermediate sums
    //   Level 2: 17-bit + 17-bit -> 18-bit signed final dot product
    // Bit-growth analysis:
    //   Products range: [-16256, +16384]  -> fits in signed 16-bit
    //   Level-1 sums:  [-32512, +32768]   -> fits in signed 17-bit [-65536, 65535]
    //   Level-2 sum:   [-65024, +65536]   -> fits in signed 18-bit [-131072, 131071]
    // -------------------------------------------------------------------------
    wire signed [16:0] sum_01  = prod0 + prod1;  // Level 1a
    wire signed [16:0] sum_23  = prod2 + prod3;  // Level 1b
    wire signed [17:0] sum_all = sum_01 + sum_23; // Level 2 (final dot product)

    // -------------------------------------------------------------------------
    // Task: check_vector - verify individual multiplier products
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Task: check_adder_tree - verify full reduction sum against expected value
    // -------------------------------------------------------------------------
    task check_adder_tree(
        input [31:0] v_rs1,
        input [31:0] v_rs2,
        input signed [17:0] exp_sum,
        input [127:0] test_name
    );
        begin
            rs1_data = v_rs1;
            rs2_data = v_rs2;
            #10;
            test_count = test_count + 1;
            if (sum_all !== exp_sum) begin
                $display("[FAIL] Adder Tree - %0s: RS1=%h RS2=%h", test_name, v_rs1, v_rs2);
                $display("       Expected sum_all=%d, Got sum_all=%d", exp_sum, sum_all);
                errors = errors + 1;
            end else begin
                $display("[PASS] Adder Tree - %0s: sum_all=%d", test_name, sum_all);
            end
        end
    endtask

    initial begin
        $display("==================================================");
        $display(" Running Multiplier Array + Adder Tree Testbench ");
        $display("==================================================");
        $display("--- Plan 03.2: 4-Way 8x8 Signed Multiplier Tests ---");

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
        // Byte 0: 10 * 20 = 200; Byte 1: -10 * 20 = -200
        // Byte 2: 50 * -50 = -2500; Byte 3: -1 * -1 = +1
        // rs1: [-1 (0xFF), 50 (0x32), -10 (0xF6), 10 (0x0A)] = 32'hFF32F60A
        // rs2: [-1 (0xFF), -50 (0xCE), 20 (0x14), 20 (0x14)] = 32'hFFCE1414
        check_vector(32'hFF32F60A, 32'hFFCE1414, 16'sd200, -16'sd200, -16'sd2500, 16'sd1, "Mixed Signs");

        $display("");
        $display("--- Plan 03.3: Balanced Adder Reduction Tree Tests ---");

        // Tree Test A: Maximum positive dot product
        // 4 * (-128 * -128) = 4 * 16384 = +65536
        // RS1=0x80808080 (all -128), RS2=0x80808080 (all -128)
        // 18-bit signed range: [-131072, 131071] -> +65536 fits safely
        check_adder_tree(32'h80808080, 32'h80808080, 18'sd65536, "MaxPos DotProduct (4*16384=+65536)");

        // Tree Test B: Maximum negative dot product
        // 4 * (-128 * +127) = 4 * (-16256) = -65024
        // RS1=0x80808080 (all -128), RS2=0x7F7F7F7F (all +127)
        check_adder_tree(32'h80808080, 32'h7F7F7F7F, -18'sd65024, "MaxNeg DotProduct (4*-16256=-65024)");

        // Tree Test C: Complete cancellation -> sum = 0
        // Byte 0: 25 * 40 = +1000; Byte 1: -25 * 40 = -1000
        // Byte 2: 25 * 20 = +500;  Byte 3: -25 * 20 = -500
        // RS1 = [25=0x19, -25=0xE7, 25=0x19, -25=0xE7] = 0xE719E719
        // RS2 = [40=0x28,  40=0x28, 20=0x14,  20=0x14] = 0x14142828
        check_adder_tree(32'hE719E719, 32'h14142828, 18'sd0, "Cancellation (+1000-1000+500-500=0)");

        // Tree Test D: All zeros -> sum_all = 0
        check_adder_tree(32'h00000000, 32'h00000000, 18'sd0, "Zero Vector DotProduct");

        $display("--------------------------------------------------");
        if (errors == 0) begin
            $display(">> [SUCCESS] All %0d Tests PASSED! <<", test_count);
        end else begin
            $display(">> [FAILURE] %0d / %0d Tests FAILED! <<", errors, test_count);
        end
        $display("==================================================");
        $finish;
    end

endmodule
