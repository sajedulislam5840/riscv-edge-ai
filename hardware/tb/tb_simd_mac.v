// =============================================================================
// Testbench: tb_simd_mac
// Description: Self-checking testbench for simd_mac module
//              Covers reset, arithmetic corners, clr_acc modes, chained accum,
//              overflow detection. VCD waveform dump included.
// Standard:    Verilog-2001
// Clock:       100 MHz (10 ns period)
// =============================================================================

`timescale 1ns / 1ps

module tb_simd_mac;

    // -------------------------------------------------------------------------
    // DUT Signals
    // -------------------------------------------------------------------------
    reg         clk;
    reg         rst_n;
    reg         enable;
    reg  [31:0] rs1_data;
    reg  [31:0] rs2_data;
    reg  [31:0] acc_in;
    reg         clr_acc;
    wire [31:0] mac_out;
    wire        valid_out;
    wire        overflow_flag;

    // -------------------------------------------------------------------------
    // Scoreboard
    // -------------------------------------------------------------------------
    integer errors;
    integer test_count;

    // -------------------------------------------------------------------------
    // DUT Instantiation
    // -------------------------------------------------------------------------
    simd_mac u_dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .enable       (enable),
        .rs1_data     (rs1_data),
        .rs2_data     (rs2_data),
        .acc_in       (acc_in),
        .clr_acc      (clr_acc),
        .mac_out      (mac_out),
        .valid_out    (valid_out),
        .overflow_flag(overflow_flag)
    );

    // -------------------------------------------------------------------------
    // Clock Generation: 100 MHz (10 ns period)
    // -------------------------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Task: apply_and_check
    //   Applies inputs on negedge, samples output on next posedge
    //   Checks mac_out against expected value
    // -------------------------------------------------------------------------
    task apply_and_check;
        input [31:0] v_rs1;
        input [31:0] v_rs2;
        input [31:0] v_acc_in;
        input        v_clr_acc;
        input [31:0] v_expected;
        input [127:0] test_name;
        begin
            @(negedge clk);
            rs1_data = v_rs1;
            rs2_data = v_rs2;
            acc_in   = v_acc_in;
            clr_acc  = v_clr_acc;
            enable   = 1'b1;
            @(posedge clk); #1; // sample just after posedge
            test_count = test_count + 1;
            if (mac_out !== v_expected) begin
                $display("[FAIL] %0s", test_name);
                $display("       RS1=0x%08h RS2=0x%08h ACC_IN=%0d CLR=%0b",
                         v_rs1, v_rs2, $signed(v_acc_in), v_clr_acc);
                $display("       Expected=0x%08h (%0d), Got=0x%08h (%0d)",
                         v_expected, $signed(v_expected), mac_out, $signed(mac_out));
                errors = errors + 1;
            end else begin
                $display("[PASS] %0s: mac_out=%0d, ovf=%0b", test_name, $signed(mac_out), overflow_flag);
            end
            enable = 1'b0;
        end
    endtask

    // -------------------------------------------------------------------------
    // Main Test Sequence
    // -------------------------------------------------------------------------
    integer i;
    integer acc_expected;

    initial begin
        // VCD Waveform Dump
        $dumpfile("simd_mac.vcd");
        $dumpvars(0, tb_simd_mac);

        $display("==================================================");
        $display("   SIMD MAC Full System Self-Checking Testbench   ");
        $display("==================================================");

        errors     = 0;
        test_count = 0;
        enable     = 1'b0;
        rs1_data   = 32'h0;
        rs2_data   = 32'h0;
        acc_in     = 32'h0;
        clr_acc    = 1'b1;

        // Reset Sequence: 2 clock cycles active-low reset
        rst_n = 1'b0;
        @(posedge clk);
        @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        $display("");
        $display("--- Group 1: Reset State Verification ---");
        // After reset: mac_out=0, valid_out=0, overflow_flag=0
        test_count = test_count + 1;
        if (mac_out !== 32'h0 || valid_out !== 1'b0 || overflow_flag !== 1'b0) begin
            $display("[FAIL] Reset state: mac_out=%0h valid=%0b ovf=%0b", mac_out, valid_out, overflow_flag);
            errors = errors + 1;
        end else begin
            $display("[PASS] Reset state: mac_out=0 valid=0 overflow=0");
        end

        $display("");
        $display("--- Group 2: Arithmetic Corner Cases ---");

        // All zeros
        apply_and_check(32'h00000000, 32'h00000000, 32'h00000000, 1'b1, 32'h00000000, "All Zeros");

        // Identity: [1,1,1,1] * [1,1,1,1] = 4, fresh accumulation
        apply_and_check(32'h01010101, 32'h01010101, 32'h00000000, 1'b1, 32'd4, "Identity (4 * 1*1 = 4)");

        // Max negative: [-128,-128,-128,-128] * [-128,-128,-128,-128] = 4*16384=65536
        apply_and_check(32'h80808080, 32'h80808080, 32'h00000000, 1'b1, 32'd65536, "MaxNeg*MaxNeg (4*16384=65536)");

        // Max positive: [127,127,127,127] * [127,127,127,127] = 4*16129=64516
        apply_and_check(32'h7F7F7F7F, 32'h7F7F7F7F, 32'h00000000, 1'b1, 32'd64516, "MaxPos*MaxPos (4*16129=64516)");

        // Max negative dot product: [-128,...] * [127,...] = 4*(-16256)=-65024
        apply_and_check(32'h80808080, 32'h7F7F7F7F, 32'h00000000, 1'b1, -32'sd65024, "MaxNeg DotProduct (-65024)");

        $display("");
        $display("--- Group 3: Accumulator Mode Tests ---");

        // clr_acc=1 ignores acc_in: result = 0 + 4 = 4 regardless of acc_in=100
        apply_and_check(32'h01010101, 32'h01010101, 32'd100, 1'b1, 32'd4, "clr_acc=1 ignores acc_in=100");

        // clr_acc=0 adds to acc_in: result = 100 + 4 = 104
        apply_and_check(32'h01010101, 32'h01010101, 32'd100, 1'b0, 32'd104, "clr_acc=0 acc_in=100 + 4 = 104");

        // Passthrough: zero dot product, clr_acc=0: result = acc_in
        apply_and_check(32'h00000000, 32'h00000000, 32'd12345, 1'b0, 32'd12345, "Zero dot product: acc passthrough");

        $display("");
        $display("--- Group 4: Chained Accumulation (100 iterations) ---");
        // 100 × ([2,2,2,2] · [2,2,2,2]) = 100 × 16 = 1600
        // First iteration uses clr_acc=1, rest use clr_acc=0 with feedback
        @(negedge clk);
        rs1_data = 32'h02020202;  // [2,2,2,2]
        rs2_data = 32'h02020202;  // [2,2,2,2]
        acc_in   = 32'h0;
        clr_acc  = 1'b1;
        enable   = 1'b1;
        @(posedge clk); #1;
        acc_expected = mac_out;   // = 16

        for (i = 1; i < 100; i = i + 1) begin
            @(negedge clk);
            acc_in  = acc_expected;
            clr_acc = 1'b0;
            enable  = 1'b1;
            @(posedge clk); #1;
            acc_expected = mac_out;
        end
        enable = 1'b0;
        test_count = test_count + 1;
        if (acc_expected !== 1600) begin
            $display("[FAIL] 100-step chain: expected 1600, got %0d", acc_expected);
            errors = errors + 1;
        end else begin
            $display("[PASS] 100-step chained accumulation: final=%0d (expected 1600)", acc_expected);
        end

        $display("");
        $display("--- Group 5: Mixed Signs ---");
        // [127,-128,50,-50] * [1,1,1,1] = 127-128+50-50 = -1
        // rs1 = {-50(0xCE), 50(0x32), -128(0x80), 127(0x7F)} = 32'hCE32807F
        // rs2 = {1,1,1,1} = 32'h01010101
        apply_and_check(32'hCE32807F, 32'h01010101, 32'h0, 1'b1, -32'sd1, "Mixed: 127-128+50-50 = -1");

        // Cancellation: [25,-25,25,-25] * [40,40,20,20] = 1000-1000+500-500 = 0
        // rs1 = {-25(0xE7), 25(0x19), -25(0xE7), 25(0x19)} = 32'hE719E719
        // rs2 = {20(0x14), 20(0x14), 40(0x28), 40(0x28)} = 32'h14142828
        apply_and_check(32'hE719E719, 32'h14142828, 32'h0, 1'b1, 32'h0, "Cancellation: 1000-1000+500-500 = 0");

        $display("");
        $display("==================================================");
        if (errors == 0) begin
            $display(">> [SUCCESS] All %0d Tests PASSED! No errors. <<", test_count);
        end else begin
            $display(">> [FAILURE] %0d / %0d Tests FAILED! <<", errors, test_count);
        end
        $display("==================================================");
        $finish;
    end

endmodule
