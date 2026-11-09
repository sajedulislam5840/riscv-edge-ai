// =============================================================================
// Testbench: tb_simd_mac_cosim
// Description: File-driven co-simulation testbench for simd_mac
//              Streams all vectors from golden_vectors.hex through the hardware
//              and asserts 100% bit-exact match with the golden reference.
// Standard:    Verilog-2001
// =============================================================================

`timescale 1ns / 1ps

module tb_simd_mac_cosim;

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

    // Scoreboard
    integer errors;
    integer test_count;

    // Vector memory: store up to 30000 test vectors
    // Format: rs1[31:0], rs2[31:0], acc_in[31:0], clr_acc[0], expected[31:0]
    reg [31:0] v_rs1    [0:29999];
    reg [31:0] v_rs2    [0:29999];
    reg [31:0] v_acc_in [0:29999];
    reg [0:0]  v_clr    [0:29999];
    reg [31:0] v_exp    [0:29999];

    integer total_vectors;
    integer i;

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
    // Clock Generation: 100 MHz
    // -------------------------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------------
    // Load and stream vectors
    // -------------------------------------------------------------------------
    initial begin
        $display("==================================================");
        $display("   SIMD MAC File-Driven Co-Simulation Testbench   ");
        $display("==================================================");

        errors       = 0;
        test_count   = 0;
        enable       = 1'b0;

        // Reset
        rst_n = 1'b0;
        @(posedge clk);
        @(posedge clk);
        rst_n = 1'b1;

        // Read golden vectors hex file
        // Each line: rs1 rs2 acc_in clr expected (space-separated hex)
        // Using $readmemh requires a pure hex format, so we use $fopen/$fscanf
        begin : LOAD_VECTORS
            integer fd;
            integer ret;
            reg [31:0] tmp_rs1, tmp_rs2, tmp_acc, tmp_exp;
            reg        tmp_clr;

            fd = $fopen("hardware/tb/golden_vectors.hex", "r");
            if (fd == 0) begin
                $display("[ERROR] Cannot open golden_vectors.hex");
                $finish;
            end

            total_vectors = 0;
            ret = 1;
            while (ret > 0 && total_vectors < 30000) begin
                ret = $fscanf(fd, "%h %h %h %b %h\n",
                              tmp_rs1, tmp_rs2, tmp_acc, tmp_clr, tmp_exp);
                if (ret == 5) begin
                    v_rs1   [total_vectors] = tmp_rs1;
                    v_rs2   [total_vectors] = tmp_rs2;
                    v_acc_in[total_vectors] = tmp_acc;
                    v_clr   [total_vectors] = tmp_clr;
                    v_exp   [total_vectors] = tmp_exp;
                    total_vectors = total_vectors + 1;
                end
            end
            $fclose(fd);
        end

        $display("Loaded %0d test vectors from golden_vectors.hex", total_vectors);
        $display("Streaming through simd_mac DUT...");
        $display("");

        // Stream all vectors through DUT
        for (i = 0; i < total_vectors; i = i + 1) begin
            @(negedge clk);
            rs1_data = v_rs1[i];
            rs2_data = v_rs2[i];
            acc_in   = v_acc_in[i];
            clr_acc  = v_clr[i];
            enable   = 1'b1;
            @(posedge clk); #1;
            test_count = test_count + 1;
            if (mac_out !== v_exp[i]) begin
                $display("[FAIL] Vector %0d: RS1=%08h RS2=%08h ACC=%08h CLR=%0b",
                         i, v_rs1[i], v_rs2[i], v_acc_in[i], v_clr[i]);
                $display("       Expected=%08h (%0d), Got=%08h (%0d)",
                         v_exp[i], $signed(v_exp[i]), mac_out, $signed(mac_out));
                errors = errors + 1;
                if (errors > 10) begin
                    $display("[ABORT] >10 errors, stopping early.");
                    disable LOAD_VECTORS;
                end
            end
            enable = 1'b0;
        end

        $display("==================================================");
        if (errors == 0) begin
            $display(">> [SUCCESS] All %0d Co-Sim Vectors PASSED! <<", test_count);
            $display("   100%% bit-exact parity: RTL == Golden Model");
        end else begin
            $display(">> [FAILURE] %0d / %0d Vectors FAILED! <<", errors, test_count);
        end
        $display("==================================================");
        $finish;
    end

endmodule
