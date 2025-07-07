`timescale 5ns / 1ns

module tb_lut_mm;
    logic [9:0] data_in;     // 10-bit input
    logic [15:0] data_out;    // 16-bit output

    // Instantiate the module
    lut_mm uut (
        .data_in(data_in),
        .data_out(data_out)
    );

    initial begin
        // Apply all input combinations
        for (int i = 0; i < 16; i++) begin
            in = i; #10; // Wait for 10 time units
        end
        $finish;
    end
endmodule
