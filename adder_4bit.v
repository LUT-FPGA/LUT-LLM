// 4-bit Adder Module
// Takes two 4-bit inputs and produces one 4-bit sum output
`timescale 1 ns / 1 ps
module adder_store_unit (
    input wire ap_clk,
    input wire ap_rst,
    input wire reconfig_ce,
    input wire read_enable,
    input wire [7:0] data_in,       // sequential data for write mode
    input wire [3:0] a,        // First 4-bit input
    input wire [3:0] b,        // Second 4-bit input
    output reg [3:0] sum,       // 4-bit sum output
    output wire carry
);

    wire carry_0;
    wire carry_1;
    wire carry_2;
    wire carry_3;

    wire [3:0] out_reg;

    // bit 0

    CFGLUT5 #(
        .INIT(32'h96)
    )
    CFGLUT5_cout_inst_0 (
        .O5(out_reg[0]),
        .CDI(data_in[0]), // Use data_in for write mode
        .CE(reconfig_ce),
        .CLK(ap_clk),
        .I0(1'b0),
        .I1(b[0]),
        .I2(a[0]),
        .I3(1'b0)
    );

    CFGLUT5 #(
        .INIT(32'he8)
    )
    CFGLUT5_carry_inst_0 (
        .O5(carry_0),
        .CDI(data_in[1]), // Use data_in for write mode
        .CE(reconfig_ce),
        .CLK(ap_clk),
        .I0(1'b0),
        .I1(b[0]),
        .I2(a[0]),
        .I3(1'b0)
    );

    // bit 1

    CFGLUT5 #(
        .INIT(32'h96)
    )
    CFGLUT5_cout_inst_1 (
        .O5(out_reg[1]),
        .CDI(data_in[2]), // Use data_in for write mode
        .CE(reconfig_ce),
        .CLK(ap_clk),
        .I0(carry_0),
        .I1(b[1]),
        .I2(a[1]),
        .I3(1'b0)
    );

    CFGLUT5 #(
        .INIT(32'he8)
    )
    CFGLUT5_carry_inst_1 (
        .O5(carry_1),
        .CDI(data_in[3]), // Use data_in for write mode
        .CE(reconfig_ce),
        .CLK(ap_clk),
        .I0(carry_0),
        .I1(b[1]),
        .I2(a[1]),
        .I3(1'b0)
    );

    // bit 2
    CFGLUT5 #(
        .INIT(32'h96)
    )
    CFGLUT5_cout_inst_2 (
        .O5(out_reg[2]),
        .CDI(data_in[4]), // Use data_in for write mode
        .CE(reconfig_ce),
        .CLK(ap_clk),
        .I0(carry_1),
        .I1(b[2]),
        .I2(a[2]),
        .I3(1'b0)
    );

    CFGLUT5 #(
        .INIT(32'he8)
    )
    CFGLUT5_carry_inst_2 (
        .O5(carry_2),
        .CLK(ap_clk),
        .CDI(data_in[5]), // Use data_in for write mode
        .CE(reconfig_ce),
        .I0(carry_1),
        .I1(b[2]),
        .I2(a[2]),
        .I3(1'b0)
    );

    // bit 3
    CFGLUT5 #(
        .INIT(32'h96)
    )
    CFGLUT5_cout_inst_3 (
        .O5(out_reg[3]),
        .CLK(ap_clk),
        .CDI(data_in[6]), // Use data_in for write mode
        .CE(reconfig_ce),
        .I0(carry_2),
        .I1(b[3]),
        .I2(a[3]),
        .I3(1'b0)
    );

    CFGLUT5 #(
        .INIT(32'he8)
    )
    CFGLUT5_carry_inst_3 (
        .O5(carry),
        .CDI(data_in[7]), // Use data_in for write mode
        .CE(reconfig_ce),
        .CLK(ap_clk),
        .I0(carry_2),
        .I1(b[3]),
        .I2(a[3]),
        .I3(1'b0)
    );

    // Output assignment
    always @(posedge ap_clk or posedge ap_rst) begin
        if (ap_rst) begin
            sum <= 4'b0000;
        end else if (reconfig_ce == 1'b0 & read_enable) begin
            sum <= out_reg;  // read mode: update sum with the output register
        end
    end

endmodule

// Testbench for the 4-bit adder
module tb_adder_4bit;
    
    // Testbench signals
    reg ap_clk;
    reg ap_rst;
    reg reconfig_ce;
    reg read_enable;
    reg [7:0] data_in;
    reg [3:0] a, b;
    wire [3:0] sum;
    wire carry;
    
    // Instantiate the adder
    adder_store_unit uut (
        .ap_clk(ap_clk),
        .ap_rst(ap_rst),
        .reconfig_ce(reconfig_ce),
        .read_enable(read_enable),
        .data_in(data_in),
        .a(a),
        .b(b),
        .sum(sum),
        .carry(carry)
    );
    
    // Clock generation
    initial begin
        ap_clk = 0;
        forever #5 ap_clk = ~ap_clk;
    end
    
    // Test stimulus
    initial begin
        $display("Testing 4-bit Adder");
        $display("Time\t a\t b\t sum\t carry");
        $monitor("%0t\t %b\t %b\t %b\t %b", $time, a, b, sum, carry);
        
        // Initialize signals
        ap_rst = 1;
        reconfig_ce = 0;
        read_enable = 1;
        data_in = 8'h00;
        a = 4'b0000;
        b = 4'b0000;
        
        // Release reset
        #20 ap_rst = 0;
        
        // Test cases in read mode (reconfig_ce = 0)
        #10 a = 4'b0000; b = 4'b0000; #20;  // 0 + 0 = 0
        a = 4'b0001; b = 4'b0001; #20;  // 1 + 1 = 2
        a = 4'b0011; b = 4'b0101; #20;  // 3 + 5 = 8
        a = 4'b1111; b = 4'b0001; #20;  // 15 + 1 = 0 (overflow)
        a = 4'b1010; b = 4'b0101; #20;  // 10 + 5 = 15
        a = 4'b0111; b = 4'b0111; #20;  // 7 + 7 = 14
        a = 4'b1111; b = 4'b1111; #20;  // 15 + 15 = 14 (overflow)
        
        $display("start writing data...");
        
        #20 reconfig_ce = 1; read_enable = 0;
        #10 data_in = 8'b00010101; #10;
        data_in = 8'b00010001; #10;
        data_in = 8'b00000101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00000101; #10;
        data_in = 8'b00000101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b01000101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010001; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b00010101; #10;
        data_in = 8'b01010101; #10;
        data_in = 8'b01010101; #10;
        reconfig_ce = 0; #40;
        a = 4'b0000; b = 4'b0000; #40;
        read_enable = 1; #10;
        read_enable = 0; #10;
        a = 4'b1111; b = 4'b0000; #40;
        read_enable = 1; #10;
        read_enable = 0; #10;
        a = 4'b0000; b = 4'b1111; #40;
        read_enable = 1; #10;
        read_enable = 0; #10;
        
        #50 $finish;
    end
    
endmodule
