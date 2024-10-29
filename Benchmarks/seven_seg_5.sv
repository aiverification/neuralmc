module SEVEN(input clk, input rst, input [13:0] both7seg, output reg[6:0] segment);
	localparam freq = 2500;
	localparam CBITS = 12;

	reg [CBITS-1:0] cnt;
	reg digit_select;

	always @(posedge clk) begin
		if(rst == 1) begin
			cnt = 0;
			digit_select = 0;
			segment = 0;
		end
		if(cnt < freq)
			cnt = cnt + 1;
		else begin
			cnt = 0;
			if(digit_select == 0) begin
				digit_select = 1;
				segment = both7seg[13:7];
			end
			else begin
				digit_select = 0;
				segment = both7seg[6:0];
			end
		end
	end
endmodule