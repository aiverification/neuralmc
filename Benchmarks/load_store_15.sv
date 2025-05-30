module Load_Store (input clk, input rst, output reg sig);
    localparam N = 200000;
    localparam CBITS = 18;
    reg [CBITS-1:0] vol;
    reg m;
    always @(posedge clk) begin
        if (rst) begin m = 0; vol = 0;
        end else begin
            if (m) begin
                if (vol >= N) m = 0; else vol = vol + 1;
            end else begin
                if (vol <= 0) m = 1; else vol = vol - 1;
            end 
        end 
        sig = (vol == N);
    end
endmodule