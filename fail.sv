module dynamic_checker (
    input wire clk,
    input wire rst,
    input wire bus_adr,
    input wire bus_ren,
    input wire bus_wdt,
    input wire bus_wen,
    input wire owr_i,
    output wire bus_irq,
    output wire bus_rdt,
    output wire owr_e,
    output wire owr_p
);
    wire bus_irq;
    wire bus_rdt;
    wire owr_e;
    wire owr_p;
    arbiter DUT (
        .clk(clk),
        .rst(rst),
        .bus_adr(bus_adr),
        .bus_ren(bus_ren),
        .bus_wdt(bus_wdt),
        .bus_wen(bus_wen),
        .owr_i(owr_i),
        .bus_irq(bus_irq),
        .bus_rdt(bus_rdt),
        .owr_e(owr_e),
        .owr_p(owr_p)
    );
    reg initialized = 0;
    always @(posedge clk) begin
        initialized <= 1;
        assume(rst == !initialized);
        if (initialized) begin
            if (bus_wen && bus_adr[4]) begin
                assert(owr_p == $past(owr_p, 1));
            end
        end
    end
endmodule