
# assign tb clk and reset to dummy pins for synthesis
set_property PACKAGE_PIN {BP53} [get_ports pl0_ref_clk_0]
set_property IOSTANDARD LVDCI_15 [get_ports pl0_ref_clk_0]

# Initialize an empty list to store undefined cells to verify floorplan correctness
set undefined_cells {}
