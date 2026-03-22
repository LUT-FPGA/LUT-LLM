
proc concat_axi_pins { cell } {
    set pins [get_bd_intf_pins -of $cell]
    set result []

    foreach pin $pins {
        set last_word [lindex [split $pin /] end]
        lappend result $last_word
    }

    set final_result [join $result :]
    return $final_result
}

proc get_bd_clk_pins { cell } {
    set result [get_bd_pins -of $cell -filter {TYPE == clk}]
    return $result
}

proc get_bd_rst_pins { cell } {
    set result [get_bd_pins -of $cell -filter {TYPE == rst}]
    return $result
}



# Create block design
set top_bd_file [get_files top_arm.bd]
if {[llength $top_bd_file] > 0} {
    remove_files $top_bd_file
}
create_bd_design "top_arm"
update_compile_order -fileset sources_1

# Create instance: CIPS_0
set CIPS_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips:3.4 CIPS_0 ]


# Set CIPS properties
set_property -dict [list CONFIG.CLOCK_MODE {Custom} CONFIG.DDR_MEMORY_MODE {Custom} CONFIG.PS_BOARD_INTERFACE {Custom} CONFIG.PS_PL_CONNECTIVITY_MODE {Custom} CONFIG.PS_PMC_CONFIG {     BOOT_MODE {Custom}     CLOCK_MODE {Custom}     DESIGN_MODE {1}     DEVICE_INTEGRITY_MODE {Sysmon temperature voltage and external IO monitoring}     PMC_CRP_PL0_REF_CTRL_FREQMHZ {99.9992}     PMC_GPIO0_MIO_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 0 .. 25}}}     PMC_GPIO1_MIO_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 26 .. 51}}}     PMC_MIO12 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}         {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}}     PMC_MIO37 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA high}         {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}}     PMC_OSPI_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 0 .. 11}} {MODE Single}}     PMC_QSPI_PERIPHERAL_ENABLE {0}     PMC_REF_CLK_FREQMHZ {33.333}     PMC_SD1 {{CD_ENABLE 1} {CD_IO {PMC_MIO 28}} {POW_ENABLE 1} {POW_IO {PMC_MIO 51}}         {RESET_ENABLE 0} {RESET_IO {PMC_MIO 12}} {WP_ENABLE 0} {WP_IO {PMC_MIO 1}}}     PMC_SD1_PERIPHERAL {{CLK_100_SDR_OTAP_DLY 0x3} {CLK_200_SDR_OTAP_DLY 0x2}         {CLK_50_DDR_ITAP_DLY 0x2A} {CLK_50_DDR_OTAP_DLY 0x3} {CLK_50_SDR_ITAP_DLY 0x25}        {CLK_50_SDR_OTAP_DLY 0x4} {ENABLE 1} {IO {PMC_MIO 26 .. 36}}}     PMC_SD1_SLOT_TYPE {SD 3.0 AUTODIR}     PMC_USE_PMC_NOC_AXI0 {1}     PS_BOARD_INTERFACE {ps_pmc_fixed_io}     PS_ENET0_MDIO {{ENABLE 1} {IO {PS_MIO 24 .. 25}}}     PS_ENET0_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 0 .. 11}}}     PS_GEN_IPI0_ENABLE {1}     PS_GEN_IPI0_MASTER {A72}     PS_GEN_IPI1_ENABLE {1}     PS_GEN_IPI2_ENABLE {1}     PS_GEN_IPI3_ENABLE {1}     PS_GEN_IPI4_ENABLE {1}     PS_GEN_IPI5_ENABLE {1}     PS_GEN_IPI6_ENABLE {1}     PS_I2C0_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 46 .. 47}}}     PS_I2C1_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 44 .. 45}}}     PS_I2CSYSMON_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 39 .. 40}}}     PS_IRQ_USAGE {{CH0 0} {CH1 0} {CH10 0} {CH11 0} {CH12 0} {CH13 0} {CH14 0}         {CH15 0} {CH2 0} {CH3 0} {CH4 0} {CH5 0} {CH6 0} {CH7 0} {CH8 1} {CH9 0}}     PS_MIO7 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}         {PULL disable} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}     PS_MIO9 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}         {PULL disable} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}     PS_NUM_FABRIC_RESETS {1}     PS_PCIE_EP_RESET1_IO {PS_MIO 18}     PS_PCIE_EP_RESET2_IO {PS_MIO 19}     PS_PCIE_RESET {ENABLE 1}     PS_UART0_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 42 .. 43}}}     PS_USB3_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 13 .. 25}}}     PS_USE_FPD_AXI_NOC0 {1}     PS_USE_FPD_AXI_NOC1 {1}     PS_USE_FPD_CCI_NOC {1}     PS_USE_M_AXI_FPD {1}     PS_USE_NOC_LPD_AXI0 {1}     PS_USE_PMCPL_CLK0 {1}     SMON_ALARMS {Set_Alarms_On}     SMON_ENABLE_TEMP_AVERAGING {0}     SMON_INTERFACE_TO_USE {I2C}     SMON_PMBUS_ADDRESS {0x18}     SMON_TEMP_AVERAGING_SAMPLES {0}     } ] [get_bd_cells CIPS_0]


# set_property -dict [list CONFIG.PS_PMC_CONFIG { PMC_CRP_PL0_REF_CTRL_FREQMHZ {400.00}} ] $CIPS_0

set_property CONFIG.PS_PMC_CONFIG {PS_USE_M_AXI_FPD {0}} $CIPS_0

# Create instance: cips_noc, and set properties
set cips_noc [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 cips_noc ]
set_property -dict [list     CONFIG.NUM_CLKS {9}     CONFIG.NUM_MI {0}     CONFIG.NUM_NMI {1}     CONFIG.NUM_NSI {0}     CONFIG.NUM_SI {8} ] $cips_noc

set_property CONFIG.ASSOCIATED_BUSIF {S00_AXI} [get_bd_pins /cips_noc/aclk0]
set_property CONFIG.ASSOCIATED_BUSIF {S01_AXI} [get_bd_pins /cips_noc/aclk1]
set_property CONFIG.ASSOCIATED_BUSIF {S02_AXI} [get_bd_pins /cips_noc/aclk2]
set_property CONFIG.ASSOCIATED_BUSIF {S03_AXI} [get_bd_pins /cips_noc/aclk3]
set_property CONFIG.ASSOCIATED_BUSIF {S04_AXI} [get_bd_pins /cips_noc/aclk4]
set_property CONFIG.ASSOCIATED_BUSIF {S05_AXI} [get_bd_pins /cips_noc/aclk5]
set_property CONFIG.ASSOCIATED_BUSIF {S06_AXI} [get_bd_pins /cips_noc/aclk6]
set_property CONFIG.ASSOCIATED_BUSIF {S07_AXI} [get_bd_pins /cips_noc/aclk7]

# Create interface connections
connect_bd_intf_net -intf_net CIPS_0_FPD_AXI_NOC_0     [get_bd_intf_pins CIPS_0/FPD_AXI_NOC_0] [get_bd_intf_pins cips_noc/S04_AXI]
connect_bd_intf_net -intf_net CIPS_0_FPD_AXI_NOC_1     [get_bd_intf_pins CIPS_0/FPD_AXI_NOC_1] [get_bd_intf_pins cips_noc/S05_AXI]
connect_bd_intf_net -intf_net CIPS_0_FPD_CCI_NOC_0     [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_0] [get_bd_intf_pins cips_noc/S00_AXI]
connect_bd_intf_net -intf_net CIPS_0_FPD_CCI_NOC_1     [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_1] [get_bd_intf_pins cips_noc/S01_AXI]
connect_bd_intf_net -intf_net CIPS_0_FPD_CCI_NOC_2     [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_2] [get_bd_intf_pins cips_noc/S02_AXI]
connect_bd_intf_net -intf_net CIPS_0_FPD_CCI_NOC_3     [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_3] [get_bd_intf_pins cips_noc/S03_AXI]
connect_bd_intf_net -intf_net CIPS_0_LPD_AXI_NOC_0     [get_bd_intf_pins CIPS_0/LPD_AXI_NOC_0] [get_bd_intf_pins cips_noc/S06_AXI]
connect_bd_intf_net -intf_net CIPS_0_PMC_NOC_AXI_0     [get_bd_intf_pins CIPS_0/PMC_NOC_AXI_0] [get_bd_intf_pins cips_noc/S07_AXI]

# Create port connections
connect_bd_net [get_bd_pins CIPS_0/fpd_axi_noc_axi0_clk] [get_bd_pins cips_noc/aclk4]
connect_bd_net [get_bd_pins CIPS_0/fpd_axi_noc_axi1_clk] [get_bd_pins cips_noc/aclk5]
connect_bd_net [get_bd_pins CIPS_0/fpd_cci_noc_axi0_clk] [get_bd_pins cips_noc/aclk0]
connect_bd_net [get_bd_pins CIPS_0/fpd_cci_noc_axi1_clk] [get_bd_pins cips_noc/aclk1]
connect_bd_net [get_bd_pins CIPS_0/fpd_cci_noc_axi2_clk] [get_bd_pins cips_noc/aclk2]
connect_bd_net [get_bd_pins CIPS_0/fpd_cci_noc_axi3_clk] [get_bd_pins cips_noc/aclk3]
connect_bd_net [get_bd_pins CIPS_0/lpd_axi_noc_clk] [get_bd_pins cips_noc/aclk6]
connect_bd_net [get_bd_pins CIPS_0/pmc_axi_noc_axi0_clk]     [get_bd_pins cips_noc/aclk7]


# Create instance: axi_intc_0, and set properties
set axi_intc_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 axi_intc_0 ]
set_property -dict [list     CONFIG.C_ASYNC_INTR {0xFFFFFFFF}     CONFIG.C_IRQ_CONNECTION {1} ] $axi_intc_0

# Create instance: clk_wizard_0, and set properties
# set clk_wizard_0 [ create_bd_cell -type ip #     -vlnv xilinx.com:ip:clk_wizard:1.0 clk_wizard_0 ]
# set_property -dict [list #     CONFIG.CLKOUT_DRIVES {BUFG,BUFG,BUFG,BUFG,BUFG,BUFG,BUFG} #     CONFIG.CLKOUT_DYN_PS {None,None,None,None,None,None,None} #     CONFIG.CLKOUT_MATCHED_ROUTING {false,false,false,false,false,false,false} #     CONFIG.CLKOUT_PORT {clk_out1,clk_out2,clk_out3,clk_out4,clk_out5,clk_out6,clk_out7}#     CONFIG.CLKOUT_REQUESTED_DUTY_CYCLE #         {50.000,50.000,50.000,50.000,50.000,50.000,50.000} #     CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY #         {300.000,250.000,200.000,100.000,100.000,100.000,100.000} #     CONFIG.CLKOUT_REQUESTED_PHASE {0.000,0.000,0.000,0.000,0.000,0.000,0.000} #     CONFIG.CLKOUT_USED {true,false,false,false,false,false,false} #     CONFIG.JITTER_SEL {Min_O_Jitter} #     CONFIG.PRIM_SOURCE {No_buffer} #     CONFIG.RESET_TYPE {ACTIVE_LOW} #     CONFIG.USE_LOCKED {true} #     CONFIG.USE_PHASE_ALIGNMENT {true} #     CONFIG.USE_RESET {true} # ] $clk_wizard_0


set clk_wizard_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wizard:1.0 clk_wizard_0 ]
set_property -dict [list \
  CONFIG.CLKOUT_DRIVES {BUFG,BUFG,BUFG,BUFG,BUFG,BUFG,BUFG} \
  CONFIG.CLKOUT_DYN_PS {None,None,None,None,None,None,None} \
  CONFIG.CLKOUT_GROUPING {Auto,Auto,Auto,Auto,Auto,Auto,Auto} \
  CONFIG.CLKOUT_MATCHED_ROUTING {false,false,false,false,false,false,false} \
  CONFIG.CLKOUT_PORT {clk_out1,clk_out2,clk_out3,clk_out4,clk_out5,clk_out6,clk_out7} \
  CONFIG.CLKOUT_REQUESTED_DUTY_CYCLE {50.000,50.000,50.000,50.000,50.000,50.000,50.000} \
  CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {250.000,100.000,100.000,100.000,100.000,100.000,100.000} \
  CONFIG.CLKOUT_REQUESTED_PHASE {0.000,0.000,0.000,0.000,0.000,0.000,0.000} \
  CONFIG.CLKOUT_USED {true,false,false,false,false,false,false} \
  CONFIG.RESET_TYPE {ACTIVE_LOW} \
  CONFIG.USE_LOCKED {true} \
  CONFIG.USE_PHASE_ALIGNMENT {true} \
  CONFIG.USE_RESET {true} \
] [get_bd_cells clk_wizard_0]


# Create instance: proc_sys_reset_0, and set properties
set proc_sys_reset_0 [ create_bd_cell -type ip     -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0 ]

# Create instance: icn_ctrl, and set properties
set icn_ctrl [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 icn_ctrl ]
set_property -dict [list     CONFIG.NUM_CLKS {1}     CONFIG.NUM_MI {2}     CONFIG.NUM_SI {1} ] $icn_ctrl

# Create interface connections
connect_bd_intf_net -intf_net icn_ctrl_M00_AXI     [get_bd_intf_pins axi_intc_0/s_axi] [get_bd_intf_pins icn_ctrl/M00_AXI]

# Create port connections
connect_bd_net -net axi_intc_0_irq     [get_bd_pins axi_intc_0/irq] [get_bd_pins CIPS_0/pl_ps_irq8]
connect_bd_net -net proc_sys_reset_0_peripheral_aresetn     [get_bd_pins proc_sys_reset_0/peripheral_aresetn]     [get_bd_pins icn_ctrl/aresetn] [get_bd_pins axi_intc_0/s_axi_aresetn]

# with clk_wizard
connect_bd_net [get_bd_pins CIPS_0/pl0_ref_clk] [get_bd_pins clk_wizard_0/clk_in1]
connect_bd_net [get_bd_pins CIPS_0/pl0_resetn] [get_bd_pins clk_wizard_0/resetn] [get_bd_pins proc_sys_reset_0/ext_reset_in]
connect_bd_net -net clk_wizard_0_clk_out1 [get_bd_pins clk_wizard_0/clk_out1] [get_bd_pins axi_intc_0/s_axi_aclk] [get_bd_pins icn_ctrl/aclk] [get_bd_pins proc_sys_reset_0/slowest_sync_clk]
connect_bd_net -net clk_wizard_0_locked [get_bd_pins clk_wizard_0/locked] [get_bd_pins proc_sys_reset_0/dcm_locked]

# no clk_wizard
# connect_bd_net [get_bd_pins CIPS_0/pl0_resetn]     [get_bd_pins proc_sys_reset_0/ext_reset_in]
# connect_bd_net [get_bd_pins CIPS_0/pl0_ref_clk]     [get_bd_pins axi_intc_0/s_axi_aclk] [get_bd_pins icn_ctrl/aclk]     [get_bd_pins proc_sys_reset_0/slowest_sync_clk]


set_property CONFIG.NUM_MI {1} $cips_noc
set_property -dict [ list     CONFIG.CONNECTIONS {         M00_AXI { read_bw {0} write_bw {1}}         M00_INI { read_bw {1} write_bw {0}} }     CONFIG.CATEGORY {ps_rpu} ] [get_bd_intf_pins /cips_noc/S06_AXI]

connect_bd_intf_net -intf_net CIPS_0_M_AXI_GP0     [get_bd_intf_pins /cips_noc/M00_AXI] [get_bd_intf_pins icn_ctrl/S00_AXI]
# connect_bd_net [get_bd_pins CIPS_0/pl0_ref_clk] [get_bd_pins /cips_noc/aclk8]
connect_bd_net [get_bd_pins clk_wizard_0/clk_out1] [get_bd_pins /cips_noc/aclk8]
set_property CONFIG.ASSOCIATED_BUSIF M00_AXI [get_bd_pins /cips_noc/aclk8]


set_property -dict [list     CONFIG.NUM_NMI {1} ] $cips_noc


set_property -dict [ list     CONFIG.CONNECTIONS {M00_INI { read_bw {1} write_bw {0} }}     CONFIG.CATEGORY {ps_cci} ] [get_bd_intf_pins /cips_noc/S00_AXI]

set_property -dict [ list     CONFIG.CONNECTIONS {M00_INI { read_bw {1} write_bw {0} }}     CONFIG.CATEGORY {ps_cci} ] [get_bd_intf_pins /cips_noc/S01_AXI]

set_property -dict [ list     CONFIG.CONNECTIONS {M00_INI { read_bw {1} write_bw {0} }}     CONFIG.CATEGORY {ps_cci} ] [get_bd_intf_pins /cips_noc/S02_AXI]

set_property -dict [ list     CONFIG.CONNECTIONS {M00_INI { read_bw {1} write_bw {0} }}     CONFIG.CATEGORY {ps_cci} ] [get_bd_intf_pins /cips_noc/S03_AXI]

set_property -dict [ list     CONFIG.CONNECTIONS {M00_INI { read_bw {1} write_bw {0} }}     CONFIG.CATEGORY {ps_nci} ] [get_bd_intf_pins /cips_noc/S04_AXI]

set_property -dict [ list     CONFIG.CONNECTIONS {M00_INI { read_bw {1} write_bw {0} }}     CONFIG.CATEGORY {ps_nci} ] [get_bd_intf_pins /cips_noc/S05_AXI]

set_property -dict [ list     CONFIG.CONNECTIONS {M00_INI { read_bw {1} write_bw {0} }}     CONFIG.CATEGORY {ps_pmc} ] [get_bd_intf_pins /cips_noc/S07_AXI]


set_property -dict [ list     CONFIG.CONNECTIONS {M00_AXI { read_bw {0} write_bw {1}}                         M00_INI { read_bw {1} write_bw {0} } }     CONFIG.CATEGORY {ps_rpu} ] [get_bd_intf_pins /cips_noc/S06_AXI]


# Create instance: axi_noc_dut, and set properties
set axi_noc_dut [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 axi_noc_dut ]
set_property -dict [list     CONFIG.HBM_NUM_CHNL {16}     CONFIG.NUM_CLKS {1}     CONFIG.NUM_HBM_BLI {0}     CONFIG.NUM_MI {0}     CONFIG.NUM_NSI {1} ] $axi_noc_dut


set_property -dict [list CONFIG.CONNECTIONS {
    HBM15_PORT0 {read_bw {5} write_bw {0} read_avg_burst {4} write_avg_burst {4}}
}] [get_bd_intf_pins $axi_noc_dut/S00_INI]

connect_bd_intf_net     [get_bd_intf_pins cips_noc/M00_INI] [get_bd_intf_pins axi_noc_dut/S00_INI]


# ======================= Adding DUT =======================

startgroup
# Add RTL module to BD
set dut [create_bd_cell -type module -reference qwen_block dut_0]

# Associate AXI interfaces to clock
# Assumes there is one clock and all AXI pins use the same clock
set_property CONFIG.ASSOCIATED_BUSIF [concat_axi_pins $dut] [get_bd_clk_pins $dut]
endgroup


startgroup

set_property -dict [list     CONFIG.NUM_HBM_BLI {55}     CONFIG.NUM_SI {0}     CONFIG.NUM_CLKS {1}     CONFIG.HBM_MEM_BACKDOOR_WRITE {false}     CONFIG.HBM_MEM_INIT_FILE {no_file_loaded}] $axi_noc_dut


set_property -dict [list CONFIG.CONNECTIONS {HBM0_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {0} }}] [get_bd_intf_pins /axi_noc_dut/HBM00_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_22]     [get_bd_intf_pins /axi_noc_dut/HBM00_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM0_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {1} }}] [get_bd_intf_pins /axi_noc_dut/HBM01_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_23]     [get_bd_intf_pins /axi_noc_dut/HBM01_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM0_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {2} }}] [get_bd_intf_pins /axi_noc_dut/HBM02_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_24]     [get_bd_intf_pins /axi_noc_dut/HBM02_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM0_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {3} }}] [get_bd_intf_pins /axi_noc_dut/HBM03_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_25]     [get_bd_intf_pins /axi_noc_dut/HBM03_AXI]


set_property -dict [list CONFIG.CONNECTIONS {HBM1_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {4} }}] [get_bd_intf_pins /axi_noc_dut/HBM04_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_26]     [get_bd_intf_pins /axi_noc_dut/HBM04_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM1_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {5} }}] [get_bd_intf_pins /axi_noc_dut/HBM05_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_27]     [get_bd_intf_pins /axi_noc_dut/HBM05_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM1_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {6} }}] [get_bd_intf_pins /axi_noc_dut/HBM06_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_28]     [get_bd_intf_pins /axi_noc_dut/HBM06_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM1_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {7} }}] [get_bd_intf_pins /axi_noc_dut/HBM07_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_29]     [get_bd_intf_pins /axi_noc_dut/HBM07_AXI]


set_property -dict [list CONFIG.CONNECTIONS {HBM2_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {8} }}] [get_bd_intf_pins /axi_noc_dut/HBM08_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_30]     [get_bd_intf_pins /axi_noc_dut/HBM08_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM2_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {9} }}] [get_bd_intf_pins /axi_noc_dut/HBM09_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_31]     [get_bd_intf_pins /axi_noc_dut/HBM09_AXI]


set_property -dict [list CONFIG.CONNECTIONS {HBM2_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {10} }}] [get_bd_intf_pins /axi_noc_dut/HBM10_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_v_cache_buffer_0]     [get_bd_intf_pins /axi_noc_dut/HBM10_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM3_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {11} }}] [get_bd_intf_pins /axi_noc_dut/HBM11_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_v_cache_buffer_1]     [get_bd_intf_pins /axi_noc_dut/HBM11_AXI]


set_property -dict [list CONFIG.CONNECTIONS {HBM3_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {12} }}] [get_bd_intf_pins /axi_noc_dut/HBM12_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_input_buffer_0]     [get_bd_intf_pins /axi_noc_dut/HBM12_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM3_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {13} }}] [get_bd_intf_pins /axi_noc_dut/HBM13_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_input_buffer_1]     [get_bd_intf_pins /axi_noc_dut/HBM13_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM3_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {14} }}] [get_bd_intf_pins /axi_noc_dut/HBM14_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_input_buffer_2]     [get_bd_intf_pins /axi_noc_dut/HBM14_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM4_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {15} }}] [get_bd_intf_pins /axi_noc_dut/HBM15_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_input_buffer_3]     [get_bd_intf_pins /axi_noc_dut/HBM15_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM4_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {16} }}] [get_bd_intf_pins /axi_noc_dut/HBM16_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_centroid_buffer_0]     [get_bd_intf_pins /axi_noc_dut/HBM16_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM4_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {17} }}] [get_bd_intf_pins /axi_noc_dut/HBM17_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_centroid_buffer_1]     [get_bd_intf_pins /axi_noc_dut/HBM17_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM4_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {18} }}] [get_bd_intf_pins /axi_noc_dut/HBM18_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_scale_zero_buffer]     [get_bd_intf_pins /axi_noc_dut/HBM18_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM5_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {19} }}] [get_bd_intf_pins /axi_noc_dut/HBM19_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_sin_buffer_0]     [get_bd_intf_pins /axi_noc_dut/HBM19_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM5_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {20} }}] [get_bd_intf_pins /axi_noc_dut/HBM20_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_sin_buffer_1]     [get_bd_intf_pins /axi_noc_dut/HBM20_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM5_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {21} }}] [get_bd_intf_pins /axi_noc_dut/HBM21_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_cos_buffer_0]     [get_bd_intf_pins /axi_noc_dut/HBM21_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM5_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {22} }}] [get_bd_intf_pins /axi_noc_dut/HBM22_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_cos_buffer_1]     [get_bd_intf_pins /axi_noc_dut/HBM22_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM6_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {23} }}] [get_bd_intf_pins /axi_noc_dut/HBM23_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_rms_norm_weight_buffer_0]     [get_bd_intf_pins /axi_noc_dut/HBM23_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM6_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {24} }}] [get_bd_intf_pins /axi_noc_dut/HBM24_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_rms_norm_weight_buffer_1]     [get_bd_intf_pins /axi_noc_dut/HBM24_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM6_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {25} }}] [get_bd_intf_pins /axi_noc_dut/HBM25_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_out_buffer_0]     [get_bd_intf_pins /axi_noc_dut/HBM25_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM6_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {26} }}] [get_bd_intf_pins /axi_noc_dut/HBM26_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_out_buffer_1]     [get_bd_intf_pins /axi_noc_dut/HBM26_AXI]


set_property -dict [list CONFIG.CONNECTIONS {HBM7_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {27} }}] [get_bd_intf_pins /axi_noc_dut/HBM27_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_out_buffer_2]     [get_bd_intf_pins /axi_noc_dut/HBM27_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM7_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {28} }}] [get_bd_intf_pins /axi_noc_dut/HBM28_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_out_buffer_3]     [get_bd_intf_pins /axi_noc_dut/HBM28_AXI]


set_property -dict [list CONFIG.CONNECTIONS {HBM8_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {29} }}] [get_bd_intf_pins /axi_noc_dut/HBM29_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_0]     [get_bd_intf_pins /axi_noc_dut/HBM29_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM8_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {30} }}] [get_bd_intf_pins /axi_noc_dut/HBM30_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_1]     [get_bd_intf_pins /axi_noc_dut/HBM30_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM8_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {31} }}] [get_bd_intf_pins /axi_noc_dut/HBM31_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_2]     [get_bd_intf_pins /axi_noc_dut/HBM31_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM8_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {32} }}] [get_bd_intf_pins /axi_noc_dut/HBM32_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_3]     [get_bd_intf_pins /axi_noc_dut/HBM32_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM9_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {33} }}] [get_bd_intf_pins /axi_noc_dut/HBM33_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_4]     [get_bd_intf_pins /axi_noc_dut/HBM33_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM9_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {34} }}] [get_bd_intf_pins /axi_noc_dut/HBM34_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_5]     [get_bd_intf_pins /axi_noc_dut/HBM34_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM9_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {35} }}] [get_bd_intf_pins /axi_noc_dut/HBM35_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_6]     [get_bd_intf_pins /axi_noc_dut/HBM35_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM9_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {36} }}] [get_bd_intf_pins /axi_noc_dut/HBM36_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_7]     [get_bd_intf_pins /axi_noc_dut/HBM36_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM10_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {37} }}] [get_bd_intf_pins /axi_noc_dut/HBM37_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_8]     [get_bd_intf_pins /axi_noc_dut/HBM37_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM10_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {38} }}] [get_bd_intf_pins /axi_noc_dut/HBM38_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_9]     [get_bd_intf_pins /axi_noc_dut/HBM38_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM10_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {39} }}] [get_bd_intf_pins /axi_noc_dut/HBM39_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_10]     [get_bd_intf_pins /axi_noc_dut/HBM39_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM10_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {40} }}] [get_bd_intf_pins /axi_noc_dut/HBM40_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_11]     [get_bd_intf_pins /axi_noc_dut/HBM40_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM11_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {41} }}] [get_bd_intf_pins /axi_noc_dut/HBM41_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_12]     [get_bd_intf_pins /axi_noc_dut/HBM41_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM11_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {42} }}] [get_bd_intf_pins /axi_noc_dut/HBM42_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_13]     [get_bd_intf_pins /axi_noc_dut/HBM42_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM11_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {43} }}] [get_bd_intf_pins /axi_noc_dut/HBM43_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_14]     [get_bd_intf_pins /axi_noc_dut/HBM43_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM11_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {44} }}] [get_bd_intf_pins /axi_noc_dut/HBM44_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_15]     [get_bd_intf_pins /axi_noc_dut/HBM44_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM12_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {45} }}] [get_bd_intf_pins /axi_noc_dut/HBM45_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_16]     [get_bd_intf_pins /axi_noc_dut/HBM45_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM12_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {46} }}] [get_bd_intf_pins /axi_noc_dut/HBM46_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_17]     [get_bd_intf_pins /axi_noc_dut/HBM46_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM12_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {47} }}] [get_bd_intf_pins /axi_noc_dut/HBM47_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_18]     [get_bd_intf_pins /axi_noc_dut/HBM47_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM12_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {48} }}] [get_bd_intf_pins /axi_noc_dut/HBM48_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_19]     [get_bd_intf_pins /axi_noc_dut/HBM48_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM13_PORT0 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {49} }}] [get_bd_intf_pins /axi_noc_dut/HBM49_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_20]     [get_bd_intf_pins /axi_noc_dut/HBM49_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM13_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {50} }}] [get_bd_intf_pins /axi_noc_dut/HBM50_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_lut_weight_idx_buffer_21]     [get_bd_intf_pins /axi_noc_dut/HBM50_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM7_PORT3 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {51} }}] [get_bd_intf_pins /axi_noc_dut/HBM51_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_k_cache_buffer_0]     [get_bd_intf_pins /axi_noc_dut/HBM51_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM14_PORT1 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {52} }}] [get_bd_intf_pins /axi_noc_dut/HBM52_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_k_cache_buffer_1]     [get_bd_intf_pins /axi_noc_dut/HBM52_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM14_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {53} }}] [get_bd_intf_pins /axi_noc_dut/HBM53_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_centroid_buffer_2]     [get_bd_intf_pins /axi_noc_dut/HBM53_AXI]

set_property -dict [list CONFIG.CONNECTIONS {HBM7_PORT2 {     read_bw {2000}     write_bw {2000}     read_avg_burst {4}     write_avg_burst {4} \
    # excl_group {} \
    sep_rt_group {54} }}] [get_bd_intf_pins /axi_noc_dut/HBM54_AXI]
connect_bd_intf_net [get_bd_intf_pins $dut/m_axi_centroid_buffer_3]     [get_bd_intf_pins /axi_noc_dut/HBM54_AXI]

set_property -dict [list CONFIG.ASSOCIATED_BUSIF {HBM00_AXI:HBM01_AXI:HBM02_AXI:HBM03_AXI:HBM04_AXI:HBM05_AXI:HBM06_AXI:HBM07_AXI:HBM08_AXI:HBM09_AXI:HBM10_AXI:HBM11_AXI:HBM12_AXI:HBM13_AXI:HBM14_AXI:HBM15_AXI:HBM16_AXI:HBM17_AXI:HBM18_AXI:HBM19_AXI:HBM20_AXI:HBM21_AXI:HBM22_AXI:HBM23_AXI:HBM24_AXI:HBM25_AXI:HBM26_AXI:HBM27_AXI:HBM28_AXI:HBM29_AXI:HBM30_AXI:HBM31_AXI:HBM32_AXI:HBM33_AXI:HBM34_AXI:HBM35_AXI:HBM36_AXI:HBM37_AXI:HBM38_AXI:HBM39_AXI:HBM40_AXI:HBM41_AXI:HBM42_AXI:HBM43_AXI:HBM44_AXI:HBM45_AXI:HBM46_AXI:HBM47_AXI:HBM48_AXI:HBM49_AXI:HBM50_AXI:HBM51_AXI:HBM52_AXI:HBM53_AXI:HBM54_AXI}]             [get_bd_pins /axi_noc_dut/aclk0]
endgroup

# Create external clk and reset ports for simulation
set pl0_ref_clk_0 [ create_bd_port -dir O -type clk pl0_ref_clk_0 ]
# connect_bd_net [get_bd_pins CIPS_0/pl0_ref_clk] [get_bd_ports pl0_ref_clk_0]
connect_bd_net [get_bd_pins clk_wizard_0/clk_out1] [get_bd_ports pl0_ref_clk_0]

# connect_bd_net [get_bd_pins clk_wizard_0/clk_out1] [get_bd_clk_pins $dut] #     [get_bd_pins $axi_noc_dut/aclk0]
# connect_bd_net [get_bd_pins CIPS_0/pl0_ref_clk] [get_bd_clk_pins $dut]     [get_bd_pins $axi_noc_dut/aclk0]
connect_bd_net [get_bd_pins clk_wizard_0/clk_out1] [get_bd_clk_pins $dut]     [get_bd_pins $axi_noc_dut/aclk0]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_rst_pins $dut]
connect_bd_intf_net [get_bd_intf_pins icn_ctrl/M01_AXI]     [get_bd_intf_pins dut_0/s_axi_control]
connect_bd_net [get_bd_pins dut_0/interrupt] [get_bd_pins axi_intc_0/intr]
# connect_bd_net [get_bd_pins CIPS_0/pl0_ref_clk] [get_bd_pins CIPS_0/m_axi_fpd_aclk]
connect_bd_net [get_bd_pins clk_wizard_0/clk_out1] [get_bd_pins CIPS_0/m_axi_fpd_aclk]

# https://support.xilinx.com/s/article/000036160?language=en_US


# Auto-assigns all
assign_bd_address
