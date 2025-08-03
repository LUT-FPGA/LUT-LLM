
################################################################
# This is a generated script based on design: top
#
# Though there are limitations about the generated script,
# the main purpose of this utility is to make learning
# IP Integrator Tcl commands easier.
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   set script_folder [file dirname $script_path]
   return $script_folder
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

################################################################
# Check if script is running in correct Vivado version.
################################################################
set scripts_vivado_version 2024.2
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   common::send_gid_msg -ssname BD::TCL -id 2040 -severity "CRITICAL WARNING" "This script was generated using Vivado <$scripts_vivado_version> without IP versions in the create_bd_cell commands, but is now being run in <$current_vivado_version> of Vivado. There may have been changes to the IP between Vivado <$scripts_vivado_version> and <$current_vivado_version>, which could impact the functionality and configuration of the design."

}

################################################################
# START
################################################################

# To test this script, run the following commands from Vivado Tcl console:
# source top_script.tcl


# The design that will be created by this Tcl script contains the following 
# module references:
# Serpens

# Please add the sources of those modules before sourcing this Tcl script.

set bCheckIPsPassed 1
##################################################################
# CHECK IPs
##################################################################
set bCheckIPs 1
if { $bCheckIPs == 1 } {
   set list_check_ips "\ 
xilinx.com:ip:versal_cips:*\
xilinx.com:ip:axi_noc:*\
xilinx.com:ip:clk_wizard:*\
xilinx.com:ip:smartconnect:*\
xilinx.com:ip:proc_sys_reset:*\
xilinx.com:ip:axis_ila:*\
xilinx.com:ip:hw_discovery:*\
xilinx.com:ip:shell_utils_uuid_rom:*\
xilinx.com:ip:smbus:*\
xilinx.com:ip:cmd_queue:*\
xilinx.com:ip:axi_gpio:*\
xilinx.com:ip:util_vector_logic:*\
xilinx.com:ip:xlconcat:*\
xilinx.com:ip:util_reduced_logic:*\
"

   set list_ips_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2011 -severity "INFO" "Checking if the following IPs exist in the project's IP catalog: $list_check_ips ."

   foreach ip_vlnv $list_check_ips {
      set ip_obj [get_ipdefs -all $ip_vlnv]
      if { $ip_obj eq "" } {
         lappend list_ips_missing $ip_vlnv
      }
   }

   if { $list_ips_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2012 -severity "ERROR" "The following IPs are not found in the IP Catalog:\n  $list_ips_missing\n\nResolution: Please add the repository containing the IP(s) to the project." }
      set bCheckIPsPassed 0
   }

}

### import rtl

proc import_ips_from_dir {dir} {
    # Get a list of all .xci files in the specified directory and its subdirectories
    foreach file [glob -nocomplain -directory $dir *] {
        if {[file isdirectory $file]} {
            set ip_file [glob -nocomplain -directory $file *.xci]
            puts "Importing IP: $ip_file"
            import_ip $ip_file
        }
    }
}

# rename folder if needed
import_ips_from_dir rtl/
import_files rtl/

upgrade_ip -quiet [get_ips *]
generate_target synthesis [ get_files *.xci ]

##################################################################
# CHECK Modules
##################################################################
set bCheckModules 1
if { $bCheckModules == 1 } {
   set list_check_mods "\ 
Serpens\
"

   set list_mods_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2020 -severity "INFO" "Checking if the following modules exist in the project's sources: $list_check_mods ."

   foreach mod_vlnv $list_check_mods {
      if { [can_resolve_reference $mod_vlnv] == 0 } {
         lappend list_mods_missing $mod_vlnv
      }
   }

   if { $list_mods_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2021 -severity "ERROR" "The following module(s) are not found in the project: $list_mods_missing" }
      common::send_gid_msg -ssname BD::TCL -id 2022 -severity "INFO" "Please add source files for the missing module(s) above."
      set bCheckIPsPassed 0
   }
}

if { $bCheckIPsPassed != 1 } {
  common::send_gid_msg -ssname BD::TCL -id 2023 -severity "WARNING" "Will not continue with creation of design due to the error(s) above."
  return 3
}

##################################################################
# DESIGN PROCs
##################################################################


# Hierarchical cell: pcie_mgmt_pdi_reset
proc create_hier_cell_pcie_mgmt_pdi_reset { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_pcie_mgmt_pdi_reset() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi


  # Create pins
  create_bd_pin -dir I -type clk clk
  create_bd_pin -dir I -type rst resetn
  create_bd_pin -dir I -type rst resetn_in

  # Create instance: pcie_mgmt_pdi_reset_gpio, and set properties
  set pcie_mgmt_pdi_reset_gpio [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio pcie_mgmt_pdi_reset_gpio ]
  set_property -dict [list \
    CONFIG.C_ALL_INPUTS_2 {1} \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_DOUT_DEFAULT {0x00000000} \
    CONFIG.C_GPIO2_WIDTH {1} \
    CONFIG.C_GPIO_WIDTH {1} \
    CONFIG.C_IS_DUAL {1} \
  ] $pcie_mgmt_pdi_reset_gpio


  # Create instance: inv, and set properties
  set inv [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic inv ]
  set_property -dict [list \
    CONFIG.C_OPERATION {not} \
    CONFIG.C_SIZE {1} \
  ] $inv


  # Create instance: ccat, and set properties
  set ccat [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat ccat ]

  # Create instance: and_0, and set properties
  set and_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_reduced_logic and_0 ]
  set_property CONFIG.C_SIZE {2} $and_0


  # Create interface connections
  connect_bd_intf_net -intf_net s_axi_1 [get_bd_intf_pins s_axi] [get_bd_intf_pins pcie_mgmt_pdi_reset_gpio/S_AXI]

  # Create port connections
  connect_bd_net -net and_0_Res  [get_bd_pins and_0/Res] \
  [get_bd_pins pcie_mgmt_pdi_reset_gpio/gpio2_io_i]
  connect_bd_net -net ccat_dout  [get_bd_pins ccat/dout] \
  [get_bd_pins and_0/Op1]
  connect_bd_net -net clk_1  [get_bd_pins clk] \
  [get_bd_pins pcie_mgmt_pdi_reset_gpio/s_axi_aclk]
  connect_bd_net -net inv_Res  [get_bd_pins inv/Res] \
  [get_bd_pins ccat/In1]
  connect_bd_net -net pcie_mgmt_pdi_reset_gpio_gpio_io_o  [get_bd_pins pcie_mgmt_pdi_reset_gpio/gpio_io_o] \
  [get_bd_pins ccat/In0]
  connect_bd_net -net resetn_1  [get_bd_pins resetn] \
  [get_bd_pins pcie_mgmt_pdi_reset_gpio/s_axi_aresetn]
  connect_bd_net -net resetn_in_1  [get_bd_pins resetn_in] \
  [get_bd_pins inv/Op1]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: base_logic
proc create_hier_cell_base_logic { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_base_logic() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_pcie_mgmt_slr0

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_rpu

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:pcie3_cfg_ext_rtl:1.0 pcie_cfg_ext

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 smbus_rpu

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 m_axi_pcie_mgmt_pdi_reset


  # Create pins
  create_bd_pin -dir I -type clk clk_pcie
  create_bd_pin -dir I -type clk clk_pl
  create_bd_pin -dir I -type rst resetn_pcie_periph
  create_bd_pin -dir I -type rst resetn_pl_periph
  create_bd_pin -dir I -type rst resetn_pl_ic
  create_bd_pin -dir O -type intr irq_gcq_m2r
  create_bd_pin -dir O -type intr irq_axi_smbus_rpu

  # Create instance: pcie_slr0_mgmt_sc, and set properties
  set pcie_slr0_mgmt_sc [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect pcie_slr0_mgmt_sc ]
  set_property -dict [list \
    CONFIG.NUM_CLKS {1} \
    CONFIG.NUM_MI {4} \
    CONFIG.NUM_SI {1} \
  ] $pcie_slr0_mgmt_sc


  # Create instance: rpu_sc, and set properties
  set rpu_sc [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect rpu_sc ]
  set_property -dict [list \
    CONFIG.NUM_CLKS {1} \
    CONFIG.NUM_MI {2} \
    CONFIG.NUM_SI {1} \
  ] $rpu_sc


  # Create instance: hw_discovery, and set properties
  set hw_discovery [ create_bd_cell -type ip -vlnv xilinx.com:ip:hw_discovery hw_discovery ]
  set_property -dict [list \
    CONFIG.C_CAP_BASE_ADDR {0x600} \
    CONFIG.C_INJECT_ENDPOINTS {0} \
    CONFIG.C_MANUAL {1} \
    CONFIG.C_NEXT_CAP_ADDR {0x000} \
    CONFIG.C_NUM_PFS {1} \
    CONFIG.C_PF0_BAR_INDEX {0} \
    CONFIG.C_PF0_ENDPOINT_NAMES {0} \
    CONFIG.C_PF0_ENTRY_ADDR_0 {0x000001001000} \
    CONFIG.C_PF0_ENTRY_ADDR_1 {0x000001010000} \
    CONFIG.C_PF0_ENTRY_ADDR_2 {0x000008000000} \
    CONFIG.C_PF0_ENTRY_BAR_0 {0} \
    CONFIG.C_PF0_ENTRY_BAR_1 {0} \
    CONFIG.C_PF0_ENTRY_BAR_2 {0} \
    CONFIG.C_PF0_ENTRY_MAJOR_VERSION_0 {1} \
    CONFIG.C_PF0_ENTRY_MAJOR_VERSION_1 {1} \
    CONFIG.C_PF0_ENTRY_MAJOR_VERSION_2 {1} \
    CONFIG.C_PF0_ENTRY_MINOR_VERSION_0 {0} \
    CONFIG.C_PF0_ENTRY_MINOR_VERSION_1 {2} \
    CONFIG.C_PF0_ENTRY_MINOR_VERSION_2 {0} \
    CONFIG.C_PF0_ENTRY_RSVD0_0 {0x0} \
    CONFIG.C_PF0_ENTRY_RSVD0_1 {0x0} \
    CONFIG.C_PF0_ENTRY_RSVD0_2 {0x0} \
    CONFIG.C_PF0_ENTRY_TYPE_0 {0x50} \
    CONFIG.C_PF0_ENTRY_TYPE_1 {0x54} \
    CONFIG.C_PF0_ENTRY_TYPE_2 {0x55} \
    CONFIG.C_PF0_ENTRY_VERSION_TYPE_0 {0x01} \
    CONFIG.C_PF0_ENTRY_VERSION_TYPE_1 {0x01} \
    CONFIG.C_PF0_ENTRY_VERSION_TYPE_2 {0x01} \
    CONFIG.C_PF0_HIGH_OFFSET {0x00000000} \
    CONFIG.C_PF0_LOW_OFFSET {0x0100000} \
    CONFIG.C_PF0_NUM_SLOTS_BAR_LAYOUT_TABLE {3} \
    CONFIG.C_PF0_S_AXI_ADDR_WIDTH {32} \
  ] $hw_discovery


  # Create instance: uuid_rom, and set properties
  set uuid_rom [ create_bd_cell -type ip -vlnv xilinx.com:ip:shell_utils_uuid_rom uuid_rom ]
  set_property CONFIG.C_INITIAL_UUID {00000000000000000000000000000000} $uuid_rom


  # Create instance: axi_smbus_rpu, and set properties
  set axi_smbus_rpu [ create_bd_cell -type ip -vlnv xilinx.com:ip:smbus axi_smbus_rpu ]
  set_property -dict [list \
    CONFIG.NUM_TARGET_DEVICES {8} \
    CONFIG.SMBUS_DEV_CLASS {0} \
  ] $axi_smbus_rpu


  # Create instance: gcq_m2r, and set properties
  set gcq_m2r [ create_bd_cell -type ip -vlnv xilinx.com:ip:cmd_queue gcq_m2r ]

  # Create interface connections
  connect_bd_intf_net -intf_net axi_smbus_rpu_SMBUS [get_bd_intf_pins axi_smbus_rpu/SMBUS] [get_bd_intf_pins smbus_rpu]
  connect_bd_intf_net -intf_net pcie_cfg_ext_1 [get_bd_intf_pins pcie_cfg_ext] [get_bd_intf_pins hw_discovery/s_pcie4_cfg_ext]
  connect_bd_intf_net -intf_net pcie_slr0_mgmt_sc_M00_AXI [get_bd_intf_pins pcie_slr0_mgmt_sc/M00_AXI] [get_bd_intf_pins hw_discovery/s_axi_ctrl_pf0]
  connect_bd_intf_net -intf_net pcie_slr0_mgmt_sc_M01_AXI [get_bd_intf_pins pcie_slr0_mgmt_sc/M01_AXI] [get_bd_intf_pins uuid_rom/S_AXI]
  connect_bd_intf_net -intf_net pcie_slr0_mgmt_sc_M02_AXI [get_bd_intf_pins pcie_slr0_mgmt_sc/M02_AXI] [get_bd_intf_pins gcq_m2r/S00_AXI]
  connect_bd_intf_net -intf_net pcie_slr0_mgmt_sc_M03_AXI [get_bd_intf_pins pcie_slr0_mgmt_sc/M03_AXI] [get_bd_intf_pins m_axi_pcie_mgmt_pdi_reset]
  connect_bd_intf_net -intf_net rpu_sc_M00_AXI [get_bd_intf_pins rpu_sc/M00_AXI] [get_bd_intf_pins gcq_m2r/S01_AXI]
  connect_bd_intf_net -intf_net rpu_sc_M01_AXI [get_bd_intf_pins axi_smbus_rpu/S_AXI] [get_bd_intf_pins rpu_sc/M01_AXI]
  connect_bd_intf_net -intf_net s_axi_pcie_mgmt_slr0_1 [get_bd_intf_pins s_axi_pcie_mgmt_slr0] [get_bd_intf_pins pcie_slr0_mgmt_sc/S00_AXI]
  connect_bd_intf_net -intf_net s_axi_rpu_1 [get_bd_intf_pins s_axi_rpu] [get_bd_intf_pins rpu_sc/S00_AXI]

  # Create port connections
  connect_bd_net -net axi_smbus_rpu_ip2intc_irpt  [get_bd_pins axi_smbus_rpu/ip2intc_irpt] \
  [get_bd_pins irq_axi_smbus_rpu]
  connect_bd_net -net clk_pcie_1  [get_bd_pins clk_pcie] \
  [get_bd_pins hw_discovery/aclk_pcie]
  connect_bd_net -net clk_pl_1  [get_bd_pins clk_pl] \
  [get_bd_pins pcie_slr0_mgmt_sc/aclk] \
  [get_bd_pins rpu_sc/aclk] \
  [get_bd_pins hw_discovery/aclk_ctrl] \
  [get_bd_pins uuid_rom/S_AXI_ACLK] \
  [get_bd_pins gcq_m2r/aclk] \
  [get_bd_pins axi_smbus_rpu/s_axi_aclk]
  connect_bd_net -net gcq_m2r_irq_sq  [get_bd_pins gcq_m2r/irq_sq] \
  [get_bd_pins irq_gcq_m2r]
  connect_bd_net -net resetn_pcie_periph_1  [get_bd_pins resetn_pcie_periph] \
  [get_bd_pins hw_discovery/aresetn_pcie]
  connect_bd_net -net resetn_pl_ic_1  [get_bd_pins resetn_pl_ic] \
  [get_bd_pins pcie_slr0_mgmt_sc/aresetn] \
  [get_bd_pins rpu_sc/aresetn]
  connect_bd_net -net resetn_pl_periph_1  [get_bd_pins resetn_pl_periph] \
  [get_bd_pins hw_discovery/aresetn_ctrl] \
  [get_bd_pins uuid_rom/S_AXI_ARESETN] \
  [get_bd_pins gcq_m2r/aresetn] \
  [get_bd_pins axi_smbus_rpu/s_axi_aresetn]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: clock_reset
proc create_hier_cell_clock_reset { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_clock_reset() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_pcie_mgmt_pdi_reset


  # Create pins
  create_bd_pin -dir I -type clk clk_pl
  create_bd_pin -dir I -type clk clk_freerun
  create_bd_pin -dir I -type clk clk_pcie
  create_bd_pin -dir I -type rst dma_axi_aresetn
  create_bd_pin -dir I -type rst resetn_pl_axi
  create_bd_pin -dir O -from 0 -to 0 -type rst resetn_pcie_ic
  create_bd_pin -dir O -from 0 -to 0 -type rst resetn_pcie_periph
  create_bd_pin -dir O -from 0 -to 0 -type rst resetn_pl_ic
  create_bd_pin -dir O -from 0 -to 0 -type rst resetn_pl_periph
  create_bd_pin -dir O -type clk clk_usr_0
  create_bd_pin -dir O -from 0 -to 0 -type rst resetn_usr_0_ic
  create_bd_pin -dir O -from 0 -to 0 -type rst resetn_usr_0_periph
  create_bd_pin -dir O -type clk clk_usr_1
  create_bd_pin -dir O -from 0 -to 0 -type rst resetn_usr_1_ic
  create_bd_pin -dir O -from 0 -to 0 -type rst resetn_usr_1_periph

  # Create instance: pcie_psr, and set properties
  set pcie_psr [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset pcie_psr ]
  set_property CONFIG.C_EXT_RST_WIDTH {1} $pcie_psr


  # Create instance: pl_psr, and set properties
  set pl_psr [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset pl_psr ]
  set_property CONFIG.C_EXT_RST_WIDTH {1} $pl_psr


  # Create instance: usr_clk_wiz, and set properties
  set usr_clk_wiz [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wizard usr_clk_wiz ]
  set_property -dict [list \
    CONFIG.CLKOUT_DRIVES {No_buffer,No_buffer} \
    CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {300,500} \
    CONFIG.CLKOUT_USED {true,true} \
    CONFIG.PRIM_SOURCE {No_buffer} \
    CONFIG.USE_DYN_RECONFIG {false} \
    CONFIG.USE_LOCKED {true} \
    CONFIG.USE_POWER_DOWN {false} \
    CONFIG.USE_RESET {false} \
  ] $usr_clk_wiz


  # Create instance: usr_0_psr, and set properties
  set usr_0_psr [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset usr_0_psr ]
  set_property CONFIG.C_EXT_RST_WIDTH {1} $usr_0_psr


  # Create instance: usr_1_psr, and set properties
  set usr_1_psr [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset usr_1_psr ]
  set_property CONFIG.C_EXT_RST_WIDTH {1} $usr_1_psr


  # Create instance: pcie_mgmt_pdi_reset
  create_hier_cell_pcie_mgmt_pdi_reset $hier_obj pcie_mgmt_pdi_reset

  # Create interface connections
  connect_bd_intf_net -intf_net s_axi_pcie_mgmt_pdi_reset_1 [get_bd_intf_pins s_axi_pcie_mgmt_pdi_reset] [get_bd_intf_pins pcie_mgmt_pdi_reset/s_axi]

  # Create port connections
  connect_bd_net -net clk_freerun_1  [get_bd_pins clk_freerun] \
  [get_bd_pins usr_clk_wiz/clk_in1]
  connect_bd_net -net clk_pcie_1  [get_bd_pins clk_pcie] \
  [get_bd_pins pcie_psr/slowest_sync_clk]
  connect_bd_net -net clk_pl_1  [get_bd_pins clk_pl] \
  [get_bd_pins pl_psr/slowest_sync_clk] \
  [get_bd_pins pcie_mgmt_pdi_reset/clk]
  connect_bd_net -net dma_axi_aresetn_1  [get_bd_pins dma_axi_aresetn] \
  [get_bd_pins pcie_mgmt_pdi_reset/resetn_in]
  connect_bd_net -net pcie_psr_interconnect_aresetn  [get_bd_pins pcie_psr/interconnect_aresetn] \
  [get_bd_pins resetn_pcie_ic]
  connect_bd_net -net pcie_psr_peripheral_aresetn  [get_bd_pins pcie_psr/peripheral_aresetn] \
  [get_bd_pins resetn_pcie_periph]
  connect_bd_net -net pl_psr_interconnect_aresetn  [get_bd_pins pl_psr/interconnect_aresetn] \
  [get_bd_pins resetn_pl_ic] \
  [get_bd_pins pcie_psr/ext_reset_in] \
  [get_bd_pins usr_0_psr/ext_reset_in] \
  [get_bd_pins usr_1_psr/ext_reset_in]
  connect_bd_net -net pl_psr_peripheral_aresetn  [get_bd_pins pl_psr/peripheral_aresetn] \
  [get_bd_pins resetn_pl_periph] \
  [get_bd_pins pcie_mgmt_pdi_reset/resetn]
  connect_bd_net -net resetn_pl_axi_1  [get_bd_pins resetn_pl_axi] \
  [get_bd_pins pl_psr/ext_reset_in]
  connect_bd_net -net usr_0_psr_interconnect_aresetn  [get_bd_pins usr_0_psr/interconnect_aresetn] \
  [get_bd_pins resetn_usr_0_ic]
  connect_bd_net -net usr_0_psr_peripheral_aresetn  [get_bd_pins usr_0_psr/peripheral_aresetn] \
  [get_bd_pins resetn_usr_0_periph]
  connect_bd_net -net usr_1_psr_interconnect_aresetn  [get_bd_pins usr_1_psr/interconnect_aresetn] \
  [get_bd_pins resetn_usr_1_ic]
  connect_bd_net -net usr_1_psr_peripheral_aresetn  [get_bd_pins usr_1_psr/peripheral_aresetn] \
  [get_bd_pins resetn_usr_1_periph]
  connect_bd_net -net usr_clk_wiz_clk_out1  [get_bd_pins usr_clk_wiz/clk_out1] \
  [get_bd_pins clk_usr_0] \
  [get_bd_pins usr_0_psr/slowest_sync_clk]
  connect_bd_net -net usr_clk_wiz_clk_out2  [get_bd_pins usr_clk_wiz/clk_out2] \
  [get_bd_pins clk_usr_1] \
  [get_bd_pins usr_1_psr/slowest_sync_clk]
  connect_bd_net -net usr_clk_wiz_locked  [get_bd_pins usr_clk_wiz/locked] \
  [get_bd_pins usr_0_psr/dcm_locked] \
  [get_bd_pins usr_1_psr/dcm_locked]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: slash
proc create_hier_cell_slash { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_slash() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axilite

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI18

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI19

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI20

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI21

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI22

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI23

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI24

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI25

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI26

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI27

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI28

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI29

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI30

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI31

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI32

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI33

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI34

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI35

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI36

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI37

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI38

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI39

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI40

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI41

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI42

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI43

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI44

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI45

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI46

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI47

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI48

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI49

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI50

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI51

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI52

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI53

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI54

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI55

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI56

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI57

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI58

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI59

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI60

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI61

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI62

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI63

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI17

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI16

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI15

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI14

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI13

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI12

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI11

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI10

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI9

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI8

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI7

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI6

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI5

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI4

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI3

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI2

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI01

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI00


  # Create pins
  create_bd_pin -dir I -type clk s_axi_aclk
  create_bd_pin -dir I -type rst arstn
  create_bd_pin -dir O -type clk clk_out1

  # Create instance: clk_wizard_0, and set properties
  set clk_wizard_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wizard clk_wizard_0 ]
  set_property -dict [list \
    CONFIG.CLKOUT_DRIVES {BUFG,BUFG,BUFG,BUFG,BUFG,BUFG,BUFG} \
    CONFIG.CLKOUT_DYN_PS {None,None,None,None,None,None,None} \
    CONFIG.CLKOUT_GROUPING {Auto,Auto,Auto,Auto,Auto,Auto,Auto} \
    CONFIG.CLKOUT_MATCHED_ROUTING {false,false,false,false,false,false,false} \
    CONFIG.CLKOUT_PORT {clk_out1,clk_out2,clk_out3,clk_out4,clk_out5,clk_out6,clk_out7} \
    CONFIG.CLKOUT_REQUESTED_DUTY_CYCLE {50.000,50.000,50.000,50.000,50.000,50.000,50.000} \
    CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {400,100.000,100.000,100.000,100.000,100.000,100.000} \
    CONFIG.CLKOUT_REQUESTED_PHASE {0.000,0.000,0.000,0.000,0.000,0.000,0.000} \
    CONFIG.CLKOUT_USED {true,false,false,false,false,false,false} \
    CONFIG.RESET_TYPE {ACTIVE_LOW} \
    CONFIG.USE_DYN_RECONFIG {false} \
  ] $clk_wizard_0


  # Create instance: smartconnect_0, and set properties
  set smartconnect_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smartconnect_0 ]
  set_property -dict [list \
    CONFIG.NUM_CLKS {2} \
    CONFIG.NUM_MI {1} \
  ] $smartconnect_0


  # Create instance: proc_sys_reset_1, and set properties
  set proc_sys_reset_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset proc_sys_reset_1 ]

  # Create instance: axis_ila_0, and set properties
  set axis_ila_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_ila axis_ila_0 ]
  set_property -dict [list \
    CONFIG.C_MON_TYPE {Interface_Monitor} \
    CONFIG.C_NUM_MONITOR_SLOTS {4} \
  ] $axis_ila_0


  # Create instance: dut_0, and set properties
  set block_name Serpens
  set block_cell_name dut_0
  if { [catch {set dut_0 [create_bd_cell -type module -reference $block_name $block_cell_name] } errmsg] } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2095 -severity "ERROR" "Unable to add referenced block <$block_name>. Please add the files for ${block_name}'s definition into the project."}
     return 1
   } elseif { $dut_0 eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2096 -severity "ERROR" "Unable to referenced block <$block_name>. Please add the files for ${block_name}'s definition into the project."}
     return 1
   }
  
  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {m_axi_edge_list_ch_0:m_axi_edge_list_ch_1:m_axi_edge_list_ch_10:m_axi_edge_list_ch_11:m_axi_edge_list_ch_12:m_axi_edge_list_ch_13:m_axi_edge_list_ch_14:m_axi_edge_list_ch_15:m_axi_edge_list_ch_16:m_axi_edge_list_ch_17:m_axi_edge_list_ch_18:m_axi_edge_list_ch_19:m_axi_edge_list_ch_2:m_axi_edge_list_ch_20:m_axi_edge_list_ch_21:m_axi_edge_list_ch_22:m_axi_edge_list_ch_23:m_axi_edge_list_ch_24:m_axi_edge_list_ch_25:m_axi_edge_list_ch_26:m_axi_edge_list_ch_27:m_axi_edge_list_ch_28:m_axi_edge_list_ch_29:m_axi_edge_list_ch_3:m_axi_edge_list_ch_30:m_axi_edge_list_ch_31:m_axi_edge_list_ch_4:m_axi_edge_list_ch_5:m_axi_edge_list_ch_6:m_axi_edge_list_ch_7:m_axi_edge_list_ch_8:m_axi_edge_list_ch_9:m_axi_edge_list_ptr:m_axi_vec_X:m_axi_vec_Y:m_axi_vec_Y_out:s_axi_control} \
 ] [get_bd_pins /slash/dut_0/ap_clk]

  # Create interface connections
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_0 [get_bd_intf_pins M_AXI00] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_0]
  connect_bd_intf_net -intf_net [get_bd_intf_nets dut_0_m_axi_edge_list_ch_0] [get_bd_intf_pins M_AXI00] [get_bd_intf_pins axis_ila_0/SLOT_0_AXI]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_1 [get_bd_intf_pins M_AXI01] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_1]
  connect_bd_intf_net -intf_net [get_bd_intf_nets dut_0_m_axi_edge_list_ch_1] [get_bd_intf_pins M_AXI01] [get_bd_intf_pins axis_ila_0/SLOT_1_AXI]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_2 [get_bd_intf_pins M_AXI12] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_2]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_3 [get_bd_intf_pins M_AXI23] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_3]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_4 [get_bd_intf_pins M_AXI26] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_4]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_5 [get_bd_intf_pins M_AXI27] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_5]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_6 [get_bd_intf_pins M_AXI28] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_6]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_7 [get_bd_intf_pins M_AXI29] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_7]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_8 [get_bd_intf_pins M_AXI30] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_8]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_9 [get_bd_intf_pins M_AXI31] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_9]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_10 [get_bd_intf_pins M_AXI2] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_10]
  connect_bd_intf_net -intf_net [get_bd_intf_nets dut_0_m_axi_edge_list_ch_10] [get_bd_intf_pins M_AXI2] [get_bd_intf_pins axis_ila_0/SLOT_2_AXI]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_11 [get_bd_intf_pins M_AXI3] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_11]
  connect_bd_intf_net -intf_net [get_bd_intf_nets dut_0_m_axi_edge_list_ch_11] [get_bd_intf_pins M_AXI3] [get_bd_intf_pins axis_ila_0/SLOT_3_AXI]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_12 [get_bd_intf_pins M_AXI4] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_12]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_13 [get_bd_intf_pins M_AXI5] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_13]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_14 [get_bd_intf_pins M_AXI6] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_14]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_15 [get_bd_intf_pins M_AXI7] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_15]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_16 [get_bd_intf_pins M_AXI8] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_16]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_17 [get_bd_intf_pins M_AXI9] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_17]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_18 [get_bd_intf_pins M_AXI10] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_18]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_19 [get_bd_intf_pins M_AXI11] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_19]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_20 [get_bd_intf_pins M_AXI13] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_20]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_21 [get_bd_intf_pins M_AXI14] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_21]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_22 [get_bd_intf_pins M_AXI15] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_22]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_23 [get_bd_intf_pins M_AXI16] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_23]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_24 [get_bd_intf_pins M_AXI17] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_24]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_25 [get_bd_intf_pins M_AXI18] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_25]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_26 [get_bd_intf_pins M_AXI19] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_26]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_27 [get_bd_intf_pins M_AXI20] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_27]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_28 [get_bd_intf_pins M_AXI21] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_28]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_29 [get_bd_intf_pins M_AXI22] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_29]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_30 [get_bd_intf_pins M_AXI24] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_30]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ch_31 [get_bd_intf_pins M_AXI25] [get_bd_intf_pins dut_0/m_axi_edge_list_ch_31]
  connect_bd_intf_net -intf_net dut_0_m_axi_edge_list_ptr [get_bd_intf_pins M_AXI32] [get_bd_intf_pins dut_0/m_axi_edge_list_ptr]
  connect_bd_intf_net -intf_net dut_0_m_axi_vec_X [get_bd_intf_pins M_AXI33] [get_bd_intf_pins dut_0/m_axi_vec_X]
  connect_bd_intf_net -intf_net dut_0_m_axi_vec_Y [get_bd_intf_pins M_AXI34] [get_bd_intf_pins dut_0/m_axi_vec_Y]
  connect_bd_intf_net -intf_net dut_0_m_axi_vec_Y_out [get_bd_intf_pins M_AXI35] [get_bd_intf_pins dut_0/m_axi_vec_Y_out]
  connect_bd_intf_net -intf_net s_axilite_1 [get_bd_intf_pins s_axilite] [get_bd_intf_pins smartconnect_0/S00_AXI]
  connect_bd_intf_net -intf_net smartconnect_0_M00_AXI [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins dut_0/s_axi_control]

  # Create port connections
  connect_bd_net -net clk_wizard_0_clk_out1  [get_bd_pins clk_wizard_0/clk_out1] \
  [get_bd_pins clk_out1] \
  [get_bd_pins smartconnect_0/aclk1] \
  [get_bd_pins proc_sys_reset_1/slowest_sync_clk] \
  [get_bd_pins axis_ila_0/clk] \
  [get_bd_pins dut_0/ap_clk]
  connect_bd_net -net proc_sys_reset_0_peripheral_aresetn  [get_bd_pins proc_sys_reset_1/peripheral_aresetn] \
  [get_bd_pins axis_ila_0/resetn] \
  [get_bd_pins dut_0/ap_rst_n]
  connect_bd_net -net reset_rtl_0_1  [get_bd_pins arstn] \
  [get_bd_pins smartconnect_0/aresetn] \
  [get_bd_pins proc_sys_reset_1/ext_reset_in]
  connect_bd_net -net s_axi_aclk_1  [get_bd_pins s_axi_aclk] \
  [get_bd_pins clk_wizard_0/clk_in1] \
  [get_bd_pins smartconnect_0/aclk]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: noc
proc create_hier_cell_noc { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_noc() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 CH0_DDR4_0_1

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 sys_clk0_1

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 sys_clk0_0

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 CH0_DDR4_0_0

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S03_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 hbm_ref_clk_0

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 hbm_ref_clk_1

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S01_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S00_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S02_AXI

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M00_AXI

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M01_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM00_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM01_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM02_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM03_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM04_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM05_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM06_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM07_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM08_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM09_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM10_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM11_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM12_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM13_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM14_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM15_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM16_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM17_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM18_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM19_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM20_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM21_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM22_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM23_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM24_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM25_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM26_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM27_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM28_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM29_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM30_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM31_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM32_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM33_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM34_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM35_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM36_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM37_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM38_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM39_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM40_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM41_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM42_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM43_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM44_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM45_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM46_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM47_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM48_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM49_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM50_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM51_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM52_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM53_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM54_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM55_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM56_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM57_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM58_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM59_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM60_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM61_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM62_AXI

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 HBM63_AXI

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M02_AXI


  # Create pins
  create_bd_pin -dir I -type clk aclk0
  create_bd_pin -dir I -type clk aclk3
  create_bd_pin -dir I -type clk aclk1
  create_bd_pin -dir I -type clk aclk2
  create_bd_pin -dir I -type clk aclk4
  create_bd_pin -dir I -type clk aclk5
  create_bd_pin -dir I -type clk aclk6

  # Create instance: axi_noc_mc_ddr4_0, and set properties
  set axi_noc_mc_ddr4_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc axi_noc_mc_ddr4_0 ]
  set_property -dict [list \
    CONFIG.CONTROLLERTYPE {DDR4_SDRAM} \
    CONFIG.MC_CHAN_REGION1 {DDR_CH1} \
    CONFIG.MC_COMPONENT_WIDTH {x16} \
    CONFIG.MC_DATAWIDTH {72} \
    CONFIG.MC_DM_WIDTH {9} \
    CONFIG.MC_DQS_WIDTH {9} \
    CONFIG.MC_DQ_WIDTH {72} \
    CONFIG.MC_INIT_MEM_USING_ECC_SCRUB {true} \
    CONFIG.MC_INPUTCLK0_PERIOD {5000} \
    CONFIG.MC_MEMORY_DEVICETYPE {Components} \
    CONFIG.MC_MEMORY_SPEEDGRADE {DDR4-3200AA(22-22-22)} \
    CONFIG.MC_NO_CHANNELS {Single} \
    CONFIG.MC_RANK {1} \
    CONFIG.MC_ROWADDRESSWIDTH {16} \
    CONFIG.MC_STACKHEIGHT {1} \
    CONFIG.MC_SYSTEM_CLOCK {Differential} \
    CONFIG.NUM_CLKS {0} \
    CONFIG.NUM_MC {1} \
    CONFIG.NUM_MCP {4} \
    CONFIG.NUM_MI {0} \
    CONFIG.NUM_NMI {0} \
    CONFIG.NUM_NSI {2} \
    CONFIG.NUM_SI {0} \
  ] $axi_noc_mc_ddr4_0


  set_property -dict [ list \
   CONFIG.CONNECTIONS { MC_0 {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64} } } \
 ] [get_bd_intf_pins /noc/axi_noc_mc_ddr4_0/S00_INI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS { MC_1 {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64} } } \
 ] [get_bd_intf_pins /noc/axi_noc_mc_ddr4_0/S01_INI]

  # Create instance: axi_noc_cips, and set properties
  set axi_noc_cips [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc axi_noc_cips ]
  set_property -dict [list \
    CONFIG.HBM_CHNL0_CONFIG {HBM_REORDER_EN FALSE HBM_MAINTAIN_COHERENCY TRUE HBM_Q_AGE_LIMIT 0x7F HBM_CLOSE_PAGE_REORDER FALSE HBM_LOOKAHEAD_PCH TRUE HBM_COMMAND_PARITY FALSE HBM_DQ_WR_PARITY FALSE HBM_DQ_RD_PARITY\
FALSE HBM_RD_DBI TRUE HBM_WR_DBI TRUE HBM_REFRESH_MODE SINGLE_BANK_REFRESH HBM_PC0_PRE_DEFINED_ADDRESS_MAP USER_DEFINED_ADDRESS_MAP HBM_PC1_PRE_DEFINED_ADDRESS_MAP USER_DEFINED_ADDRESS_MAP HBM_PC0_USER_DEFINED_ADDRESS_MAP\
1BG-15RA-1SID-2BA-5CA-1BG HBM_PC1_USER_DEFINED_ADDRESS_MAP 1BG-15RA-1SID-2BA-5CA-1BG HBM_PC0_ADDRESS_MAP BA3,RA14,RA13,RA12,RA11,RA10,RA9,RA8,RA7,RA6,RA5,RA4,RA3,RA2,RA1,RA0,SID,BA1,BA0,CA5,CA4,CA3,CA2,CA1,BA2,NC,NA,NA,NA,NA\
HBM_PC1_ADDRESS_MAP BA3,RA14,RA13,RA12,RA11,RA10,RA9,RA8,RA7,RA6,RA5,RA4,RA3,RA2,RA1,RA0,SID,BA1,BA0,CA5,CA4,CA3,CA2,CA1,BA2,NC,NA,NA,NA,NA HBM_PWR_DWN_IDLE_TIMEOUT_ENTRY FALSE HBM_SELF_REF_IDLE_TIMEOUT_ENTRY\
FALSE HBM_IDLE_TIME_TO_ENTER_PWR_DWN_MODE 0x0001000 HBM_IDLE_TIME_TO_ENTER_SELF_REF_MODE 1X HBM_ECC_CORRECTION_EN FALSE HBM_WRITE_BACK_CORRECTED_DATA TRUE HBM_ECC_SCRUBBING FALSE HBM_ECC_INITIALIZE_EN\
FALSE HBM_ECC_SCRUB_SIZE 1092 HBM_WRITE_DATA_MASK TRUE HBM_REF_PERIOD_TEMP_COMP FALSE HBM_PARITY_LATENCY 3 HBM_PC0_PAGE_HIT 100.000 HBM_PC1_PAGE_HIT 100.000 HBM_PC0_READ_RATE 25.000 HBM_PC1_READ_RATE 25.000\
HBM_PC0_WRITE_RATE 25.000 HBM_PC1_WRITE_RATE 25.000 HBM_PC0_PHY_ACTIVE ENABLED HBM_PC1_PHY_ACTIVE ENABLED HBM_PC0_SCRUB_START_ADDRESS 0x0000000 HBM_PC0_SCRUB_END_ADDRESS 0x3FFFBFF HBM_PC0_SCRUB_INTERVAL\
24.000 HBM_PC1_SCRUB_START_ADDRESS 0x0000000 HBM_PC1_SCRUB_END_ADDRESS 0x3FFFBFF HBM_PC1_SCRUB_INTERVAL 24.000} \
    CONFIG.HBM_NUM_CHNL {16} \
    CONFIG.HBM_REF_CLK_FREQ0 {200.000} \
    CONFIG.HBM_REF_CLK_FREQ1 {200.000} \
    CONFIG.HBM_REF_CLK_SELECTION {External} \
    CONFIG.MI_NAMES {} \
    CONFIG.NUM_CLKS {7} \
    CONFIG.NUM_HBM_BLI {64} \
    CONFIG.NUM_MI {3} \
    CONFIG.NUM_NMI {4} \
    CONFIG.NUM_NSI {0} \
    CONFIG.NUM_SI {4} \
    CONFIG.SI_NAMES {} \
    CONFIG.SI_SIDEBAND_PINS { ,0,0,0} \
  ] $axi_noc_cips


  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X2Y0} \
   CONFIG.CONNECTIONS {HBM0_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM00_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X3Y0} \
   CONFIG.CONNECTIONS {HBM0_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM01_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X1Y0} \
   CONFIG.CONNECTIONS {HBM0_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM02_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X0Y0} \
   CONFIG.CONNECTIONS {HBM0_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM03_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X4Y0} \
   CONFIG.CONNECTIONS {HBM1_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM04_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X5Y0} \
   CONFIG.CONNECTIONS {HBM1_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM05_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X6Y0} \
   CONFIG.CONNECTIONS {HBM1_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM06_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X7Y0} \
   CONFIG.CONNECTIONS {HBM1_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM07_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X10Y0} \
   CONFIG.CONNECTIONS {HBM2_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM08_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X11Y0} \
   CONFIG.CONNECTIONS {HBM2_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM09_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X8Y0} \
   CONFIG.CONNECTIONS {HBM2_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM10_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X9Y0} \
   CONFIG.CONNECTIONS {HBM2_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM11_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X12Y0} \
   CONFIG.CONNECTIONS {HBM3_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM12_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X13Y0} \
   CONFIG.CONNECTIONS {HBM3_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM13_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X15Y0} \
   CONFIG.CONNECTIONS {HBM3_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM14_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X14Y0} \
   CONFIG.CONNECTIONS {HBM3_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM15_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X19Y0} \
   CONFIG.CONNECTIONS {HBM4_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM16_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X18Y0} \
   CONFIG.CONNECTIONS {HBM4_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM17_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X17Y0} \
   CONFIG.CONNECTIONS {HBM4_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM18_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X16Y0} \
   CONFIG.CONNECTIONS {HBM4_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM19_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X20Y0} \
   CONFIG.CONNECTIONS {HBM5_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM20_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X21Y0} \
   CONFIG.CONNECTIONS {HBM5_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM21_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X22Y0} \
   CONFIG.CONNECTIONS {HBM5_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM22_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X23Y0} \
   CONFIG.CONNECTIONS {HBM5_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM23_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X27Y0} \
   CONFIG.CONNECTIONS {HBM6_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM24_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X26Y0} \
   CONFIG.CONNECTIONS {HBM6_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM25_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X24Y0} \
   CONFIG.CONNECTIONS {HBM6_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM26_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X25Y0} \
   CONFIG.CONNECTIONS {HBM6_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM27_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X29Y0} \
   CONFIG.CONNECTIONS {HBM7_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM28_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X28Y0} \
   CONFIG.CONNECTIONS {HBM7_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM29_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X31Y0} \
   CONFIG.CONNECTIONS {HBM7_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM30_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X30Y0} \
   CONFIG.CONNECTIONS {HBM7_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM31_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X35Y0} \
   CONFIG.CONNECTIONS {HBM8_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM32_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X34Y0} \
   CONFIG.CONNECTIONS {HBM8_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM33_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X33Y0} \
   CONFIG.CONNECTIONS {HBM8_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM34_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X32Y0} \
   CONFIG.CONNECTIONS {HBM8_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM35_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X37Y0} \
   CONFIG.CONNECTIONS {HBM9_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM36_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X36Y0} \
   CONFIG.CONNECTIONS {HBM9_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM37_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X39Y0} \
   CONFIG.CONNECTIONS {HBM9_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM38_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X38Y0} \
   CONFIG.CONNECTIONS {HBM9_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM39_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X43Y0} \
   CONFIG.CONNECTIONS {HBM10_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM40_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X42Y0} \
   CONFIG.CONNECTIONS {HBM10_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM41_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X41Y0} \
   CONFIG.CONNECTIONS {HBM10_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM42_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X40Y0} \
   CONFIG.CONNECTIONS {HBM10_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM43_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X44Y0} \
   CONFIG.CONNECTIONS {HBM11_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM44_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X45Y0} \
   CONFIG.CONNECTIONS {HBM11_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM45_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X47Y0} \
   CONFIG.CONNECTIONS {HBM11_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM46_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X46Y0} \
   CONFIG.CONNECTIONS {HBM11_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM47_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X51Y0} \
   CONFIG.CONNECTIONS {HBM12_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM48_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X50Y0} \
   CONFIG.CONNECTIONS {HBM12_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM49_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X48Y0} \
   CONFIG.CONNECTIONS {HBM12_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM50_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X49Y0} \
   CONFIG.CONNECTIONS {HBM12_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM51_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X52Y0} \
   CONFIG.CONNECTIONS {HBM13_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM52_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X53Y0} \
   CONFIG.CONNECTIONS {HBM13_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM53_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X55Y0} \
   CONFIG.CONNECTIONS {HBM13_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM54_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X54Y0} \
   CONFIG.CONNECTIONS {HBM13_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM55_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X58Y0} \
   CONFIG.CONNECTIONS {HBM14_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM56_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X59Y0} \
   CONFIG.CONNECTIONS {HBM14_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM57_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X57Y0} \
   CONFIG.CONNECTIONS {HBM14_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM58_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X56Y0} \
   CONFIG.CONNECTIONS {HBM14_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM59_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X61Y0} \
   CONFIG.CONNECTIONS {HBM15_PORT0 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM60_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X60Y0} \
   CONFIG.CONNECTIONS {HBM15_PORT1 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM61_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X63Y0} \
   CONFIG.CONNECTIONS {HBM15_PORT2 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM62_AXI]

  set_property -dict [ list \
   CONFIG.PHYSICAL_LOC {NOC_NMU_HBM2E_X62Y0} \
   CONFIG.CONNECTIONS {HBM15_PORT3 {read_bw {2000} write_bw {2000} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {pl_hbm} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/HBM63_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.APERTURES {{0x201_0000_0000 0x200_0000}} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/M00_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.APERTURES {{0x202_0000_0000 0x800_0000}} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/M01_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CATEGORY {ps_pmc} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/M02_AXI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {HBM10_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M02_INI {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64}} HBM15_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM10_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M01_AXI {read_bw {100} write_bw {100} read_avg_burst {4} write_avg_burst {4}} HBM5_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM15_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM5_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM1_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM1_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM6_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM12_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM0_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM6_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM14_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM12_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM0_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M02_AXI {read_bw {500} write_bw {500} read_avg_burst {4} write_avg_burst {4}} HBM8_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM8_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM14_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM3_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM3_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM4_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM4_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM9_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM2_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM11_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M00_INI {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64}} HBM9_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM11_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM7_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM13_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM7_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM13_PORT0 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM2_PORT2 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M00_AXI {read_bw {5} write_bw {5} read_avg_burst {64} write_avg_burst {64}}} \
   CONFIG.DEST_IDS {M01_AXI:0x41:M02_AXI:0x1:M00_AXI:0x40} \
   CONFIG.REMAPS {M00_INI {{0x20108000000 0x00038000000 0x08000000}}} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_pcie} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/S00_AXI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {HBM10_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M01_AXI {read_bw {100} write_bw {100} read_avg_burst {4} write_avg_burst {4}} HBM10_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM5_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM15_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM0_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM15_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM1_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM5_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM1_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M01_INI {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64}} HBM0_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM6_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM8_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM14_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM12_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM6_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM12_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M02_AXI {read_bw {500} write_bw {500} read_avg_burst {4} write_avg_burst {4}} HBM8_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM14_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM3_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM3_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM4_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM9_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM4_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM9_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM11_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM11_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM7_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM13_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM7_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} HBM2_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M03_INI {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64}} HBM2_PORT1 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}} M00_AXI {read_bw {5} write_bw {5} read_avg_burst {64} write_avg_burst {64}} HBM13_PORT3 {read_bw {250} write_bw {250} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M01_AXI:0x41:M02_AXI:0x1:M00_AXI:0x40} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_pcie} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/S01_AXI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {M02_INI {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64}} M00_INI {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_pmc} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/S02_AXI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {M00_INI {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_rpu} \
 ] [get_bd_intf_pins /noc/axi_noc_cips/S03_AXI]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S00_AXI} \
 ] [get_bd_pins /noc/axi_noc_cips/aclk0]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S01_AXI} \
 ] [get_bd_pins /noc/axi_noc_cips/aclk1]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S02_AXI} \
 ] [get_bd_pins /noc/axi_noc_cips/aclk2]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S03_AXI} \
 ] [get_bd_pins /noc/axi_noc_cips/aclk3]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {M00_AXI:M01_AXI} \
 ] [get_bd_pins /noc/axi_noc_cips/aclk4]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {HBM00_AXI:HBM01_AXI:HBM02_AXI:HBM03_AXI:HBM04_AXI:HBM05_AXI:HBM06_AXI:HBM07_AXI:HBM08_AXI:HBM09_AXI:HBM10_AXI:HBM11_AXI:HBM12_AXI:HBM13_AXI:HBM14_AXI:HBM15_AXI:HBM16_AXI:HBM17_AXI:HBM18_AXI:HBM19_AXI:HBM20_AXI:HBM21_AXI:HBM22_AXI:HBM23_AXI:HBM24_AXI:HBM25_AXI:HBM26_AXI:HBM27_AXI:HBM28_AXI:HBM29_AXI:HBM30_AXI:HBM31_AXI:HBM32_AXI:HBM33_AXI:HBM34_AXI:HBM35_AXI:HBM36_AXI:HBM37_AXI:HBM38_AXI:HBM39_AXI:HBM40_AXI:HBM41_AXI:HBM42_AXI:HBM43_AXI:HBM44_AXI:HBM45_AXI:HBM46_AXI:HBM47_AXI:HBM48_AXI:HBM49_AXI:HBM50_AXI:HBM51_AXI:HBM52_AXI:HBM53_AXI:HBM54_AXI:HBM55_AXI:HBM56_AXI:HBM57_AXI:HBM58_AXI:HBM59_AXI:HBM60_AXI:HBM61_AXI:HBM62_AXI:HBM63_AXI} \
 ] [get_bd_pins /noc/axi_noc_cips/aclk5]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {M02_AXI} \
 ] [get_bd_pins /noc/axi_noc_cips/aclk6]

  # Create instance: axi_noc_mc_ddr4_1, and set properties
  set axi_noc_mc_ddr4_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc axi_noc_mc_ddr4_1 ]
  set_property -dict [list \
    CONFIG.CONTROLLERTYPE {DDR4_SDRAM} \
    CONFIG.MC0_CONFIG_NUM {config21} \
    CONFIG.MC0_FLIPPED_PINOUT {false} \
    CONFIG.MC_CHAN_REGION0 {DDR_CH2} \
    CONFIG.MC_COMPONENT_WIDTH {x4} \
    CONFIG.MC_DATAWIDTH {72} \
    CONFIG.MC_INIT_MEM_USING_ECC_SCRUB {true} \
    CONFIG.MC_INPUTCLK0_PERIOD {5000} \
    CONFIG.MC_MEMORY_DEVICETYPE {RDIMMs} \
    CONFIG.MC_MEMORY_SPEEDGRADE {DDR4-3200AA(22-22-22)} \
    CONFIG.MC_NO_CHANNELS {Single} \
    CONFIG.MC_PARITY {true} \
    CONFIG.MC_RANK {1} \
    CONFIG.MC_ROWADDRESSWIDTH {18} \
    CONFIG.MC_STACKHEIGHT {1} \
    CONFIG.MC_SYSTEM_CLOCK {Differential} \
    CONFIG.NUM_CLKS {1} \
    CONFIG.NUM_MC {1} \
    CONFIG.NUM_MCP {4} \
    CONFIG.NUM_MI {0} \
    CONFIG.NUM_NMI {0} \
    CONFIG.NUM_NSI {2} \
    CONFIG.NUM_SI {0} \
  ] $axi_noc_mc_ddr4_1


  set_property -dict [ list \
   CONFIG.CONNECTIONS { MC_0 {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64} } } \
 ] [get_bd_intf_pins /noc/axi_noc_mc_ddr4_1/S00_INI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS { MC_1 {read_bw {800} write_bw {800} read_avg_burst {64} write_avg_burst {64} } } \
 ] [get_bd_intf_pins /noc/axi_noc_mc_ddr4_1/S01_INI]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {} \
 ] [get_bd_pins /noc/axi_noc_mc_ddr4_1/aclk0]

  # Create interface connections
  connect_bd_intf_net -intf_net Conn1 [get_bd_intf_pins axi_noc_mc_ddr4_1/CH0_DDR4_0] [get_bd_intf_pins CH0_DDR4_0_1]
  connect_bd_intf_net -intf_net Conn2 [get_bd_intf_pins axi_noc_mc_ddr4_1/sys_clk0] [get_bd_intf_pins sys_clk0_1]
  connect_bd_intf_net -intf_net Conn3 [get_bd_intf_pins axi_noc_mc_ddr4_0/sys_clk0] [get_bd_intf_pins sys_clk0_0]
  connect_bd_intf_net -intf_net Conn4 [get_bd_intf_pins axi_noc_mc_ddr4_0/CH0_DDR4_0] [get_bd_intf_pins CH0_DDR4_0_0]
  connect_bd_intf_net -intf_net Conn5 [get_bd_intf_pins axi_noc_cips/S03_AXI] [get_bd_intf_pins S03_AXI]
  connect_bd_intf_net -intf_net Conn6 [get_bd_intf_pins axi_noc_cips/hbm_ref_clk0] [get_bd_intf_pins hbm_ref_clk_0]
  connect_bd_intf_net -intf_net Conn7 [get_bd_intf_pins axi_noc_cips/hbm_ref_clk1] [get_bd_intf_pins hbm_ref_clk_1]
  connect_bd_intf_net -intf_net Conn8 [get_bd_intf_pins axi_noc_cips/S01_AXI] [get_bd_intf_pins S01_AXI]
  connect_bd_intf_net -intf_net Conn9 [get_bd_intf_pins axi_noc_cips/S00_AXI] [get_bd_intf_pins S00_AXI]
  connect_bd_intf_net -intf_net Conn10 [get_bd_intf_pins axi_noc_cips/S02_AXI] [get_bd_intf_pins S02_AXI]
  connect_bd_intf_net -intf_net Conn11 [get_bd_intf_pins axi_noc_cips/M00_AXI] [get_bd_intf_pins M00_AXI]
  connect_bd_intf_net -intf_net Conn12 [get_bd_intf_pins axi_noc_cips/M01_AXI] [get_bd_intf_pins M01_AXI]
  connect_bd_intf_net -intf_net Conn13 [get_bd_intf_pins axi_noc_cips/HBM02_AXI] [get_bd_intf_pins HBM02_AXI]
  connect_bd_intf_net -intf_net Conn14 [get_bd_intf_pins axi_noc_cips/HBM03_AXI] [get_bd_intf_pins HBM03_AXI]
  connect_bd_intf_net -intf_net Conn15 [get_bd_intf_pins axi_noc_cips/HBM04_AXI] [get_bd_intf_pins HBM04_AXI]
  connect_bd_intf_net -intf_net Conn16 [get_bd_intf_pins axi_noc_cips/HBM05_AXI] [get_bd_intf_pins HBM05_AXI]
  connect_bd_intf_net -intf_net Conn17 [get_bd_intf_pins axi_noc_cips/HBM06_AXI] [get_bd_intf_pins HBM06_AXI]
  connect_bd_intf_net -intf_net Conn18 [get_bd_intf_pins axi_noc_cips/HBM07_AXI] [get_bd_intf_pins HBM07_AXI]
  connect_bd_intf_net -intf_net Conn19 [get_bd_intf_pins axi_noc_cips/HBM08_AXI] [get_bd_intf_pins HBM08_AXI]
  connect_bd_intf_net -intf_net Conn20 [get_bd_intf_pins axi_noc_cips/HBM09_AXI] [get_bd_intf_pins HBM09_AXI]
  connect_bd_intf_net -intf_net Conn21 [get_bd_intf_pins axi_noc_cips/HBM10_AXI] [get_bd_intf_pins HBM10_AXI]
  connect_bd_intf_net -intf_net Conn22 [get_bd_intf_pins axi_noc_cips/HBM11_AXI] [get_bd_intf_pins HBM11_AXI]
  connect_bd_intf_net -intf_net Conn23 [get_bd_intf_pins axi_noc_cips/HBM12_AXI] [get_bd_intf_pins HBM12_AXI]
  connect_bd_intf_net -intf_net Conn24 [get_bd_intf_pins axi_noc_cips/HBM13_AXI] [get_bd_intf_pins HBM13_AXI]
  connect_bd_intf_net -intf_net Conn25 [get_bd_intf_pins axi_noc_cips/HBM14_AXI] [get_bd_intf_pins HBM14_AXI]
  connect_bd_intf_net -intf_net Conn26 [get_bd_intf_pins axi_noc_cips/HBM15_AXI] [get_bd_intf_pins HBM15_AXI]
  connect_bd_intf_net -intf_net Conn27 [get_bd_intf_pins axi_noc_cips/HBM16_AXI] [get_bd_intf_pins HBM16_AXI]
  connect_bd_intf_net -intf_net Conn28 [get_bd_intf_pins axi_noc_cips/HBM17_AXI] [get_bd_intf_pins HBM17_AXI]
  connect_bd_intf_net -intf_net Conn29 [get_bd_intf_pins axi_noc_cips/HBM18_AXI] [get_bd_intf_pins HBM18_AXI]
  connect_bd_intf_net -intf_net Conn30 [get_bd_intf_pins axi_noc_cips/HBM19_AXI] [get_bd_intf_pins HBM19_AXI]
  connect_bd_intf_net -intf_net Conn31 [get_bd_intf_pins axi_noc_cips/HBM20_AXI] [get_bd_intf_pins HBM20_AXI]
  connect_bd_intf_net -intf_net Conn32 [get_bd_intf_pins axi_noc_cips/HBM21_AXI] [get_bd_intf_pins HBM21_AXI]
  connect_bd_intf_net -intf_net Conn33 [get_bd_intf_pins axi_noc_cips/HBM22_AXI] [get_bd_intf_pins HBM22_AXI]
  connect_bd_intf_net -intf_net Conn34 [get_bd_intf_pins axi_noc_cips/HBM23_AXI] [get_bd_intf_pins HBM23_AXI]
  connect_bd_intf_net -intf_net Conn35 [get_bd_intf_pins axi_noc_cips/HBM24_AXI] [get_bd_intf_pins HBM24_AXI]
  connect_bd_intf_net -intf_net Conn36 [get_bd_intf_pins axi_noc_cips/HBM25_AXI] [get_bd_intf_pins HBM25_AXI]
  connect_bd_intf_net -intf_net Conn37 [get_bd_intf_pins axi_noc_cips/HBM26_AXI] [get_bd_intf_pins HBM26_AXI]
  connect_bd_intf_net -intf_net Conn38 [get_bd_intf_pins axi_noc_cips/HBM27_AXI] [get_bd_intf_pins HBM27_AXI]
  connect_bd_intf_net -intf_net Conn39 [get_bd_intf_pins axi_noc_cips/HBM28_AXI] [get_bd_intf_pins HBM28_AXI]
  connect_bd_intf_net -intf_net Conn40 [get_bd_intf_pins axi_noc_cips/HBM29_AXI] [get_bd_intf_pins HBM29_AXI]
  connect_bd_intf_net -intf_net Conn41 [get_bd_intf_pins axi_noc_cips/HBM30_AXI] [get_bd_intf_pins HBM30_AXI]
  connect_bd_intf_net -intf_net Conn42 [get_bd_intf_pins axi_noc_cips/HBM31_AXI] [get_bd_intf_pins HBM31_AXI]
  connect_bd_intf_net -intf_net Conn43 [get_bd_intf_pins axi_noc_cips/HBM32_AXI] [get_bd_intf_pins HBM32_AXI]
  connect_bd_intf_net -intf_net Conn44 [get_bd_intf_pins axi_noc_cips/HBM33_AXI] [get_bd_intf_pins HBM33_AXI]
  connect_bd_intf_net -intf_net Conn45 [get_bd_intf_pins axi_noc_cips/HBM34_AXI] [get_bd_intf_pins HBM34_AXI]
  connect_bd_intf_net -intf_net Conn46 [get_bd_intf_pins axi_noc_cips/HBM35_AXI] [get_bd_intf_pins HBM35_AXI]
  connect_bd_intf_net -intf_net Conn47 [get_bd_intf_pins axi_noc_cips/HBM36_AXI] [get_bd_intf_pins HBM36_AXI]
  connect_bd_intf_net -intf_net Conn48 [get_bd_intf_pins axi_noc_cips/HBM37_AXI] [get_bd_intf_pins HBM37_AXI]
  connect_bd_intf_net -intf_net Conn49 [get_bd_intf_pins axi_noc_cips/HBM38_AXI] [get_bd_intf_pins HBM38_AXI]
  connect_bd_intf_net -intf_net Conn50 [get_bd_intf_pins axi_noc_cips/HBM39_AXI] [get_bd_intf_pins HBM39_AXI]
  connect_bd_intf_net -intf_net Conn51 [get_bd_intf_pins axi_noc_cips/HBM40_AXI] [get_bd_intf_pins HBM40_AXI]
  connect_bd_intf_net -intf_net Conn52 [get_bd_intf_pins axi_noc_cips/HBM41_AXI] [get_bd_intf_pins HBM41_AXI]
  connect_bd_intf_net -intf_net Conn53 [get_bd_intf_pins axi_noc_cips/HBM42_AXI] [get_bd_intf_pins HBM42_AXI]
  connect_bd_intf_net -intf_net Conn54 [get_bd_intf_pins axi_noc_cips/HBM43_AXI] [get_bd_intf_pins HBM43_AXI]
  connect_bd_intf_net -intf_net Conn55 [get_bd_intf_pins axi_noc_cips/HBM44_AXI] [get_bd_intf_pins HBM44_AXI]
  connect_bd_intf_net -intf_net Conn56 [get_bd_intf_pins axi_noc_cips/HBM45_AXI] [get_bd_intf_pins HBM45_AXI]
  connect_bd_intf_net -intf_net Conn57 [get_bd_intf_pins axi_noc_cips/HBM46_AXI] [get_bd_intf_pins HBM46_AXI]
  connect_bd_intf_net -intf_net Conn58 [get_bd_intf_pins axi_noc_cips/HBM47_AXI] [get_bd_intf_pins HBM47_AXI]
  connect_bd_intf_net -intf_net Conn59 [get_bd_intf_pins axi_noc_cips/HBM48_AXI] [get_bd_intf_pins HBM48_AXI]
  connect_bd_intf_net -intf_net Conn60 [get_bd_intf_pins axi_noc_cips/HBM49_AXI] [get_bd_intf_pins HBM49_AXI]
  connect_bd_intf_net -intf_net Conn61 [get_bd_intf_pins axi_noc_cips/HBM50_AXI] [get_bd_intf_pins HBM50_AXI]
  connect_bd_intf_net -intf_net Conn62 [get_bd_intf_pins axi_noc_cips/HBM51_AXI] [get_bd_intf_pins HBM51_AXI]
  connect_bd_intf_net -intf_net Conn63 [get_bd_intf_pins axi_noc_cips/HBM52_AXI] [get_bd_intf_pins HBM52_AXI]
  connect_bd_intf_net -intf_net Conn64 [get_bd_intf_pins axi_noc_cips/HBM53_AXI] [get_bd_intf_pins HBM53_AXI]
  connect_bd_intf_net -intf_net Conn65 [get_bd_intf_pins axi_noc_cips/HBM54_AXI] [get_bd_intf_pins HBM54_AXI]
  connect_bd_intf_net -intf_net Conn66 [get_bd_intf_pins axi_noc_cips/HBM55_AXI] [get_bd_intf_pins HBM55_AXI]
  connect_bd_intf_net -intf_net Conn67 [get_bd_intf_pins axi_noc_cips/HBM56_AXI] [get_bd_intf_pins HBM56_AXI]
  connect_bd_intf_net -intf_net Conn68 [get_bd_intf_pins axi_noc_cips/HBM57_AXI] [get_bd_intf_pins HBM57_AXI]
  connect_bd_intf_net -intf_net Conn69 [get_bd_intf_pins axi_noc_cips/HBM58_AXI] [get_bd_intf_pins HBM58_AXI]
  connect_bd_intf_net -intf_net Conn70 [get_bd_intf_pins axi_noc_cips/HBM59_AXI] [get_bd_intf_pins HBM59_AXI]
  connect_bd_intf_net -intf_net Conn71 [get_bd_intf_pins axi_noc_cips/HBM60_AXI] [get_bd_intf_pins HBM60_AXI]
  connect_bd_intf_net -intf_net Conn72 [get_bd_intf_pins axi_noc_cips/HBM61_AXI] [get_bd_intf_pins HBM61_AXI]
  connect_bd_intf_net -intf_net Conn73 [get_bd_intf_pins axi_noc_cips/HBM62_AXI] [get_bd_intf_pins HBM62_AXI]
  connect_bd_intf_net -intf_net Conn74 [get_bd_intf_pins axi_noc_cips/HBM63_AXI] [get_bd_intf_pins HBM63_AXI]
  connect_bd_intf_net -intf_net HBM00_AXI_1 [get_bd_intf_pins HBM00_AXI] [get_bd_intf_pins axi_noc_cips/HBM00_AXI]
  connect_bd_intf_net -intf_net HBM01_AXI_1 [get_bd_intf_pins HBM01_AXI] [get_bd_intf_pins axi_noc_cips/HBM01_AXI]
  connect_bd_intf_net -intf_net axi_noc_cips_M00_INI [get_bd_intf_pins axi_noc_cips/M00_INI] [get_bd_intf_pins axi_noc_mc_ddr4_0/S00_INI]
  connect_bd_intf_net -intf_net axi_noc_cips_M01_INI [get_bd_intf_pins axi_noc_cips/M01_INI] [get_bd_intf_pins axi_noc_mc_ddr4_0/S01_INI]
  connect_bd_intf_net -intf_net axi_noc_cips_M02_AXI [get_bd_intf_pins M02_AXI] [get_bd_intf_pins axi_noc_cips/M02_AXI]
  connect_bd_intf_net -intf_net axi_noc_cips_M02_INI [get_bd_intf_pins axi_noc_cips/M02_INI] [get_bd_intf_pins axi_noc_mc_ddr4_1/S00_INI]
  connect_bd_intf_net -intf_net axi_noc_cips_M03_INI [get_bd_intf_pins axi_noc_cips/M03_INI] [get_bd_intf_pins axi_noc_mc_ddr4_1/S01_INI]

  # Create port connections
  connect_bd_net -net aclk0_1  [get_bd_pins aclk0] \
  [get_bd_pins axi_noc_mc_ddr4_1/aclk0] \
  [get_bd_pins axi_noc_cips/aclk4]
  connect_bd_net -net aclk1_1  [get_bd_pins aclk1] \
  [get_bd_pins axi_noc_cips/aclk1]
  connect_bd_net -net aclk2_1  [get_bd_pins aclk2] \
  [get_bd_pins axi_noc_cips/aclk2]
  connect_bd_net -net aclk3_1  [get_bd_pins aclk3] \
  [get_bd_pins axi_noc_cips/aclk3]
  connect_bd_net -net aclk4_1  [get_bd_pins aclk4] \
  [get_bd_pins axi_noc_cips/aclk0]
  connect_bd_net -net aclk5_1  [get_bd_pins aclk5] \
  [get_bd_pins axi_noc_cips/aclk5]
  connect_bd_net -net aclk6_1  [get_bd_pins aclk6] \
  [get_bd_pins axi_noc_cips/aclk6]

  # Restore current instance
  current_bd_instance $oldCurInst
}

# Hierarchical cell: aved
proc create_hier_cell_aved { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_aved() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 gt_pcie_refclk

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:gt_rtl:1.0 gt_pciea1

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 smbus_0

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 CPM_PCIE_NOC_0

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 CPM_PCIE_NOC_1

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 PMC_NOC_AXI_0

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 LPD_AXI_NOC_0

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 s_axi_pcie_mgmt_slr0

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 NOC_PMC_AXI_0


  # Create pins
  create_bd_pin -dir O -type clk pl0_ref_clk
  create_bd_pin -dir O -type clk lpd_axi_noc_clk
  create_bd_pin -dir O -type clk pmc_axi_noc_axi0_clk
  create_bd_pin -dir O -type clk cpm_pcie_noc_axi1_clk
  create_bd_pin -dir O -type clk cpm_pcie_noc_axi0_clk
  create_bd_pin -dir O -from 0 -to 0 -type rst resetn_pl_periph
  create_bd_pin -dir O -type clk noc_pmc_axi_axi0_clk

  # Create instance: clock_reset
  create_hier_cell_clock_reset $hier_obj clock_reset

  # Create instance: base_logic
  create_hier_cell_base_logic $hier_obj base_logic

  # Create instance: cips, and set properties
  set cips [ create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips cips ]
  set_property -dict [list \
    CONFIG.CPM_CONFIG { \
      CPM_PCIE0_MODES {None} \
      CPM_PCIE0_TANDEM {Tandem_PCIe} \
      CPM_PCIE1_ACS_CAP_ON {0} \
      CPM_PCIE1_ARI_CAP_ENABLED {1} \
      CPM_PCIE1_CFG_EXT_IF {1} \
      CPM_PCIE1_CFG_VEND_ID {10ee} \
      CPM_PCIE1_COPY_PF0_QDMA_ENABLED {0} \
      CPM_PCIE1_EXT_PCIE_CFG_SPACE_ENABLED {Extended_Large} \
      CPM_PCIE1_FUNCTIONAL_MODE {QDMA} \
      CPM_PCIE1_MAX_LINK_SPEED {32.0_GT/s} \
      CPM_PCIE1_MODES {DMA} \
      CPM_PCIE1_MODE_SELECTION {Advanced} \
      CPM_PCIE1_MSI_X_OPTIONS {MSI-X_Internal} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_BASEADDR_0 {0x0000008000000000} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_BASEADDR_1 {0x0000008040000000} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_BASEADDR_2 {0x0000008080000000} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_BASEADDR_3 {0x00000080C0000000} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_BASEADDR_4 {0x0000008100000000} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_BASEADDR_5 {0x0000008140000000} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_HIGHADDR_0 {0x000000803FFFFFFFF} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_HIGHADDR_1 {0x000000807FFFFFFFF} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_HIGHADDR_2 {0x00000080BFFFFFFFF} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_HIGHADDR_3 {0x00000080FFFFFFFFF} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_HIGHADDR_4 {0x000000813FFFFFFFF} \
      CPM_PCIE1_PF0_AXIBAR2PCIE_HIGHADDR_5 {0x000000817FFFFFFFF} \
      CPM_PCIE1_PF0_BAR0_QDMA_64BIT {1} \
      CPM_PCIE1_PF0_BAR0_QDMA_ENABLED {1} \
      CPM_PCIE1_PF0_BAR0_QDMA_PREFETCHABLE {1} \
      CPM_PCIE1_PF0_BAR0_QDMA_SCALE {Megabytes} \
      CPM_PCIE1_PF0_BAR0_QDMA_SIZE {256} \
      CPM_PCIE1_PF0_BAR0_QDMA_TYPE {AXI_Bridge_Master} \
      CPM_PCIE1_PF0_BAR2_QDMA_64BIT {0} \
      CPM_PCIE1_PF0_BAR2_QDMA_ENABLED {0} \
      CPM_PCIE1_PF0_BAR2_QDMA_PREFETCHABLE {0} \
      CPM_PCIE1_PF0_BAR2_QDMA_SCALE {Kilobytes} \
      CPM_PCIE1_PF0_BAR2_QDMA_SIZE {4} \
      CPM_PCIE1_PF0_BAR2_QDMA_TYPE {AXI_Bridge_Master} \
      CPM_PCIE1_PF0_BASE_CLASS_VALUE {12} \
      CPM_PCIE1_PF0_CFG_DEV_ID {50b4} \
      CPM_PCIE1_PF0_CFG_SUBSYS_ID {000e} \
      CPM_PCIE1_PF0_DEV_CAP_FUNCTION_LEVEL_RESET_CAPABLE {0} \
      CPM_PCIE1_PF0_MSIX_CAP_TABLE_OFFSET {40} \
      CPM_PCIE1_PF0_MSIX_CAP_TABLE_SIZE {1} \
      CPM_PCIE1_PF0_MSIX_ENABLED {0} \
      CPM_PCIE1_PF0_PCIEBAR2AXIBAR_QDMA_0 {0x0000020100000000} \
      CPM_PCIE1_PF0_SUB_CLASS_VALUE {00} \
      CPM_PCIE1_PF1_BAR0_QDMA_64BIT {1} \
      CPM_PCIE1_PF1_BAR0_QDMA_ENABLED {1} \
      CPM_PCIE1_PF1_BAR0_QDMA_PREFETCHABLE {1} \
      CPM_PCIE1_PF1_BAR0_QDMA_SCALE {Kilobytes} \
      CPM_PCIE1_PF1_BAR0_QDMA_SIZE {512} \
      CPM_PCIE1_PF1_BAR0_QDMA_TYPE {DMA} \
      CPM_PCIE1_PF1_BAR2_QDMA_64BIT {0} \
      CPM_PCIE1_PF1_BAR2_QDMA_ENABLED {0} \
      CPM_PCIE1_PF1_BAR2_QDMA_PREFETCHABLE {0} \
      CPM_PCIE1_PF1_BAR2_QDMA_SCALE {Kilobytes} \
      CPM_PCIE1_PF1_BAR2_QDMA_SIZE {4} \
      CPM_PCIE1_PF1_BAR2_QDMA_TYPE {AXI_Bridge_Master} \
      CPM_PCIE1_PF1_BASE_CLASS_VALUE {12} \
      CPM_PCIE1_PF1_CFG_DEV_ID {50b5} \
      CPM_PCIE1_PF1_CFG_SUBSYS_ID {000e} \
      CPM_PCIE1_PF1_CFG_SUBSYS_VEND_ID {10EE} \
      CPM_PCIE1_PF1_MSIX_CAP_TABLE_OFFSET {50000} \
      CPM_PCIE1_PF1_MSIX_CAP_TABLE_SIZE {8} \
      CPM_PCIE1_PF1_MSIX_ENABLED {1} \
      CPM_PCIE1_PF1_PCIEBAR2AXIBAR_QDMA_2 {0x0000020200000000} \
      CPM_PCIE1_PF1_SUB_CLASS_VALUE {00} \
      CPM_PCIE1_PF2_BAR0_QDMA_64BIT {1} \
      CPM_PCIE1_PF2_BAR0_QDMA_SCALE {Kilobytes} \
      CPM_PCIE1_PF2_BAR0_QDMA_SIZE {512} \
      CPM_PCIE1_PF2_BASE_CLASS_VALUE {12} \
      CPM_PCIE1_PF2_CFG_DEV_ID {50b6} \
      CPM_PCIE1_PF2_CFG_SUBSYS_ID {000e} \
      CPM_PCIE1_PF2_CFG_SUBSYS_VEND_ID {10EE} \
      CPM_PCIE1_PF2_PCIEBAR2AXIBAR_QDMA_0 {0x0000020200000000} \
      CPM_PCIE1_PF2_USE_CLASS_CODE_LOOKUP_ASSISTANT {0} \
      CPM_PCIE1_PL_LINK_CAP_MAX_LINK_WIDTH {X8} \
      CPM_PCIE1_TL_PF_ENABLE_REG {3} \
    } \
    CONFIG.PS_PMC_CONFIG { \
      BOOT_MODE {Custom} \
      CLOCK_MODE {Custom} \
      DDR_MEMORY_MODE {Custom} \
      DESIGN_MODE {1} \
      DEVICE_INTEGRITY_MODE {Custom} \
      IO_CONFIG_MODE {Custom} \
      PCIE_APERTURES_DUAL_ENABLE {0} \
      PCIE_APERTURES_SINGLE_ENABLE {1} \
      PMC_BANK_1_IO_STANDARD {LVCMOS3.3} \
      PMC_CRP_OSPI_REF_CTRL_FREQMHZ {200} \
      PMC_CRP_PL0_REF_CTRL_FREQMHZ {100} \
      PMC_CRP_PL1_REF_CTRL_FREQMHZ {33.3333333} \
      PMC_CRP_PL2_REF_CTRL_FREQMHZ {250} \
      PMC_GLITCH_CONFIG {{DEPTH_SENSITIVITY 1} {MIN_PULSE_WIDTH 0.5} {TYPE CUSTOM} {VCC_PMC_VALUE 0.88}} \
      PMC_GLITCH_CONFIG_1 {{DEPTH_SENSITIVITY 1} {MIN_PULSE_WIDTH 0.5} {TYPE CUSTOM} {VCC_PMC_VALUE 0.88}} \
      PMC_GLITCH_CONFIG_2 {{DEPTH_SENSITIVITY 1} {MIN_PULSE_WIDTH 0.5} {TYPE CUSTOM} {VCC_PMC_VALUE 0.88}} \
      PMC_GPIO_EMIO_PERIPHERAL_ENABLE {0} \
      PMC_MIO11 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO12 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO13 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}} \
      PMC_MIO17 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO26 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO27 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO28 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO29 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO30 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO31 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO32 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO33 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO34 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO35 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO36 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO37 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO38 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO39 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO40 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO41 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO42 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO43 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO44 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO48 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO49 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO50 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO51 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO_EN_FOR_PL_PCIE {0} \
      PMC_OSPI_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 0 .. 11}} {MODE Single}} \
      PMC_REF_CLK_FREQMHZ {33.333333} \
      PMC_SD0_DATA_TRANSFER_MODE {8Bit} \
      PMC_SD0_PERIPHERAL {{CLK_100_SDR_OTAP_DLY 0x00} {CLK_200_SDR_OTAP_DLY 0x2} {CLK_50_DDR_ITAP_DLY 0x1E} {CLK_50_DDR_OTAP_DLY 0x5} {CLK_50_SDR_ITAP_DLY 0x2C} {CLK_50_SDR_OTAP_DLY 0x5} {ENABLE 1} {IO\
{PMC_MIO 13 .. 25}}} \
      PMC_SD0_SLOT_TYPE {eMMC} \
      PMC_USE_NOC_PMC_AXI0 {1} \
      PMC_USE_PMC_NOC_AXI0 {1} \
      PS_BANK_2_IO_STANDARD {LVCMOS3.3} \
      PS_BOARD_INTERFACE {Custom} \
      PS_CRL_CPM_TOPSW_REF_CTRL_FREQMHZ {1000} \
      PS_GEN_IPI0_ENABLE {0} \
      PS_GEN_IPI1_ENABLE {0} \
      PS_GEN_IPI2_ENABLE {0} \
      PS_GEN_IPI3_ENABLE {1} \
      PS_GEN_IPI3_MASTER {R5_0} \
      PS_GEN_IPI4_ENABLE {1} \
      PS_GEN_IPI4_MASTER {R5_0} \
      PS_GEN_IPI5_ENABLE {1} \
      PS_GEN_IPI5_MASTER {R5_1} \
      PS_GEN_IPI6_ENABLE {1} \
      PS_GEN_IPI6_MASTER {R5_1} \
      PS_GPIO_EMIO_PERIPHERAL_ENABLE {0} \
      PS_I2C0_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 2 .. 3}}} \
      PS_I2C1_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 0 .. 1}}} \
      PS_IRQ_USAGE {{CH0 1} {CH1 1} {CH10 0} {CH11 0} {CH12 0} {CH13 0} {CH14 0} {CH15 0} {CH2 0} {CH3 0} {CH4 0} {CH5 0} {CH6 0} {CH7 0} {CH8 0} {CH9 0}} \
      PS_KAT_ENABLE {0} \
      PS_KAT_ENABLE_1 {0} \
      PS_KAT_ENABLE_2 {0} \
      PS_MIO10 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO11 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO12 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}} \
      PS_MIO13 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO14 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO18 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO19 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO22 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO23 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO24 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO25 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO4 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO5 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO6 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO7 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PS_MIO8 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}} \
      PS_MIO9 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 1} {SLEW slow} {USAGE Reserved}} \
      PS_M_AXI_LPD_DATA_WIDTH {32} \
      PS_NUM_FABRIC_RESETS {1} \
      PS_PCIE1_PERIPHERAL_ENABLE {0} \
      PS_PCIE2_PERIPHERAL_ENABLE {1} \
      PS_PCIE_EP_RESET1_IO {PMC_MIO 24} \
      PS_PCIE_EP_RESET2_IO {PMC_MIO 25} \
      PS_PCIE_RESET {ENABLE 1} \
      PS_PL_CONNECTIVITY_MODE {Custom} \
      PS_SPI0 {{GRP_SS0_ENABLE 1} {GRP_SS0_IO {PS_MIO 15}} {GRP_SS1_ENABLE 0} {GRP_SS1_IO {PMC_MIO 14}} {GRP_SS2_ENABLE 0} {GRP_SS2_IO {PMC_MIO 13}} {PERIPHERAL_ENABLE 1} {PERIPHERAL_IO {PS_MIO 12 .. 17}}}\
\
      PS_SPI1 {{GRP_SS0_ENABLE 0} {GRP_SS0_IO {PS_MIO 9}} {GRP_SS1_ENABLE 0} {GRP_SS1_IO {PS_MIO 8}} {GRP_SS2_ENABLE 0} {GRP_SS2_IO {PS_MIO 7}} {PERIPHERAL_ENABLE 0} {PERIPHERAL_IO {PS_MIO 6 .. 11}}} \
      PS_TTC0_PERIPHERAL_ENABLE {1} \
      PS_TTC1_PERIPHERAL_ENABLE {1} \
      PS_TTC2_PERIPHERAL_ENABLE {1} \
      PS_TTC3_PERIPHERAL_ENABLE {1} \
      PS_UART0_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 8 .. 9}}} \
      PS_UART1_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 20 .. 21}}} \
      PS_USE_FPD_CCI_NOC {0} \
      PS_USE_M_AXI_FPD {0} \
      PS_USE_M_AXI_LPD {1} \
      PS_USE_NOC_LPD_AXI0 {1} \
      PS_USE_PMCPL_CLK0 {1} \
      PS_USE_PMCPL_CLK1 {1} \
      PS_USE_PMCPL_CLK2 {1} \
      PS_USE_S_AXI_LPD {0} \
      SMON_ALARMS {Set_Alarms_On} \
      SMON_ENABLE_TEMP_AVERAGING {0} \
      SMON_MEAS100 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 4.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {4 V unipolar}} {NAME VCCO_500} {SUPPLY_NUM 9}} \
      SMON_MEAS101 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 4.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {4 V unipolar}} {NAME VCCO_501} {SUPPLY_NUM 10}} \
      SMON_MEAS102 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 4.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {4 V unipolar}} {NAME VCCO_502} {SUPPLY_NUM 11}} \
      SMON_MEAS103 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 4.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {4 V unipolar}} {NAME VCCO_503} {SUPPLY_NUM 12}} \
      SMON_MEAS104 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCO_700} {SUPPLY_NUM 13}} \
      SMON_MEAS105 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCO_701} {SUPPLY_NUM 14}} \
      SMON_MEAS106 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCO_702} {SUPPLY_NUM 15}} \
      SMON_MEAS118 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_PMC} {SUPPLY_NUM 0}} \
      SMON_MEAS119 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_PSFP} {SUPPLY_NUM 1}} \
      SMON_MEAS120 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_PSLP} {SUPPLY_NUM 2}} \
      SMON_MEAS121 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_RAM} {SUPPLY_NUM 3}} \
      SMON_MEAS122 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_SOC} {SUPPLY_NUM 4}} \
      SMON_MEAS47 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVCCAUX_104} {SUPPLY_NUM 20}} \
      SMON_MEAS48 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVCCAUX_105} {SUPPLY_NUM 21}} \
      SMON_MEAS64 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVCC_104} {SUPPLY_NUM 18}} \
      SMON_MEAS65 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVCC_105} {SUPPLY_NUM 19}} \
      SMON_MEAS81 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVTT_104} {SUPPLY_NUM 22}} \
      SMON_MEAS82 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVTT_105} {SUPPLY_NUM 23}} \
      SMON_MEAS96 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCAUX} {SUPPLY_NUM 6}} \
      SMON_MEAS97 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCAUX_PMC} {SUPPLY_NUM 7}} \
      SMON_MEAS98 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCAUX_SMON} {SUPPLY_NUM 8}} \
      SMON_MEAS99 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCINT} {SUPPLY_NUM 5}} \
      SMON_TEMP_AVERAGING_SAMPLES {0} \
      SMON_VOLTAGE_AVERAGING_SAMPLES {8} \
    } \
    CONFIG.PS_PMC_CONFIG_APPLIED {1} \
  ] $cips


  # Create interface connections
  connect_bd_intf_net -intf_net Conn1 [get_bd_intf_pins cips/gt_refclk1] [get_bd_intf_pins gt_pcie_refclk]
  connect_bd_intf_net -intf_net Conn2 [get_bd_intf_pins cips/PCIE1_GT] [get_bd_intf_pins gt_pciea1]
  connect_bd_intf_net -intf_net Conn3 [get_bd_intf_pins base_logic/smbus_rpu] [get_bd_intf_pins smbus_0]
  connect_bd_intf_net -intf_net Conn4 [get_bd_intf_pins cips/CPM_PCIE_NOC_0] [get_bd_intf_pins CPM_PCIE_NOC_0]
  connect_bd_intf_net -intf_net Conn5 [get_bd_intf_pins cips/CPM_PCIE_NOC_1] [get_bd_intf_pins CPM_PCIE_NOC_1]
  connect_bd_intf_net -intf_net Conn6 [get_bd_intf_pins cips/PMC_NOC_AXI_0] [get_bd_intf_pins PMC_NOC_AXI_0]
  connect_bd_intf_net -intf_net Conn7 [get_bd_intf_pins cips/LPD_AXI_NOC_0] [get_bd_intf_pins LPD_AXI_NOC_0]
  connect_bd_intf_net -intf_net Conn8 [get_bd_intf_pins base_logic/s_axi_pcie_mgmt_slr0] [get_bd_intf_pins s_axi_pcie_mgmt_slr0]
  connect_bd_intf_net -intf_net NOC_PMC_AXI_0_1 [get_bd_intf_pins NOC_PMC_AXI_0] [get_bd_intf_pins cips/NOC_PMC_AXI_0]
  connect_bd_intf_net -intf_net base_logic_m_axi_pcie_mgmt_pdi_reset [get_bd_intf_pins base_logic/m_axi_pcie_mgmt_pdi_reset] [get_bd_intf_pins clock_reset/s_axi_pcie_mgmt_pdi_reset]
  connect_bd_intf_net -intf_net cips_M_AXI_LPD [get_bd_intf_pins cips/M_AXI_LPD] [get_bd_intf_pins base_logic/s_axi_rpu]
  connect_bd_intf_net -intf_net cips_pcie1_cfg_ext [get_bd_intf_pins cips/pcie1_cfg_ext] [get_bd_intf_pins base_logic/pcie_cfg_ext]

  # Create port connections
  connect_bd_net -net base_logic_irq_axi_smbus_rpu  [get_bd_pins base_logic/irq_axi_smbus_rpu] \
  [get_bd_pins cips/pl_ps_irq1]
  connect_bd_net -net base_logic_irq_gcq_m2r  [get_bd_pins base_logic/irq_gcq_m2r] \
  [get_bd_pins cips/pl_ps_irq0]
  connect_bd_net -net cips_cpm_pcie_noc_axi0_clk  [get_bd_pins cips/cpm_pcie_noc_axi0_clk] \
  [get_bd_pins cpm_pcie_noc_axi0_clk]
  connect_bd_net -net cips_cpm_pcie_noc_axi1_clk  [get_bd_pins cips/cpm_pcie_noc_axi1_clk] \
  [get_bd_pins cpm_pcie_noc_axi1_clk]
  connect_bd_net -net cips_dma1_axi_aresetn  [get_bd_pins cips/dma1_axi_aresetn] \
  [get_bd_pins clock_reset/dma_axi_aresetn]
  connect_bd_net -net cips_lpd_axi_noc_clk  [get_bd_pins cips/lpd_axi_noc_clk] \
  [get_bd_pins lpd_axi_noc_clk]
  connect_bd_net -net cips_noc_pmc_axi_axi0_clk  [get_bd_pins cips/noc_pmc_axi_axi0_clk] \
  [get_bd_pins noc_pmc_axi_axi0_clk]
  connect_bd_net -net cips_pl0_ref_clk  [get_bd_pins cips/pl0_ref_clk] \
  [get_bd_pins pl0_ref_clk] \
  [get_bd_pins cips/m_axi_lpd_aclk] \
  [get_bd_pins base_logic/clk_pl] \
  [get_bd_pins clock_reset/clk_pl]
  connect_bd_net -net cips_pl0_resetn  [get_bd_pins cips/pl0_resetn] \
  [get_bd_pins clock_reset/resetn_pl_axi]
  connect_bd_net -net cips_pl1_ref_clk  [get_bd_pins cips/pl1_ref_clk] \
  [get_bd_pins clock_reset/clk_freerun]
  connect_bd_net -net cips_pl2_ref_clk  [get_bd_pins cips/pl2_ref_clk] \
  [get_bd_pins cips/dma1_intrfc_clk] \
  [get_bd_pins base_logic/clk_pcie] \
  [get_bd_pins clock_reset/clk_pcie]
  connect_bd_net -net cips_pmc_axi_noc_axi0_clk  [get_bd_pins cips/pmc_axi_noc_axi0_clk] \
  [get_bd_pins pmc_axi_noc_axi0_clk]
  connect_bd_net -net clock_reset_resetn_pcie_ic  [get_bd_pins clock_reset/resetn_pcie_ic] \
  [get_bd_pins cips/dma1_intrfc_resetn]
  connect_bd_net -net clock_reset_resetn_pcie_periph  [get_bd_pins clock_reset/resetn_pcie_periph] \
  [get_bd_pins base_logic/resetn_pcie_periph]
  connect_bd_net -net clock_reset_resetn_pl_ic  [get_bd_pins clock_reset/resetn_pl_ic] \
  [get_bd_pins base_logic/resetn_pl_ic]
  connect_bd_net -net clock_reset_resetn_pl_periph  [get_bd_pins clock_reset/resetn_pl_periph] \
  [get_bd_pins base_logic/resetn_pl_periph] \
  [get_bd_pins resetn_pl_periph]

  # Restore current instance
  current_bd_instance $oldCurInst
}


# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell } {

  variable script_folder

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj


  # Create interface ports
  set CH0_DDR4_0_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 CH0_DDR4_0_0 ]

  set sys_clk0_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 sys_clk0_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {200000000} \
   ] $sys_clk0_0

  set CH0_DDR4_0_1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 CH0_DDR4_0_1 ]

  set sys_clk0_1 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 sys_clk0_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {200000000} \
   ] $sys_clk0_1

  set hbm_ref_clk_0 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 hbm_ref_clk_0 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {200000000} \
   ] $hbm_ref_clk_0

  set hbm_ref_clk_1 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 hbm_ref_clk_1 ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {200000000} \
   ] $hbm_ref_clk_1

  set gt_pcie_refclk [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 gt_pcie_refclk ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {100000000} \
   ] $gt_pcie_refclk

  set gt_pciea1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:gt_rtl:1.0 gt_pciea1 ]

  set smbus_0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:iic_rtl:1.0 smbus_0 ]


  # Create ports

  # Create instance: aved
  create_hier_cell_aved [current_bd_instance .] aved

  # Create instance: noc
  create_hier_cell_noc [current_bd_instance .] noc

  # Create instance: slash
  create_hier_cell_slash [current_bd_instance .] slash

  # Create interface connections
  connect_bd_intf_net -intf_net NOC_PMC_AXI_0_1 [get_bd_intf_pins aved/NOC_PMC_AXI_0] [get_bd_intf_pins noc/M02_AXI]
  connect_bd_intf_net -intf_net aved_CPM_PCIE_NOC_0 [get_bd_intf_pins aved/CPM_PCIE_NOC_0] [get_bd_intf_pins noc/S00_AXI]
  connect_bd_intf_net -intf_net aved_CPM_PCIE_NOC_1 [get_bd_intf_pins aved/CPM_PCIE_NOC_1] [get_bd_intf_pins noc/S01_AXI]
  connect_bd_intf_net -intf_net aved_LPD_AXI_NOC_0 [get_bd_intf_pins aved/LPD_AXI_NOC_0] [get_bd_intf_pins noc/S03_AXI]
  connect_bd_intf_net -intf_net aved_PMC_NOC_AXI_0 [get_bd_intf_pins aved/PMC_NOC_AXI_0] [get_bd_intf_pins noc/S02_AXI]
  connect_bd_intf_net -intf_net aved_gt_pciea1 [get_bd_intf_ports gt_pciea1] [get_bd_intf_pins aved/gt_pciea1]
  connect_bd_intf_net -intf_net aved_smbus_0 [get_bd_intf_ports smbus_0] [get_bd_intf_pins aved/smbus_0]
  connect_bd_intf_net -intf_net gt_pcie_refclk_1 [get_bd_intf_ports gt_pcie_refclk] [get_bd_intf_pins aved/gt_pcie_refclk]
  connect_bd_intf_net -intf_net hbm_ref_clk_0_1 [get_bd_intf_ports hbm_ref_clk_0] [get_bd_intf_pins noc/hbm_ref_clk_0]
  connect_bd_intf_net -intf_net hbm_ref_clk_1_1 [get_bd_intf_ports hbm_ref_clk_1] [get_bd_intf_pins noc/hbm_ref_clk_1]
  connect_bd_intf_net -intf_net noc_CH0_DDR4_0_0 [get_bd_intf_ports CH0_DDR4_0_0] [get_bd_intf_pins noc/CH0_DDR4_0_0]
  connect_bd_intf_net -intf_net noc_CH0_DDR4_0_1 [get_bd_intf_ports CH0_DDR4_0_1] [get_bd_intf_pins noc/CH0_DDR4_0_1]
  connect_bd_intf_net -intf_net s_axi_pcie_mgmt_slr0_1 [get_bd_intf_pins aved/s_axi_pcie_mgmt_slr0] [get_bd_intf_pins noc/M00_AXI]
  connect_bd_intf_net -intf_net s_axilite_1 [get_bd_intf_pins slash/s_axilite] [get_bd_intf_pins noc/M01_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI00 [get_bd_intf_pins noc/HBM00_AXI] [get_bd_intf_pins slash/M_AXI00]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets slash_M_AXI00]
  connect_bd_intf_net -intf_net slash_M_AXI01 [get_bd_intf_pins noc/HBM01_AXI] [get_bd_intf_pins slash/M_AXI01]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets slash_M_AXI01]
  connect_bd_intf_net -intf_net slash_M_AXI2 [get_bd_intf_pins noc/HBM02_AXI] [get_bd_intf_pins slash/M_AXI2]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets slash_M_AXI2]
  connect_bd_intf_net -intf_net slash_M_AXI3 [get_bd_intf_pins noc/HBM03_AXI] [get_bd_intf_pins slash/M_AXI3]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets slash_M_AXI3]
  connect_bd_intf_net -intf_net slash_M_AXI4 [get_bd_intf_pins noc/HBM04_AXI] [get_bd_intf_pins slash/M_AXI4]
  connect_bd_intf_net -intf_net slash_M_AXI5 [get_bd_intf_pins noc/HBM05_AXI] [get_bd_intf_pins slash/M_AXI5]
  connect_bd_intf_net -intf_net slash_M_AXI6 [get_bd_intf_pins noc/HBM06_AXI] [get_bd_intf_pins slash/M_AXI6]
  connect_bd_intf_net -intf_net slash_M_AXI7 [get_bd_intf_pins noc/HBM07_AXI] [get_bd_intf_pins slash/M_AXI7]
  connect_bd_intf_net -intf_net slash_M_AXI8 [get_bd_intf_pins noc/HBM08_AXI] [get_bd_intf_pins slash/M_AXI8]
  connect_bd_intf_net -intf_net slash_M_AXI9 [get_bd_intf_pins noc/HBM09_AXI] [get_bd_intf_pins slash/M_AXI9]
  connect_bd_intf_net -intf_net slash_M_AXI10 [get_bd_intf_pins noc/HBM10_AXI] [get_bd_intf_pins slash/M_AXI10]
  connect_bd_intf_net -intf_net slash_M_AXI11 [get_bd_intf_pins noc/HBM11_AXI] [get_bd_intf_pins slash/M_AXI11]
  connect_bd_intf_net -intf_net slash_M_AXI12 [get_bd_intf_pins noc/HBM12_AXI] [get_bd_intf_pins slash/M_AXI12]
  connect_bd_intf_net -intf_net slash_M_AXI13 [get_bd_intf_pins noc/HBM13_AXI] [get_bd_intf_pins slash/M_AXI13]
  connect_bd_intf_net -intf_net slash_M_AXI14 [get_bd_intf_pins noc/HBM14_AXI] [get_bd_intf_pins slash/M_AXI14]
  connect_bd_intf_net -intf_net slash_M_AXI15 [get_bd_intf_pins noc/HBM15_AXI] [get_bd_intf_pins slash/M_AXI15]
  connect_bd_intf_net -intf_net slash_M_AXI16 [get_bd_intf_pins noc/HBM16_AXI] [get_bd_intf_pins slash/M_AXI16]
  connect_bd_intf_net -intf_net slash_M_AXI17 [get_bd_intf_pins noc/HBM17_AXI] [get_bd_intf_pins slash/M_AXI17]
  connect_bd_intf_net -intf_net slash_M_AXI18 [get_bd_intf_pins slash/M_AXI18] [get_bd_intf_pins noc/HBM18_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI19 [get_bd_intf_pins slash/M_AXI19] [get_bd_intf_pins noc/HBM19_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI20 [get_bd_intf_pins slash/M_AXI20] [get_bd_intf_pins noc/HBM20_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI21 [get_bd_intf_pins slash/M_AXI21] [get_bd_intf_pins noc/HBM21_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI22 [get_bd_intf_pins slash/M_AXI22] [get_bd_intf_pins noc/HBM22_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI23 [get_bd_intf_pins slash/M_AXI23] [get_bd_intf_pins noc/HBM23_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI24 [get_bd_intf_pins slash/M_AXI24] [get_bd_intf_pins noc/HBM24_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI25 [get_bd_intf_pins slash/M_AXI25] [get_bd_intf_pins noc/HBM25_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI26 [get_bd_intf_pins slash/M_AXI26] [get_bd_intf_pins noc/HBM26_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI27 [get_bd_intf_pins slash/M_AXI27] [get_bd_intf_pins noc/HBM27_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI28 [get_bd_intf_pins slash/M_AXI28] [get_bd_intf_pins noc/HBM28_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI29 [get_bd_intf_pins slash/M_AXI29] [get_bd_intf_pins noc/HBM29_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI30 [get_bd_intf_pins slash/M_AXI30] [get_bd_intf_pins noc/HBM30_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI31 [get_bd_intf_pins slash/M_AXI31] [get_bd_intf_pins noc/HBM31_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI32 [get_bd_intf_pins slash/M_AXI32] [get_bd_intf_pins noc/HBM32_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI33 [get_bd_intf_pins slash/M_AXI33] [get_bd_intf_pins noc/HBM33_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI34 [get_bd_intf_pins slash/M_AXI34] [get_bd_intf_pins noc/HBM34_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI35 [get_bd_intf_pins slash/M_AXI35] [get_bd_intf_pins noc/HBM35_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI36 [get_bd_intf_pins slash/M_AXI36] [get_bd_intf_pins noc/HBM36_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI37 [get_bd_intf_pins slash/M_AXI37] [get_bd_intf_pins noc/HBM37_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI38 [get_bd_intf_pins slash/M_AXI38] [get_bd_intf_pins noc/HBM38_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI39 [get_bd_intf_pins slash/M_AXI39] [get_bd_intf_pins noc/HBM39_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI40 [get_bd_intf_pins slash/M_AXI40] [get_bd_intf_pins noc/HBM40_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI41 [get_bd_intf_pins slash/M_AXI41] [get_bd_intf_pins noc/HBM41_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI42 [get_bd_intf_pins slash/M_AXI42] [get_bd_intf_pins noc/HBM42_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI43 [get_bd_intf_pins slash/M_AXI43] [get_bd_intf_pins noc/HBM43_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI44 [get_bd_intf_pins slash/M_AXI44] [get_bd_intf_pins noc/HBM44_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI45 [get_bd_intf_pins slash/M_AXI45] [get_bd_intf_pins noc/HBM45_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI46 [get_bd_intf_pins slash/M_AXI46] [get_bd_intf_pins noc/HBM46_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI47 [get_bd_intf_pins slash/M_AXI47] [get_bd_intf_pins noc/HBM47_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI48 [get_bd_intf_pins slash/M_AXI48] [get_bd_intf_pins noc/HBM48_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI49 [get_bd_intf_pins slash/M_AXI49] [get_bd_intf_pins noc/HBM49_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI50 [get_bd_intf_pins slash/M_AXI50] [get_bd_intf_pins noc/HBM50_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI51 [get_bd_intf_pins slash/M_AXI51] [get_bd_intf_pins noc/HBM51_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI52 [get_bd_intf_pins slash/M_AXI52] [get_bd_intf_pins noc/HBM52_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI53 [get_bd_intf_pins slash/M_AXI53] [get_bd_intf_pins noc/HBM53_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI54 [get_bd_intf_pins slash/M_AXI54] [get_bd_intf_pins noc/HBM54_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI55 [get_bd_intf_pins slash/M_AXI55] [get_bd_intf_pins noc/HBM55_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI56 [get_bd_intf_pins slash/M_AXI56] [get_bd_intf_pins noc/HBM56_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI57 [get_bd_intf_pins slash/M_AXI57] [get_bd_intf_pins noc/HBM57_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI58 [get_bd_intf_pins slash/M_AXI58] [get_bd_intf_pins noc/HBM58_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI59 [get_bd_intf_pins slash/M_AXI59] [get_bd_intf_pins noc/HBM59_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI60 [get_bd_intf_pins slash/M_AXI60] [get_bd_intf_pins noc/HBM60_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI61 [get_bd_intf_pins slash/M_AXI61] [get_bd_intf_pins noc/HBM61_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI62 [get_bd_intf_pins slash/M_AXI62] [get_bd_intf_pins noc/HBM62_AXI]
  connect_bd_intf_net -intf_net slash_M_AXI63 [get_bd_intf_pins slash/M_AXI63] [get_bd_intf_pins noc/HBM63_AXI]
  connect_bd_intf_net -intf_net sys_clk0_0_1 [get_bd_intf_ports sys_clk0_0] [get_bd_intf_pins noc/sys_clk0_0]
  connect_bd_intf_net -intf_net sys_clk0_1_1 [get_bd_intf_ports sys_clk0_1] [get_bd_intf_pins noc/sys_clk0_1]

  # Create port connections
  connect_bd_net -net aclk5_1  [get_bd_pins slash/clk_out1] \
  [get_bd_pins noc/aclk5]
  connect_bd_net -net aved_cpm_pcie_noc_axi0_clk  [get_bd_pins aved/cpm_pcie_noc_axi0_clk] \
  [get_bd_pins noc/aclk4]
  connect_bd_net -net aved_cpm_pcie_noc_axi1_clk  [get_bd_pins aved/cpm_pcie_noc_axi1_clk] \
  [get_bd_pins noc/aclk1]
  connect_bd_net -net aved_lpd_axi_noc_clk  [get_bd_pins aved/lpd_axi_noc_clk] \
  [get_bd_pins noc/aclk3]
  connect_bd_net -net aved_noc_pmc_axi_axi0_clk  [get_bd_pins aved/noc_pmc_axi_axi0_clk] \
  [get_bd_pins noc/aclk6]
  connect_bd_net -net aved_pl0_ref_clk  [get_bd_pins aved/pl0_ref_clk] \
  [get_bd_pins noc/aclk0] \
  [get_bd_pins slash/s_axi_aclk]
  connect_bd_net -net aved_pmc_axi_noc_axi0_clk  [get_bd_pins aved/pmc_axi_noc_axi0_clk] \
  [get_bd_pins noc/aclk2]
  connect_bd_net -net reset_rtl_0_1  [get_bd_pins aved/resetn_pl_periph] \
  [get_bd_pins slash/arstn]

  # Create address segments
  assign_bd_address -offset 0x004000000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM0_PC0] -force
  assign_bd_address -offset 0x004040000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM0_PC1] -force
  assign_bd_address -offset 0x004500000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM10_PC0] -force
  assign_bd_address -offset 0x004540000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM10_PC1] -force
  assign_bd_address -offset 0x004580000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM11_PC0] -force
  assign_bd_address -offset 0x0045C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM11_PC1] -force
  assign_bd_address -offset 0x004600000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM12_PC0] -force
  assign_bd_address -offset 0x004640000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM12_PC1] -force
  assign_bd_address -offset 0x004680000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM13_PC0] -force
  assign_bd_address -offset 0x0046C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM13_PC1] -force
  assign_bd_address -offset 0x004700000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM14_PC0] -force
  assign_bd_address -offset 0x004740000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM14_PC1] -force
  assign_bd_address -offset 0x004780000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM15_PC0] -force
  assign_bd_address -offset 0x0047C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM15_PC1] -force
  assign_bd_address -offset 0x004080000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM1_PC0] -force
  assign_bd_address -offset 0x0040C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM1_PC1] -force
  assign_bd_address -offset 0x004100000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM2_PC0] -force
  assign_bd_address -offset 0x004140000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM2_PC1] -force
  assign_bd_address -offset 0x004180000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM3_PC0] -force
  assign_bd_address -offset 0x0041C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM3_PC1] -force
  assign_bd_address -offset 0x004200000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM4_PC0] -force
  assign_bd_address -offset 0x004240000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM4_PC1] -force
  assign_bd_address -offset 0x004280000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM5_PC0] -force
  assign_bd_address -offset 0x0042C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM5_PC1] -force
  assign_bd_address -offset 0x004300000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM6_PC0] -force
  assign_bd_address -offset 0x004340000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM6_PC1] -force
  assign_bd_address -offset 0x004380000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM7_PC0] -force
  assign_bd_address -offset 0x0043C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM7_PC1] -force
  assign_bd_address -offset 0x004400000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM8_PC0] -force
  assign_bd_address -offset 0x004440000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM8_PC1] -force
  assign_bd_address -offset 0x004480000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM9_PC0] -force
  assign_bd_address -offset 0x0044C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_cips/S00_AXI/HBM9_PC1] -force
  assign_bd_address -offset 0x020108000000 -range 0x08000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_mc_ddr4_0/S00_INI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x060000000000 -range 0x000800000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_mc_ddr4_1/S00_INI/C0_DDR_CH2] -force
  assign_bd_address -offset 0x000102100000 -range 0x00010000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_slave_boot_stream] -force
  assign_bd_address -offset 0x020101010000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/base_logic/gcq_m2r/S00_AXI/S00_AXI_Reg] -force
  assign_bd_address -offset 0x020101000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/base_logic/hw_discovery/s_axi_ctrl_pf0/reg0] -force
  assign_bd_address -offset 0x020101040000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/clock_reset/pcie_mgmt_pdi_reset/pcie_mgmt_pdi_reset_gpio/S_AXI/Reg] -force
  assign_bd_address -offset 0x020101001000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/base_logic/uuid_rom/S_AXI/reg0] -force
  assign_bd_address -offset 0x004000000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM0_PC0] -force
  assign_bd_address -offset 0x004040000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM0_PC1] -force
  assign_bd_address -offset 0x004500000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM10_PC0] -force
  assign_bd_address -offset 0x004540000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM10_PC1] -force
  assign_bd_address -offset 0x004580000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM11_PC0] -force
  assign_bd_address -offset 0x0045C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM11_PC1] -force
  assign_bd_address -offset 0x004600000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM12_PC0] -force
  assign_bd_address -offset 0x004640000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM12_PC1] -force
  assign_bd_address -offset 0x004680000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM13_PC0] -force
  assign_bd_address -offset 0x0046C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM13_PC1] -force
  assign_bd_address -offset 0x004700000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM14_PC0] -force
  assign_bd_address -offset 0x004740000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM14_PC1] -force
  assign_bd_address -offset 0x004780000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM15_PC0] -force
  assign_bd_address -offset 0x0047C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM15_PC1] -force
  assign_bd_address -offset 0x004080000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM1_PC0] -force
  assign_bd_address -offset 0x0040C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM1_PC1] -force
  assign_bd_address -offset 0x004100000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM2_PC0] -force
  assign_bd_address -offset 0x004140000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM2_PC1] -force
  assign_bd_address -offset 0x004180000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM3_PC0] -force
  assign_bd_address -offset 0x0041C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM3_PC1] -force
  assign_bd_address -offset 0x004200000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM4_PC0] -force
  assign_bd_address -offset 0x004240000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM4_PC1] -force
  assign_bd_address -offset 0x004280000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM5_PC0] -force
  assign_bd_address -offset 0x0042C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM5_PC1] -force
  assign_bd_address -offset 0x004300000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM6_PC0] -force
  assign_bd_address -offset 0x004340000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM6_PC1] -force
  assign_bd_address -offset 0x004380000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM7_PC0] -force
  assign_bd_address -offset 0x0043C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM7_PC1] -force
  assign_bd_address -offset 0x004400000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM8_PC0] -force
  assign_bd_address -offset 0x004440000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM8_PC1] -force
  assign_bd_address -offset 0x004480000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM9_PC0] -force
  assign_bd_address -offset 0x0044C0000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_cips/S01_AXI/HBM9_PC1] -force
  assign_bd_address -offset 0x050080000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_mc_ddr4_0/S01_INI/C1_DDR_CH1] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_mc_ddr4_0/S01_INI/C1_DDR_LOW0] -force
  assign_bd_address -offset 0x060000000000 -range 0x000800000000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs noc/axi_noc_mc_ddr4_1/S01_INI/C1_DDR_CH2] -force
  assign_bd_address -offset 0x020101010000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/base_logic/gcq_m2r/S00_AXI/S00_AXI_Reg] -force
  assign_bd_address -offset 0x020101000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/base_logic/hw_discovery/s_axi_ctrl_pf0/reg0] -force
  assign_bd_address -offset 0x020101040000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/clock_reset/pcie_mgmt_pdi_reset/pcie_mgmt_pdi_reset_gpio/S_AXI/Reg] -force
  assign_bd_address -offset 0x020101001000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/base_logic/uuid_rom/S_AXI/reg0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces aved/cips/LPD_AXI_NOC_0] [get_bd_addr_segs noc/axi_noc_mc_ddr4_0/S00_INI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x80044000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/M_AXI_LPD] [get_bd_addr_segs aved/base_logic/axi_smbus_rpu/S_AXI/Reg] -force
  assign_bd_address -offset 0x80010000 -range 0x00001000 -target_address_space [get_bd_addr_spaces aved/cips/M_AXI_LPD] [get_bd_addr_segs aved/base_logic/gcq_m2r/S01_AXI/S01_AXI_Reg] -force
  assign_bd_address -offset 0x050080000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces aved/cips/PMC_NOC_AXI_0] [get_bd_addr_segs noc/axi_noc_mc_ddr4_0/S00_INI/C0_DDR_CH1] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces aved/cips/PMC_NOC_AXI_0] [get_bd_addr_segs noc/axi_noc_mc_ddr4_0/S00_INI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x060000000000 -range 0x000800000000 -target_address_space [get_bd_addr_spaces aved/cips/PMC_NOC_AXI_0] [get_bd_addr_segs noc/axi_noc_mc_ddr4_1/S00_INI/C0_DDR_CH2] -force

  # Exclude Address Segments
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs noc/axi_noc_mc_ddr4_0/S00_INI/C0_DDR_CH1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_2]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_3]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_4]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_5]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_6]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_7]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_apu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a720_cti]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a720_dbg]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a720_etm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a720_pmu]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a721_cti]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a721_dbg]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a721_etm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a721_pmu]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_apu_cti]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_apu_ela]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_apu_etf]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_apu_fun]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_atm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_cti2a]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_cti2d]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_ela2a]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_ela2b]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_ela2c]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_ela2d]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_fun]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_rom]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_fpd_atm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_fpd_stm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_lpd_atm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_cpm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_crf_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_crl_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_crp_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_afi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_afi_2]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_cci_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_gpv_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_maincci_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_slave_xmpu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_slcr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_slcr_secure_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_smmu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_smmutcu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_gpio_2]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_i2c_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_i2c_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_3]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_4]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_5]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_6]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_pmc]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_pmc_nobuf]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_psm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_afi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_iou_secure_slcr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_iou_slcr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_slcr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_slcr_secure_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_xppu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ocm_ctrl]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ocm_ram_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ocm_xmpu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_aes]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_bbram_ctrl]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_cfi_cframe_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_cfu_apb_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_dma_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_dma_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_efuse_cache]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_efuse_ctrl]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_global_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_gpio_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_iomodule_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ospi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ppu1_mdm_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_qspi_ospi_flash_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ram]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ram_data_cntlr]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ram_instr_cntlr]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ram_npi]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_rsa]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_rtc_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_sd_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_sha]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_slave_boot]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_sysmon_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_tmr_inject_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_tmr_manager_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_trng]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_xmpu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_xppu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_xppu_npi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_psm_global_reg]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_r5_1_atcm_global]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_r5_1_btcm_global]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_r5_tcm_ram_global]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_rpu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_sbsauart_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_sbsauart_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_scntr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_scntrs_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_spi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ttc_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ttc_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ttc_2]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_0] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ttc_3]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_2]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_3]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_4]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_5]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_6]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_adma_7]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_apu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a720_cti]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a720_dbg]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a720_etm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a720_pmu]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a721_cti]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a721_dbg]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a721_etm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_a721_pmu]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_apu_cti]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_apu_ela]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_apu_etf]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_apu_fun]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_atm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_cti2a]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_cti2d]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_ela2a]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_ela2b]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_ela2c]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_ela2d]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_fun]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_cpm_rom]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_fpd_atm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_fpd_stm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_coresight_lpd_atm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_cpm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_crf_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_crl_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_crp_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_afi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_afi_2]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_cci_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_gpv_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_maincci_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_slave_xmpu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_slcr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_slcr_secure_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_smmu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_fpd_smmutcu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_gpio_2]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_i2c_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_i2c_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_3]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_4]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_5]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_6]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_pmc]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_pmc_nobuf]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ipi_psm]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_afi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_iou_secure_slcr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_iou_slcr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_slcr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_slcr_secure_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_lpd_xppu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ocm_ctrl]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ocm_ram_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ocm_xmpu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_aes]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_bbram_ctrl]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_cfi_cframe_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_cfu_apb_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_dma_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_dma_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_efuse_cache]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_efuse_ctrl]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_global_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_gpio_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_iomodule_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ospi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ppu1_mdm_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_qspi_ospi_flash_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ram]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ram_data_cntlr]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ram_instr_cntlr]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_ram_npi]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_rsa]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_rtc_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_sd_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_sha]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_slave_boot]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_slave_boot_stream]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_sysmon_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_tmr_inject_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_tmr_manager_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_trng]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_xmpu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_xppu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_pmc_xppu_npi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_psm_global_reg]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_r5_1_atcm_global]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_r5_1_btcm_global]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_r5_tcm_ram_global]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_rpu_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_sbsauart_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_sbsauart_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_scntr_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_scntrs_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_spi_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ttc_0]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ttc_1]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ttc_2]
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces aved/cips/CPM_PCIE_NOC_1] [get_bd_addr_segs aved/cips/NOC_PMC_AXI_0/pspmc_0_psv_ttc_3]
  exclude_bd_addr_seg -offset 0x050080000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces aved/cips/LPD_AXI_NOC_0] [get_bd_addr_segs noc/axi_noc_mc_ddr4_0/S00_INI/C0_DDR_CH1]


  # Restore current instance
  current_bd_instance $oldCurInst

}
# End of create_root_design()




proc available_tcl_procs { } {
   puts "##################################################################"
   puts "# Available Tcl procedures to recreate hierarchical blocks:"
   puts "#"
   puts "#    create_hier_cell_aved parentCell nameHier"
   puts "#    create_hier_cell_noc parentCell nameHier"
   puts "#    create_hier_cell_slash parentCell nameHier"
   puts "#    create_hier_cell_clock_reset parentCell nameHier"
   puts "#    create_hier_cell_base_logic parentCell nameHier"
   puts "#    create_hier_cell_pcie_mgmt_pdi_reset parentCell nameHier"
   puts "#    create_root_design"
   puts "#"
   puts "#"
   puts "# The following procedures will create hiearchical blocks with addressing "
   puts "# for IPs within those blocks and their sub-hierarchical blocks. Addressing "
   puts "# will not be handled outside those blocks:"
   puts "#"
   puts "#    create_root_design"
   puts "#"
   puts "##################################################################"
}

available_tcl_procs
