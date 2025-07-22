
proc getEnvInt { varName defaultIntValue } {
    set value [expr {[info exists ::env($varName)] ?$::env($varName) :$defaultIntValue}]
    return [expr {int($value)}]
}

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


create_project vivado_proj serpens32_none/vivado_proj -part xcv80-lsva4737-2MHP-e-S
import_ips_from_dir serpens32_none/rtl/
import_files serpens32_none/rtl/

# set_property SOURCE_SET sources_1 [get_filesets sim_1]
# import_files -fileset sim_1 -norecurse serpens32_none/tb.sv
# set_property top tb [get_filesets sim_1]
# set_property top_lib xil_defaultlib [get_filesets sim_1]
# update_compile_order -fileset sim_1
# set_property -name {xsim.simulate.log_all_signals} -value {true}     -objects [get_filesets sim_1]

set constr_file [import_files -fileset constrs_1 serpens32_none/constraint.tcl]
set_property used_in_synthesis false $constr_file

upgrade_ip -quiet [get_ips *]
generate_target synthesis [ get_files *.xci ]

source serpens32_none/arm_bd.tcl
source serpens32_none/noc_constraint.tcl
validate_bd_design
save_bd_design
make_wrapper -files [get_files top_arm.bd] -top -import
set_property top top_arm_wrapper [current_fileset]
generate_target all [get_files top_arm.bd]

set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}     -value {-mode out_of_context}     -objects [get_runs synth_1]
launch_runs synth_1 -jobs [getEnvInt VIVADO_SYNTH_JOBS 4]
wait_on_run synth_1


# open_project serpens32_none/vivado_proj/vivado_proj.xpr
# reset_run impl_1 -prev_step 
set_property -name {STEPS.PHYS_OPT_DESIGN.IS_ENABLED}     -value {1}     -objects [get_runs impl_1]
set_property -name {STEPS.OPT_DESIGN.ARGS.DIRECTIVE}     -value {Explore}     -objects [get_runs impl_1]
set_property -name {STEPS.PLACE_DESIGN.ARGS.DIRECTIVE}     -value {Explore}     -objects [get_runs impl_1]
set_property -name {STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE}     -value {Explore}     -objects [get_runs impl_1]
set_property -name {STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE}     -value {AggressiveExplore}     -objects [get_runs impl_1]
launch_runs impl_1 -jobs 8
wait_on_run impl_1

close_project