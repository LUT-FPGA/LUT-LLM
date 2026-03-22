from rapidstream import FloorplanConfig

config = FloorplanConfig(
    port_pre_assignments={".*": "SLOT_X0Y2:SLOT_X1Y2"},
    cell_pre_assignments={
        "memory_matcher_w_vq_half_dsp_final_int4_0": "SLOT_X1Y0:SLOT_X1Y0"
    },
    dse_range_min=0.65,
    dse_range_max=0.90,
)

config.save_to_file("floorplan_config.json")