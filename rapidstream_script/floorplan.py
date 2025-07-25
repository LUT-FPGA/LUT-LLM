from rapidstream import FloorplanConfig

config = FloorplanConfig(
    port_pre_assignments={".*": "SLOT_X0Y0:SLOT_X0Y1"},
    dse_range_min=0.6,
    dse_range_max=0.9,
)

config.save_to_file("floorplan_config.json")