from rapidstream import FloorplanConfig

config = FloorplanConfig(
    port_pre_assignments={".*": "SLOT_X0Y2:SLOT_X0Y2"},
    cell_pre_assignments={
        "gemm_gqa_av_0": "SLOT_X0Y0:SLOT_X0Y0",
        "gemm_gqa_qk_0": "SLOT_X0Y0:SLOT_X0Y0",
        "softmax_0": "SLOT_X0Y0:SLOT_X0Y0",
        "element_wise_mul_0": "SLOT_X0Y0:SLOT_X0Y0",
        "distributor_0": "SLOT_X0Y0:SLOT_X0Y0",
        "memory_matcher_acc_overlay_half_0": "SLOT_X0Y1:SLOT_X0Y1",
        "memory_matcher_w_vq_head_half_final_0": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_dsp_final_0": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_final_0": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_dsp_final_1": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_final_1": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_dsp_final_2": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_final_2": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_dsp_final_3": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_final_3": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_dsp_final_4": "SLOT_X0Y2:SLOT_X0Y2",
        "memory_matcher_w_vq_half_final_4": "SLOT_X0Y2:SLOT_X0Y2"
    },
    dse_range_min=0.65,
    dse_range_max=0.90,
)

config.save_to_file("floorplan_config.json")