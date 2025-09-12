from rapidstream import PipelineConfig

config = PipelineConfig(
    pp_scheme="double",
)
config.save_to_file("pipeline_config.json")