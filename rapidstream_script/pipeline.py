from rapidstream import PipelineConfig

config = PipelineConfig(
    pp_scheme="single",
)
config.save_to_file("pipeline_config.json")