#!/bin/bash

rapidstream-tapaopt \
    -j 6 \
    --work-dir ./build \
    --tapa-xo-path qwen_block.xo \
    --device-config v80_device.json \
    --floorplan-config floorplan_config.json \
    --pipeline-config pipeline_config.json