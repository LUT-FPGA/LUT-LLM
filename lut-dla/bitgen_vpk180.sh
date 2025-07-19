#!/bin/bash
TARGET=hw
# TARGET=hw_emu
# DEBUG=-g

TOP=lut_dla_core
XO='/scratch/oswaldhe/lut-dla/lut_dla_core.xo'
# CONSTRAINT='/home/oswaldhe/fpga_transformer/opt-fluid-model/gpt2-sa.tapa/constraints.tcl'
>&2 echo "Using the default clock target of the platform."
PLATFORM="/home/oswaldhe/vpk180_linux_platform/vpk180_pfm_vitis/export/vpk180_pfm_vitis/vpk180_pfm_vitis.xpfm"
TARGET_FREQUENCY=300000000
if [ -z $PLATFORM ]; then echo Please edit this file and set a valid PLATFORM= on line "${LINENO}"; exit; fi

OUTPUT_DIR="$(pwd)/vitis_run_${TARGET}"

MAX_SYNTH_JOBS=32
STRATEGY="Explore"
PLACEMENT_STRATEGY="Explore"

v++ ${DEBUG} \
  --link \
  --output "${OUTPUT_DIR}/${TOP}_vpk180.xsa" \
  --kernel ${TOP} \
  --platform ${PLATFORM} \
  --target ${TARGET} \
  --report_level 2 \
  --temp_dir "${OUTPUT_DIR}/${TOP}_vpk180.temp" \
  --optimize 3 \
  --connectivity.nk ${TOP}:1:${TOP} \
  --save-temps \
  "${XO}" \
  --vivado.synth.jobs ${MAX_SYNTH_JOBS} \
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=1 \
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=$STRATEGY \
  --vivado.prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=$PLACEMENT_STRATEGY \
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=$STRATEGY \
  --vivado.prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE="AggressiveExplore" \
  --clock.default_freqhz ${TARGET_FREQUENCY}