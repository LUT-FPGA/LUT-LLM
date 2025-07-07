TOOL=tapa
TOP=ccu_fp32_top

csim: ccu_fp32.h ccu_fp32_tb.cpp
	$(TOOL) g++ -- $^ -o ccu_fp32

hls: ccu_fp32.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 --enable-synth-util -f $< -o $(TOP).xo