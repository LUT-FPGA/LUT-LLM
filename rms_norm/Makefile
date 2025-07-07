TOOL=tapa
TOP=rms_norm_top

csim: rms_norm.h rms_norm_tb.cpp
	$(TOOL) g++ -- $^ -o rms_norm

hls: rms_norm.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo
