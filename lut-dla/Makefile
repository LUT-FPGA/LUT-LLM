TOOL=tapa
TOP=lut_dla_core

csim: lut_dla.h lut_dla_tb.cpp
	$(TOOL) g++ -- $^ -o lut_dla

hls: lut_dla.h
	$(TOOL) compile -j 16 --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo
