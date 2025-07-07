TOOL=tapa
TOP=imm

csim: imm.h imm_tb.cpp
	$(TOOL) g++ -- $^ -o imm -g

hls: imm.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 --enable-synth-util -f $< -o $(TOP).xo

clean:
	rm -f imm $(TOP).xo

.PHONY: csim hls clean
