TOOL=tapa
TOP=ffn_core

csim: ffn.h ffn_tb.cpp
	$(TOOL) g++ -- $^ -o ffn

hls: ffn.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo

clean:
	rm -f ffn *.xo work.out 