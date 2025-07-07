TOOL=tapa
TOP=silu_top

csim: silu.h silu_tb.cpp
	$(TOOL) g++ -- $^ -o silu

hls: silu.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo

clean:
	rm -f silu *.xo work.out
