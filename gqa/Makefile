TOOL=tapa
TOP=gqa

csim: gqa.h gqa_tb.cpp
	$(TOOL) g++ -- $^ -o gqa

hls: gqa.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo

clean:
	rm -f gqa *.xo work.out
