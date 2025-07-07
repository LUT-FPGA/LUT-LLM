TOOL=tapa
TOP=rope

csim: rope.h rope_tb.cpp
	$(TOOL) g++ -- $^ -o rope 

hls: rope.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo

clean:
	rm -f rope *.xo work.out
