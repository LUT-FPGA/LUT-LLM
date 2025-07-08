TOOL=tapa
TOP=attention_block

csim: attention_block.h attention_block_tb.cpp
	$(TOOL) g++ -- $^ -o attention_block

hls: attention_block.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo

clean:
	rm -f attention_block *.xo work.out
