TOOL=tapa
TOP=qwen_block

csim: qwen_block.h qwen_block_tb.cpp
	$(TOOL) g++ -- $^ -o qwen_block

csim_decode: qwen_block.h qwen_block_decode_tb.cpp
	$(TOOL) g++ -- $^ -o qwen_block_decode

hls: qwen_block.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo

clean:
	rm -f qwen_block *.xo work.out
