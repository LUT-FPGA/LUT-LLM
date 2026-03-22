TOOL=tapa
TOP=qwen_block

csim: qwen_block.h qwen_block_tb.cpp
	$(TOOL) g++ -- $^ -o qwen_block

csim_decode: qwen_block.h qwen_block_decode_tb.cpp
	$(TOOL) g++ -- $^ -o qwen_block_decode

e2e_latency: e2e_latency.cpp
	g++ -std=c++17 -O2 $< -o e2e_latency

hls: qwen_block.h
	$(TOOL) compile --top $(TOP) --part-num xcv80-lsva4737-2MHP-e-S --clock-period 3.33 -f $< -o $(TOP).xo

clean:
	rm -f qwen_block qwen_block_decode e2e_latency *.xo work.out
