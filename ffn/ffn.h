#ifndef _FFN_H_
#define _FFN_H_

#include <tapa.h>
#include <ap_int.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <limits>
#include "../imm/imm.h"
#include "../ccu/ccu_fp32.h"
#include "../silu/silu.h"

constexpr int HIDDEN_DIM = 896;
constexpr int INTERM_DIM = 4864;
constexpr int HIDDEN_DIM_DIV_2 = HIDDEN_DIM / 2;
constexpr int INTERM_DIM_DIV_2 = INTERM_DIM / 2;

template<int out_dim=INTERM_DIM>
void combiner(
    const int L,
    tapa::istream<tapa::vec_t<float, 2>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    for(int i = 0; i < (out_dim >> 1); i++) {
        for (int j = 0; j < (L >> 3); j++) {
            tapa::vec_t<float, 16> tmp;
            for(int k = 0; k < 8; k++) {
                #pragma HLS pipeline II=1
                auto tmp_small = input_fifo.read();
                tmp[k*2] = tmp_small[0];
                tmp[k*2 + 1] = tmp_small[1];
            }
            out_fifo.write(tmp);
        }
    }
}

void combiner_mid(
    const int L,
    tapa::istream<tapa::vec_t<float, 2>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    combiner<INTERM_DIM>(L, input_fifo, out_fifo);
}

void combiner_end(
    const int L,
    tapa::istream<tapa::vec_t<float, 2>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    combiner<HIDDEN_DIM>(L, input_fifo, out_fifo);
}

void splitter(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& out_fifo
) {
    for(int i = 0; i < INTERM_DIM_DIV_2; i++) {
        for (int j = 0; j < (L >> 3); j++) {
            auto tmp = input_fifo.read();
            for(int k = 0; k < 8; k++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 2> tmp_small;
                tmp_small[0] = tmp[k*2];
                tmp_small[1] = tmp[k*2 + 1];
                out_fifo.write(tmp_small);
            }
        }
    }
}

void repeater(
    const int L,
    tapa::istream<tapa::vec_t<float, 2>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& up_in_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& gate_in_fifo
) {
    for(int i = 0; i < HIDDEN_DIM_DIV_2; i++) {
        for (int j = 0; j < L;) {
            #pragma HLS pipeline II=1
            if (!input_fifo.empty()) {
                tapa::vec_t<float, 2> tmp; input_fifo.try_read(tmp);
                up_in_fifo.write(tmp);
                gate_in_fifo.write(tmp);
                j++;
            }
        }
    }
}

void element_wise_mul(
    const int L,
    tapa::istream<tapa::vec_t<float, 2>>& up_fifo,
    tapa::istream<tapa::vec_t<float, 2>>& gate_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& out_fifo
) {
    for(int i = 0; i < INTERM_DIM_DIV_2; i++) {
        for(int j = 0; j < L;) {
            #pragma HLS pipeline II=1
            if (!up_fifo.empty() & !gate_fifo.empty()) {
                tapa::vec_t<float, 2> up; up_fifo.try_read(up);
                tapa::vec_t<float, 2> gate; gate_fifo.try_read(gate);
                tapa::vec_t<float, 2> out;
                out[0] = up[0] * gate[0];
                out[1] = up[1] * gate[1];
                out_fifo.write(out);
                j++;
            }
        }
    }
}

void memory_matcher_up_gate(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istream<ap_uint<8>>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& out_fifo
) {
    memory_matcher<2, 32>(L, in_size, out_size, idx_fifo, lut_fifo, out_fifo);
}

void memory_matcher_down(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istream<ap_uint<8>>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    memory_matcher<16, 128>(L, in_size, out_size, idx_fifo, lut_fifo, out_fifo);
}


void ffn_core(
    const int L,
    tapa::mmap<tapa::vec_t<float, 2>> input_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> up_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> gate_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> down_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> up_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> gate_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> down_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> ffn_out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 2>> input_fifo("input_fifo");

    tapa::stream<tapa::vec_t<float, 2>> up_in_fifo("up_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>> up_centroid_fifo("up_centroid_fifo");
    tapa::stream<ap_uint<8>, 16> up_idx_fifo("up_idx_fifo");
    tapa::stream<tapa::vec_t<ap_uint<64>, 8>> up_lut_fifo("up_lut_fifo");
    tapa::stream<tapa::vec_t<float, 2>, 16> up_out_fifo("up_out_fifo");

    tapa::stream<tapa::vec_t<float, 2>> gate_in_fifo("gate_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>> gate_centroid_fifo("gate_centroid_fifo");
    tapa::stream<ap_uint<8>, 16> gate_idx_fifo("gate_idx_fifo");
    tapa::stream<tapa::vec_t<ap_uint<64>, 8>> gate_lut_fifo("gate_lut_fifo");
    tapa::stream<tapa::vec_t<float, 2>> gate_out_fifo("gate_out_fifo");
    tapa::stream<tapa::vec_t<float, 16>> gate_before_silu_fifo("gate_before_silu_fifo");
    tapa::stream<tapa::vec_t<float, 16>> gate_after_silu_fifo("gate_after_silu_fifo");
    tapa::stream<tapa::vec_t<float, 2>> gate_out_fifo_split("gate_out_fifo_split");

    tapa::stream<tapa::vec_t<float, 2>> down_in_fifo("down_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>> down_centroid_fifo("down_centroid_fifo");
    tapa::stream<ap_uint<8>, 16> down_idx_fifo("down_idx_fifo");
    tapa::stream<tapa::vec_t<ap_uint<64>, 8>> down_lut_fifo("down_lut_fifo");
    tapa::stream<tapa::vec_t<float, 16>> down_out_fifo("down_out_fifo");

    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(input_reader, L, HIDDEN_DIM_DIV_2, input_buffer, input_fifo)
        .invoke<tapa::join>(repeater, L, input_fifo, up_in_fifo, gate_in_fifo)
        .invoke<tapa::join>(centroid_reader, HIDDEN_DIM_DIV_2, up_centroid_buffer, up_centroid_fifo)
        .invoke<tapa::join>(lut_reader, HIDDEN_DIM_DIV_2, INTERM_DIM, up_lut_buffer, up_lut_fifo)
        .invoke<tapa::join>(ccu_fp32, L, HIDDEN_DIM_DIV_2, up_in_fifo, up_centroid_fifo, up_idx_fifo)
        .invoke<tapa::join>(memory_matcher_up_gate, L, HIDDEN_DIM_DIV_2, INTERM_DIM, up_idx_fifo, up_lut_fifo, up_out_fifo)
        .invoke<tapa::join>(centroid_reader, HIDDEN_DIM_DIV_2, gate_centroid_buffer, gate_centroid_fifo)
        .invoke<tapa::join>(lut_reader, HIDDEN_DIM_DIV_2, INTERM_DIM, gate_lut_buffer, gate_lut_fifo)
        .invoke<tapa::join>(ccu_fp32, L, HIDDEN_DIM_DIV_2, gate_in_fifo, gate_centroid_fifo, gate_idx_fifo)
        .invoke<tapa::join>(memory_matcher_up_gate, L, HIDDEN_DIM_DIV_2, INTERM_DIM, gate_idx_fifo, gate_lut_fifo, gate_out_fifo)
        .invoke<tapa::join>(combiner_mid, L, gate_out_fifo, gate_before_silu_fifo)
        .invoke<tapa::join>(silu, L, gate_before_silu_fifo, gate_after_silu_fifo)
        .invoke<tapa::join>(splitter, L, gate_after_silu_fifo, gate_out_fifo_split)
        .invoke<tapa::join>(element_wise_mul, L, up_out_fifo, gate_out_fifo_split, down_in_fifo)
        .invoke<tapa::join>(centroid_reader, INTERM_DIM_DIV_2, down_centroid_buffer, down_centroid_fifo)
        .invoke<tapa::join>(lut_reader, INTERM_DIM_DIV_2, HIDDEN_DIM, down_lut_buffer, down_lut_fifo)
        .invoke<tapa::join>(ccu_fp32, L, INTERM_DIM_DIV_2, down_in_fifo, down_centroid_fifo, down_idx_fifo)
        .invoke<tapa::join>(memory_matcher_down, L, INTERM_DIM_DIV_2, HIDDEN_DIM, down_idx_fifo, down_lut_fifo, down_out_fifo)
        .invoke<tapa::join>(linear_out_writer, L, HIDDEN_DIM, down_out_fifo, ffn_out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}


#endif // _FFN_H_