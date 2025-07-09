#ifndef _ATTN_H_
#define _ATTN_H_

#include <tapa.h>
#include <ap_int.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <limits>
#include "../imm/imm.h"
#include "../ccu/ccu_fp32.h"
#include "../rope/rope.h"
#include "../gqa/gqa.h"
#include "../config/config.h"

void repeater(
    const int L,
    tapa::istream<tapa::vec_t<float, 2>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& qk_in_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& v_in_fifo
) {
    for(int i = 0; i < HIDDEN_DIM_DIV_2; i++) {
        for (int j = 0; j < L;) {
            #pragma HLS pipeline II=1
            if (!input_fifo.empty()) {
                tapa::vec_t<float, 2> tmp; input_fifo.try_read(tmp);
                qk_in_fifo.write(tmp);
                v_in_fifo.write(tmp);
                j++;
            }
        }
    }
}

void apply_rope(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& sin_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& cos_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    apply_rotary_pos_emb<16>(
        L, input_fifo, sin_fifo, cos_fifo, out_fifo
    );
}

void transpose_head(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {

    for(int r = 0; r < 16; r++){

        float head_buf[MAX_SEQ_LEN][HEAD_DIM];
        #pragma HLS array_partition variable=head_buf cyclic factor=16 dim=1
        #pragma HLS array_partition variable=head_buf cyclic factor=16 dim=2

        read_data: for (int i = 0; i < L; i++) {
            for (int j = 0; j < (HEAD_DIM >> 4); j++) {
                #pragma HLS pipeline II=1
                auto tmp = input_fifo.read();
                for (int k = 0; k < 16; k++) {
                    head_buf[i][j*16+k] = tmp[k];
                }
            }
        }

        write_tranpose: for (int i = 0; i < HEAD_DIM; i++) {
            for (int j = 0; j < (L >> 4); j++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> tmp;
                for (int k = 0; k < 16; k++) {
                    tmp[k] = head_buf[j*16+k][i];
                }
                out_fifo.write(tmp);
            }
        }
    }
}

void transpose_vq(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& out_fifo
) {

    for(int r = 0; r < 14; r++){

        float head_buf[MAX_SEQ_LEN][HEAD_DIM];
        #pragma HLS array_partition variable=head_buf cyclic factor=16 dim=2

        read_data: for (int i = 0; i < L; i++) {
            for (int j = 0; j < (HEAD_DIM >> 4); j++) {
                #pragma HLS pipeline II=1
                auto tmp = input_fifo.read();
                for (int k = 0; k < 16; k++) {
                    head_buf[i][j*16+k] = tmp[k];
                }
            }
        }

        write_vec_transpose: for (int i = 0; i < HEAD_DIM_DIV_2; i++) {
            for (int j = 0; j < L; j++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 2> tmp;
                for (int k = 0; k < 2; k++) {
                    tmp[k] = head_buf[j][i*2+k];
                }
                out_fifo.write(tmp);
            }
        }
    }
}

void gqa_arbiter(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& qk_in_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& v_in_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    for(int r = 0; r < 18; r++) {
        for(int i = 0; i < (L * HEAD_DIM >> 4); i++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 16> tmp;
            if (r == 0 || r == 9) {
                tmp = v_in_fifo.read();
            } else {
                tmp = qk_in_fifo.read();
            }
            out_fifo.write(tmp);
        }
    }
}

void memory_matcher_qk(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    memory_matcher_attn<QK_DIM_DIV_2, 128>(L, in_size, out_size, idx_fifo, lut_fifo, out_fifo);
}

void memory_matcher_v(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    memory_matcher_attn<V_DIM_DIV_2, 32>(L, in_size, out_size, idx_fifo, lut_fifo, out_fifo);
}

void memory_matcher_out(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    memory_matcher_attn<HIDDEN_DIM_DIV_2, 32>(L, in_size, out_size, idx_fifo, lut_fifo, out_fifo);
}



void attention_block(
    const int L,
    tapa::mmap<tapa::vec_t<float, 2>> input_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> qk_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> v_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> out_proj_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> qk_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> v_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> out_proj_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> sin_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> cos_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> attn_out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 2>> input_fifo("input_fifo");

    tapa::stream<tapa::vec_t<float, 2>> qk_in_fifo("qk_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>> qk_centroid_fifo("qk_centroid_fifo");
    tapa::stream<ap_uint<8>, 16> qk_idx_fifo("qk_idx_fifo");
    tapa::stream<tapa::vec_t<ap_uint<64>, 8>> qk_lut_fifo("qk_lut_fifo");
    tapa::stream<tapa::vec_t<float, 16>> qk_out_fifo("qk_out_fifo");

    tapa::stream<tapa::vec_t<float, 16>> qk_rope_in_fifo("qk_rope_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>> qk_rope_out_fifo("qk_rope_out_fifo");
    tapa::stream<tapa::vec_t<float, 16>> rope_sin_fifo("rope_sin_fifo");
    tapa::stream<tapa::vec_t<float, 16>> rope_cos_fifo("rope_cos_fifo");

    tapa::stream<tapa::vec_t<float, 2>> v_in_fifo("v_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>> v_centroid_fifo("v_centroid_fifo");
    tapa::stream<ap_uint<8>, 16> v_idx_fifo("v_idx_fifo");
    tapa::stream<tapa::vec_t<ap_uint<64>, 8>> v_lut_fifo("v_lut_fifo");
    tapa::stream<tapa::vec_t<float, 16>, 16> v_out_fifo("v_out_fifo");

    tapa::stream<tapa::vec_t<float, 16>> gqa_in_fifo("gqa_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>> gqa_out_fifo("gqa_out_fifo");
    tapa::stream<tapa::vec_t<float, 16>> pre_softmax_fifo("pre_softmax_fifo");
    tapa::stream<tapa::vec_t<float, 16>> post_softmax_fifo("post_softmax_fifo");

    tapa::stream<tapa::vec_t<float, 2>> out_proj_in_fifo("out_proj_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_proj_centroid_fifo("out_proj_centroid_fifo");
    tapa::stream<ap_uint<8>, 16> out_proj_idx_fifo("out_proj_idx_fifo");
    tapa::stream<tapa::vec_t<ap_uint<64>, 8>> out_proj_lut_fifo("out_proj_lut_fifo");
    tapa::stream<tapa::vec_t<float, 16>> attn_out_fifo("attn_out_fifo");

    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(input_reader, L, HIDDEN_DIM_DIV_2, input_buffer, input_fifo)
        .invoke<tapa::join>(repeater, L, input_fifo, qk_in_fifo, v_in_fifo)
        .invoke<tapa::join>(centroid_reader, HIDDEN_DIM_DIV_2, qk_centroid_buffer, qk_centroid_fifo)
        .invoke<tapa::join>(lut_reader, HIDDEN_DIM_DIV_2, QK_DIM, qk_lut_buffer, qk_lut_fifo)
        .invoke<tapa::join>(ccu_fp32, L, HIDDEN_DIM_DIV_2, qk_in_fifo, qk_centroid_fifo, qk_idx_fifo)
        .invoke<tapa::join>(memory_matcher_qk, L, HIDDEN_DIM_DIV_2, QK_DIM, qk_idx_fifo, qk_lut_fifo, qk_rope_in_fifo)
        .invoke<tapa::join>(rope_input_reader, L, sin_buffer, rope_sin_fifo)
        .invoke<tapa::join>(rope_input_reader, L, cos_buffer, rope_cos_fifo)
        .invoke<tapa::join>(apply_rope, L, qk_rope_in_fifo, rope_sin_fifo, rope_cos_fifo, qk_rope_out_fifo)
        .invoke<tapa::join>(transpose_head, L, qk_rope_out_fifo, qk_out_fifo)
        .invoke<tapa::join>(centroid_reader, HIDDEN_DIM_DIV_2, v_centroid_buffer, v_centroid_fifo)
        .invoke<tapa::join>(lut_reader, HIDDEN_DIM_DIV_2, V_DIM, v_lut_buffer, v_lut_fifo)
        .invoke<tapa::join>(ccu_fp32, L, HIDDEN_DIM_DIV_2, v_in_fifo, v_centroid_fifo, v_idx_fifo)
        .invoke<tapa::join>(memory_matcher_v, L, HIDDEN_DIM_DIV_2, V_DIM, v_idx_fifo, v_lut_fifo, v_out_fifo)
        .invoke<tapa::join>(gqa_arbiter, L, qk_out_fifo, v_out_fifo, gqa_in_fifo)
        .invoke<tapa::join>(gemm_gqa, L, gqa_in_fifo, pre_softmax_fifo, post_softmax_fifo, gqa_out_fifo)
        .invoke<tapa::join>(softmax, L, pre_softmax_fifo, post_softmax_fifo)
        .invoke<tapa::join>(transpose_vq, L, gqa_out_fifo, out_proj_in_fifo)
        .invoke<tapa::join>(centroid_reader, HIDDEN_DIM_DIV_2, out_proj_centroid_buffer, out_proj_centroid_fifo)
        .invoke<tapa::join>(lut_reader, HIDDEN_DIM_DIV_2, HIDDEN_DIM, out_proj_lut_buffer, out_proj_lut_fifo)
        .invoke<tapa::join>(ccu_fp32, L, HIDDEN_DIM_DIV_2, out_proj_in_fifo, out_proj_centroid_fifo, out_proj_idx_fifo)
        .invoke<tapa::join>(memory_matcher_out, L, HIDDEN_DIM_DIV_2, HIDDEN_DIM, out_proj_idx_fifo, out_proj_lut_fifo, attn_out_fifo)
        .invoke<tapa::join>(linear_out_writer, L, HIDDEN_DIM, attn_out_fifo, attn_out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}


#endif // _ATTN_H_