#ifndef _QWEN_BLOCK_H_
#define _QWEN_BLOCK_H_

#include <tapa.h>
#include <ap_int.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <limits>
#include "../attention_block/attention_block.h"
#include "../ffn/ffn.h"
#include "../rms_norm/rms_norm.h"

void residual_bank(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& ffn_in_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& attn_in_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& norm_fifo
) {

    float residual_buf[MAX_SEQ_LEN][HIDDEN_DIM];
    #pragma HLS array_partition variable=residual_buf cyclic factor=16 dim=1

    // 1. read from input
    for (int i = 0; i < (L >> 4); i++) {
        LOG(INFO) << "input read " << i;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS pipeline II=1
            auto input_vec = input_fifo.read();
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                residual_buf[i*16+k][j] = input_vec[k];
            }
            norm_fifo.write(input_vec); // write to norm fifo
        }
    }

    for (int seq = 0; seq < 4 && seq < L; seq++) {
        std::string log_str = "r=" + std::to_string(0) + " res_buf[" + std::to_string(seq) + "] = [";
        for (int dim = 0; dim < 16; dim++) {
            log_str += std::to_string(residual_buf[seq][dim]);
            if (dim < 15) log_str += ", ";
        }
        log_str += "]";
        LOG(INFO) << log_str;
    }

    // 2. read from attention layer
    for (int i = 0; i < (L >> 4); i++) {
        LOG(INFO) << "attn read " << i;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS pipeline II=1
            auto attn_vec = attn_in_fifo.read();
            tapa::vec_t<float, 16> tmp;
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                residual_buf[i*16+k][j] += attn_vec[k]; // accumulate
                tmp[k] = residual_buf[i*16+k][j]; // prepare for output
            }
            norm_fifo.write(tmp); // write to norm fifo
        }
    }

    for (int seq = 0; seq < 4 && seq < L; seq++) {
        std::string log_str = "r=" + std::to_string(1) + " res_buf[" + std::to_string(seq) + "] = [";
        for (int dim = 0; dim < 16; dim++) {
            log_str += std::to_string(residual_buf[seq][dim]);
            if (dim < 15) log_str += ", ";
        }
        log_str += "]";
        LOG(INFO) << log_str;
    }

    // 3. read from ffn layer
    for (int i = 0; i < (L >> 4); i++) {
        LOG(INFO) << "ffn read " << i;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            #pragma HLS pipeline II=1
            auto ffn_vec = ffn_in_fifo.read();
            tapa::vec_t<float, 16> tmp;
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                residual_buf[i*16+k][j] += ffn_vec[k]; // accumulate
                tmp[k] = residual_buf[i*16+k][j]; // prepare for output
            }
            norm_fifo.write(tmp); // write to norm fifo
        }
    }

    for (int seq = 0; seq < 4 && seq < L; seq++) {
        std::string log_str = "r=" + std::to_string(2) + " res_buf[" + std::to_string(seq) + "] = [";
        for (int dim = 0; dim < 16; dim++) {
            log_str += std::to_string(residual_buf[seq][dim]);
            if (dim < 15) log_str += ", ";
        }
        log_str += "]";
        LOG(INFO) << log_str;
    }
}

void qwen_block(
    const int L,
    tapa::mmap<tapa::vec_t<float, 16>> input_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> rms_weight_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> qk_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> v_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> out_proj_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> qk_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> v_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> out_proj_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> sin_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> cos_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> up_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> gate_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> down_centroid_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> up_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> gate_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> down_lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> layer_out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 16>> input_fifo("input_fifo");

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

    tapa::stream<tapa::vec_t<float, 16>> residual_to_norm_fifo("residual_to_norm_fifo");
    tapa::stream<tapa::vec_t<float, 2>> norm_to_attn_fifo("norm_to_attn_fifo");
    tapa::stream<tapa::vec_t<float, 2>> norm_to_ffn_fifo("norm_to_ffn_fifo");
    tapa::stream<tapa::vec_t<float, 16>> norm_out_fifo("norm_out_fifo");
    tapa::stream<tapa::vec_t<float, 16>> norm_weight_fifo("norm_weight_fifo");

    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(input_reader_wide, L, HIDDEN_DIM, input_buffer, input_fifo)
        .invoke<tapa::join>(residual_bank, L, input_fifo, down_out_fifo, attn_out_fifo, residual_to_norm_fifo)
        .invoke<tapa::join>(rms_weight_reader, rms_weight_buffer, norm_weight_fifo)
        .invoke<tapa::join>(rms_norm_cache, L, residual_to_norm_fifo, norm_weight_fifo, norm_to_attn_fifo, norm_to_ffn_fifo, norm_out_fifo)
        // attention block
        .invoke<tapa::join>(repeater, L, norm_to_attn_fifo, qk_in_fifo, v_in_fifo)
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
        .invoke<tapa::join>(memory_matcher_out_proj, L, HIDDEN_DIM_DIV_2, out_proj_idx_fifo, out_proj_lut_fifo, attn_out_fifo)
        // feedforward layer
        .invoke<tapa::join>(repeater, L, norm_to_ffn_fifo, up_in_fifo, gate_in_fifo)
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
        .invoke<tapa::join>(memory_matcher_out_proj, L, INTERM_DIM_DIV_2, down_idx_fifo, down_lut_fifo, down_out_fifo)
        .invoke<tapa::join>(linear_out_writer, L, HIDDEN_DIM, norm_out_fifo, layer_out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}

#endif // _QWEN_BLOCK_H_