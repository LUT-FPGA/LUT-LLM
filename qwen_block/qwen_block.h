#ifndef _QWEN_BLOCK_H_
#define _QWEN_BLOCK_H_

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
#include "../silu/silu.h"
#include "../rms_norm/rms_norm.h"

void residual_bank(
    tapa::istream<int>& L_in_fifo,
    tapa::ostream<int>& L_out_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& linear_in_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& norm_fifo
) {

    float residual_buf[MAX_SEQ_LEN][HIDDEN_DIM];
    #pragma HLS array_partition variable=residual_buf cyclic factor=16 dim=2

    const int L = L_in_fifo.read();
    L_out_fifo.write(L);

    // 1. read from input
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < (HIDDEN_DIM >> 4); j++) {
            #pragma HLS pipeline II=1
            auto input_vec = input_fifo.read();
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                residual_buf[i][j*16+k] = input_vec[k];
            }
            norm_fifo.write(input_vec); // write to norm fifo
        }
    }

    for (int r = 0; r < 2; r++){
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < (HIDDEN_DIM >> 4); j++) {
                #pragma HLS pipeline II=1

                auto tmp_rd_vec = linear_in_fifo.read();
                tapa::vec_t<float, 16> tmp;
                for (int k = 0; k < 16; k++) {
                    #pragma HLS unroll
                    residual_buf[i][j*16+k] += tmp_rd_vec[k]; // accumulate
                    tmp[k] = residual_buf[i][j*16+k]; // prepare for output
                }
                norm_fifo.write(tmp); // write to norm fifo
            }
        }
    }

}


void memory_matcher_acc_overlay_half(
    tapa::istream<int>& L_in_fifo,
    tapa::ostream<int>& L_out_rope_fifo,
    tapa::ostream<int>& L_out_silu_fifo,
    tapa::istreams<tapa::vec_t<ap_uint<48>, 8>, 16>& inbound_fifo, // interleave up and gate
    tapa::istream<ap_uint<64>>& scale_zero_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& rope_fifo, // stream to rope
    tapa::ostream<tapa::vec_t<float, 16>>& v_fifo,
    tapa::ostream<tapa::vec_t<float, 32>>& up_out_fifo, // stream to splitter
    tapa::ostream<tapa::vec_t<float, 32>>& gate_out_fifo, // stream to silu
    tapa::ostream<tapa::vec_t<float, 16>>& res_fifo // stream to residual bank
) {
    ap_uint<64> linear_out[MAX_SEQ_LEN][MAX_OUT_SIZE_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=256 dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_2P impl=URAM

    const int L = L_in_fifo.read();
    L_out_rope_fifo.write(L);
    L_out_silu_fifo.write(L);

    for (int round = 0; round < 4; round++) {

        for (int i = 0; i < L; i++) {
            for (int j = 0; j < (INTERM_DIM >> 8); j++){
                #pragma HLS pipeline II=1
                for (int k = 0; k < 256; k++) {
                    #pragma HLS unroll
                    linear_out[i][(j << 8) + k] = ap_uint<64>(0); // Initialize output
                }
            }
        }
        
        ap_uint<64> pack_dequant[TOTAL_HEADS];
        // qkv: all
        // out: 0
        // up/gate: 0, 1
        // down: 0
        
        if (round == 0) {
            for(int i = 0; i < TOTAL_HEADS; i++) {
                #pragma HLS pipeline II=1
                pack_dequant[i] = scale_zero_fifo.read();
            }
        } else if (round == 2) {
            for(int i = 0; i < 2; i++){
                #pragma HLS pipeline II=1
                pack_dequant[i] = scale_zero_fifo.read();
            }
        } else {
            pack_dequant[0] = scale_zero_fifo.read(); 
        }

        int r_bound = (round == 3) ? (INTERM_DIM_DIV_2 >> 4) : (HIDDEN_DIM_DIV_2 >> 4);
        int j_bound = (round == 0) ? (QKV_DIM >> 9) : (HIDDEN_DIM >> 9);
        if (round == 2) j_bound = (INTERM_DIM >> 8);

        // read indices and parallel match
        for (int r = 0; r < r_bound; r++) {

            for (int i = 0; i < L; i++) {

                for (int j = 0; j < j_bound; j++) {
                    #pragma HLS pipeline II=1
                    
                    for (int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        auto tmp_vec = inbound_fifo[k].read();
                        for (int m = 0; m < 8; m++) {
                            #pragma HLS unroll
                            ap_uint<48> op2 = tmp_vec[m];
                            for (int p = 0; p < 2; p++) {
                                #pragma HLS unroll
                                ap_uint<64> acc_reg = linear_out[i][j*256 + k*16 + m*2 + p];
                                ap_uint<44> simd_a = ap_uint<44>(acc_reg(43, 0));
                                ap_uint<44> simd_b = ap_uint<44>((ap_uint<22>(op2(p*24+23, p*24+12)),ap_uint<22>(op2(p*24+11, p*24))));
                                ap_uint<44> simd_out = simd_a + simd_b;
                                #pragma HLS bind_op variable=simd_out op=add impl=dsp
                                linear_out[i][j*256 + k*16 + m * 2 + p] = simd_out;
                            }
                        }
                    }
                }
            }
        }

        if (round == 0) { //  write qkv heads
            for (int h = 0; h < TOTAL_HEADS; h++) {
                float scale = tapa::bit_cast<float>(ap_uint<32>(pack_dequant[h](31, 0)));
                float zeropoint = tapa::bit_cast<float>(ap_uint<32>(pack_dequant[h](63, 32)));
                for(int i = 0; i < L; i++) {
                    for(int j = 0; j < (HEAD_DIM >> 4); j++) {
                        #pragma HLS pipeline II=1
                        tapa::vec_t<float, 16> tmp;
                        for (int k = 0; k < 8; k++) {
                            #pragma HLS unroll
                            tmp[k*2] = (float) (ap_uint<22>(linear_out[i][h*HEAD_DIM_DIV_2 + j * 8 + k](21, 0)).to_int()) * scale - zeropoint;
                            tmp[k*2 + 1] = (float) (ap_uint<22>(linear_out[i][h*HEAD_DIM_DIV_2 + j * 8 + k](43, 22)).to_int()) * scale - zeropoint;
                        }
                        if ((h % (HEAD_PER_GROUP + 2)) == 0) {
                            v_fifo.write(tmp);
                        } else {
                            rope_fifo.write(tmp);
                        }
                    }
                }
            }
        } else if (round == 2) {
            for (int i = 0; i < (INTERM_DIM >> 4); i++) {
                for (int j = 0; j < L; j++){
                    #pragma HLS pipeline II=1
                    tapa::vec_t<float, 32> tmp;
                    float scale = tapa::bit_cast<float>(ap_uint<32>(pack_dequant[i/INTERM_DIM_DIV_32](31, 0)));
                    float zeropoint = tapa::bit_cast<float>(ap_uint<32>(pack_dequant[i/INTERM_DIM_DIV_32](63, 32)));
                    for (int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        tmp[k*2] = (float) (ap_uint<22>(linear_out[j][i * 16 + k](21, 0)).to_int()) * scale - zeropoint;
                        tmp[k*2 + 1] = (float) (ap_uint<22>(linear_out[j][i * 16 + k](43, 22)).to_int()) * scale - zeropoint;
                    }
                    if (i < INTERM_DIM_DIV_32) {
                        up_out_fifo.write(tmp);
                    } else {
                        gate_out_fifo.write(tmp);
                    }
                }
            }
        } else {
            float scale = tapa::bit_cast<float>(ap_uint<32>(pack_dequant[0](31, 0)));
            float zeropoint = tapa::bit_cast<float>(ap_uint<32>(pack_dequant[0](63, 32)));
            for (int j = 0; j < L; j++){
                for (int i = 0; i < (HIDDEN_DIM >> 4); i++) {
                    #pragma HLS pipeline II=1
                    tapa::vec_t<float, 16> tmp;
                    for (int k = 0; k < 8; k++) {
                        #pragma HLS unroll
                        tmp[k*2] = (float) (ap_uint<22>(linear_out[j][i * 8 + k](21, 0)).to_int()) * scale - zeropoint;
                        tmp[k*2 + 1] = (float) (ap_uint<22>(linear_out[j][i * 8 + k](43, 22)).to_int()) * scale - zeropoint;
                    }
                    res_fifo.write(tmp);
                }
            }
        }
    }
}

void element_wise_mul(
    tapa::istream<int>& L_in_fifo,
    tapa::istream<tapa::vec_t<float, 32>>& up_fifo,
    tapa::istream<tapa::vec_t<float, 32>>& gate_fifo,
    tapa::ostreams<tapa::vec_t<float, 16>, 2>& out_fifo
) {
    ap_uint<64> linear_out[MAX_SEQ_LEN][INTERM_DIM_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=16 dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_2P impl=URAM

    const int L = L_in_fifo.read();

    for (int i = 0; i < (INTERM_DIM >> 5); i++) {
        for (int j = 0; j < L; j++){
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 32> tmp = up_fifo.read();
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                linear_out[j][i * 16 + k] = ap_uint<64>((tapa::bit_cast<ap_uint<32>>(tmp[k*2+1]), tapa::bit_cast<ap_uint<32>>(tmp[k*2])));
            }
        }
    }

    for(int i = 0; i < (INTERM_DIM >> 5); i++) {
        for(int j = 0; j < L; j++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 32> tmp = gate_fifo.read();
            for (int k = 0; k < 16; k++) {
                #pragma HLS unroll
                float op1 = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 16 + k](31, 0)));
                float op2 = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 16 + k](63, 32)));
                op1 *= tmp[k*2];
                op2 *= tmp[k*2 + 1];
                linear_out[j][i*16+k] = ap_uint<64>((tapa::bit_cast<ap_uint<32>>(op2), tapa::bit_cast<ap_uint<32>>(op1)));
            }
        }
    }

    for (int i = 0; i < (INTERM_DIM >> 5); i++) {
        for (int j = 0; j < L; j++){
            #pragma HLS pipeline II=1
            for(int ch = 0; ch < 2; ch++){
                #pragma HLS unroll
                tapa::vec_t<float, 16> tmp;
                for (int k = 0; k < 8; k++) {
                    #pragma HLS unroll
                    tmp[k*2] = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 16 + ch * 8 + k](31, 0)));
                    tmp[k*2 + 1] = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 16 + ch * 8 + k](63, 32)));
                }
                out_fifo[ch].write(tmp);
            }
        }
    }
}

void apply_rope(
    tapa::istream<int>& L_in_fifo,
    tapa::ostream<int>& L_out_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& sin_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& cos_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    apply_rotary_pos_emb<NUM_ROPE_HEADS>(
        L_in_fifo, L_out_fifo, input_fifo, sin_fifo, cos_fifo, out_fifo
    );
}

void attn_cache(
    tapa::istream<int>& L_in_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& attn_in_fifo,
    tapa::ostreams<tapa::vec_t<float, 16>, 2>& attn_out_fifo
) {
    ap_uint<64> linear_out[MAX_SEQ_LEN][HIDDEN_DIM_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=16 dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_1P impl=URAM

    const int L = L_in_fifo.read();

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < (HEAD_DIM >> 4); j++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> tmp = attn_in_fifo.read();
                for (int k = 0; k < 8; k++) {
                    #pragma HLS unroll
                    linear_out[i][h*HEAD_DIM_DIV_2 + j * 8 + k] = ap_uint<64>((tapa::bit_cast<ap_uint<32>>(tmp[k*2+1]), tapa::bit_cast<ap_uint<32>>(tmp[k*2])));
                }
            }
        }
    }

    for (int i = 0; i < (HIDDEN_DIM >> 5); i++) {
        for (int j = 0; j < L; j++){
            #pragma HLS pipeline II=1
            for (int c = 0; c < 2; c++) {
                #pragma HLS unroll
                tapa::vec_t<float, 16> tmp;
                for (int k = 0; k < 8; k++) {
                    #pragma HLS unroll
                    tmp[k*2] = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 16 + c * 8 + k](31, 0)));
                    tmp[k*2 + 1] = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 16 + c * 8 + k](63, 32)));
                }
                attn_out_fifo[c].write(tmp);
            }
        }
    }
}

void scale_zero_reader_final(
    tapa::async_mmap<ap_uint<64>>& scale_zero_buffer,
    tapa::ostream<ap_uint<64>>& scale_zero_fifo
) {
    scale_zero_reader<TOTAL_HEADS+4>(scale_zero_buffer, scale_zero_fifo);
}

void qwen_block(
    const int L,
    tapa::mmap<tapa::vec_t<float, 16>> input_buffer,
    tapa::mmaps<tapa::vec_t<float, 16>, 2> centroid_buffer,
    tapa::mmaps<tapa::vec_t<ap_uint<8>, 64>, 16> lut_weight_idx_buffer,
    tapa::mmap<ap_uint<64>> scale_zero_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> sin_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> cos_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> rms_norm_weight_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> out_buffer,
    tapa::mmap<int> cycle_count
) {

    tapa::stream<tapa::vec_t<float, 16>> input_fifo("input_fifo");

    tapa::streams<tapa::vec_t<float, 2>, 16> input_split_fifo("input_split_fifo");
    tapa::streams<tapa::vec_t<float, 2>, 16> centroid_fifo("centroid_fifo");
    tapa::streams<ap_uint<8>, 16, 32> idx_fifo("idx_fifo");
    tapa::streams<tapa::vec_t<ap_uint<8>, 64>, 16> lut_weight_idx_fifo("lut_weight_idx_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_0_fifo("psum_0_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_1_fifo("psum_1_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_2_fifo("psum_2_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_3_fifo("psum_3_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_4_fifo("psum_4_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_5_fifo("psum_5_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_6_fifo("psum_6_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_7_fifo("psum_7_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_8_fifo("psum_8_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_9_fifo("psum_9_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_10_fifo("psum_10_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_11_fifo("psum_11_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_12_fifo("psum_12_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_13_fifo("psum_13_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_14_fifo("psum_14_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 16, 8> psum_15_fifo("psum_15_fifo");
    tapa::stream<ap_uint<64>> scale_zero_fifo("scale_zero_fifo");
    tapa::stream<tapa::vec_t<float, 16>> res_fifo("res_fifo");

    tapa::stream<tapa::vec_t<float, 16>> sin_fifo("sin_fifo");
    tapa::stream<tapa::vec_t<float, 16>> cos_fifo("cos_fifo");
    tapa::stream<tapa::vec_t<float, 16>> rope_in_fifo("rope_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>, 8> input_fifo_qk("input_fifo_qk");
    tapa::stream<tapa::vec_t<float, 16>> input_fifo_av("input_fifo_av");
    tapa::stream<tapa::vec_t<float, 16>> attn_cache_fifo("attn_cache_fifo");
    tapa::streams<tapa::vec_t<float, 16>, 2> attn_out_fifo("attn_out_fifo");
    tapa::stream<tapa::vec_t<float, 16>> pre_softmax_fifo("pre_softmax_fifo");
    tapa::stream<tapa::vec_t<float, 16>> post_softmax_fifo("post_softmax_fifo");

    tapa::streams<tapa::vec_t<float, 16>, 2> up_gate_fifo("up_gate_fifo");
    tapa::stream<tapa::vec_t<float, 32>> gate_before_silu_fifo("gate_before_silu_fifo");
    tapa::stream<tapa::vec_t<float, 32>> gate_after_silu_fifo("gate_after_silu_fifo");
    tapa::stream<tapa::vec_t<float, 32>> up_out_fifo("up_out_fifo");

    tapa::stream<tapa::vec_t<float, 16>> norm_in_fifo("norm_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>> norm_weight_fifo("norm_weight_fifo");
    tapa::streams<tapa::vec_t<float, 16>, 2> norm_to_splitter_fifo("norm_to_splitter_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");

    tapa::stream<bool> fifo_fin("fifo_fin");

    // pass seq length w/ fifo
    tapa::stream<int> L_inp_to_res_fifo("L_inp_to_res_fifo");
    tapa::stream<int> L_res_to_rms_fifo("L_res_to_rms_fifo");
    tapa::streams<int, 2> L_rms_to_splitter_fifo("L_rms_to_splitter_fifo");
    tapa::streams<int, 17> L_rms_to_ccu_fifo("L_rms_to_ccu_fifo");
    tapa::streams<int, 16> L_ccu_to_mm_fifo("L_ccu_to_mm_fifo");
    tapa::streams<int, 5> L_mm_to_rope_fifo("L_mm_to_rope_fifo");
    tapa::streams<int, 2> L_mm_to_silu_fifo("L_mm_to_silu_fifo");

    tapa::task()
        .invoke<tapa::join>(input_reader_wide, L, input_buffer, input_fifo, L_inp_to_res_fifo)
        .invoke<tapa::join>(rms_weight_reader, rms_norm_weight_buffer, norm_weight_fifo)
        .invoke<tapa::join>(residual_bank, L_inp_to_res_fifo, L_res_to_rms_fifo, input_fifo, res_fifo, norm_in_fifo)
        .invoke<tapa::join>(rms_norm_cache, L_res_to_rms_fifo, L_rms_to_splitter_fifo, L_rms_to_ccu_fifo, norm_in_fifo, norm_weight_fifo, norm_to_splitter_fifo, out_fifo)
        .invoke<tapa::join, 16>(lut_weight_idx_reader, lut_weight_idx_buffer, lut_weight_idx_fifo)
        .invoke<tapa::join>(scale_zero_reader_final, scale_zero_buffer, scale_zero_fifo)
        .invoke<tapa::join>(rope_input_reader, L, sin_buffer, sin_fifo)
        .invoke<tapa::join>(rope_input_reader, L, cos_buffer, cos_fifo)
        .invoke<tapa::join, 2>(input_splitter_final, L_rms_to_splitter_fifo, norm_to_splitter_fifo, attn_out_fifo, up_gate_fifo, input_split_fifo)
        .invoke<tapa::join, 2>(centroid_reader_split, centroid_buffer, centroid_fifo)
        .invoke<tapa::join, 16>(treeccu_fp32, L_rms_to_ccu_fifo, L_rms_to_ccu_fifo, L_ccu_to_mm_fifo, input_split_fifo, centroid_fifo, idx_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_head_half_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_0_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_0_fifo, psum_1_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_1_fifo, psum_2_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_2_fifo, psum_3_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_3_fifo, psum_4_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_4_fifo, psum_5_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_5_fifo, psum_6_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_6_fifo, psum_7_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_7_fifo, psum_8_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_8_fifo, psum_9_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_9_fifo, psum_10_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_10_fifo, psum_11_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_11_fifo, psum_12_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_12_fifo, psum_13_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_13_fifo, psum_14_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_final, L_ccu_to_mm_fifo, idx_fifo, lut_weight_idx_fifo, psum_14_fifo, psum_15_fifo)
        .invoke<tapa::join>(memory_matcher_acc_overlay_half, L_rms_to_ccu_fifo, L_mm_to_rope_fifo, L_mm_to_silu_fifo, psum_15_fifo, scale_zero_fifo, rope_in_fifo, input_fifo_av, up_out_fifo, gate_before_silu_fifo, res_fifo)
        .invoke<tapa::join>(apply_rope, L_mm_to_rope_fifo, L_mm_to_rope_fifo, rope_in_fifo, sin_fifo, cos_fifo, input_fifo_qk)
        .invoke<tapa::join>(gemm_gqa_qk, L_mm_to_rope_fifo, L_mm_to_rope_fifo, input_fifo_qk, pre_softmax_fifo)
        .invoke<tapa::join>(softmax, L_mm_to_rope_fifo, L_mm_to_rope_fifo, pre_softmax_fifo, post_softmax_fifo)
        .invoke<tapa::join>(gemm_gqa_av, L_mm_to_rope_fifo, L_mm_to_rope_fifo, input_fifo_av, post_softmax_fifo, attn_cache_fifo)
        .invoke<tapa::join>(attn_cache, L_mm_to_rope_fifo, attn_cache_fifo, attn_out_fifo)
        .invoke<tapa::join>(silu, L_mm_to_silu_fifo, L_mm_to_silu_fifo, gate_before_silu_fifo, gate_after_silu_fifo)
        .invoke<tapa::join>(element_wise_mul, L_mm_to_silu_fifo, up_out_fifo, gate_after_silu_fifo, up_gate_fifo)
        .invoke<tapa::join>(linear_out_writer, L, out_fifo, out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);

}

#endif // _QWEN_BLOCK_H_