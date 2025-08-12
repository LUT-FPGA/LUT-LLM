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

void apply_rope(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& sin_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& cos_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    apply_rotary_pos_emb<NUM_ROPE_HEADS>(
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

// void gqa_arbiter(
//     const int L,
//     tapa::istream<tapa::vec_t<float, 16>>& qk_in_fifo,
//     tapa::istream<tapa::vec_t<float, 16>>& v_in_fifo,
//     tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
// ) {
//     for(int r = 0; r < 18; r++) {
//         for(int i = 0; i < (L * HEAD_DIM >> 4); i++) {
//             #pragma HLS pipeline II=1
//             tapa::vec_t<float, 16> tmp;
//             if (r == 0 || r == 9) {
//                 tmp = v_in_fifo.read();
//             } else {
//                 tmp = qk_in_fifo.read();
//             }
//             out_fifo.write(tmp);
//         }
//     }
// }

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

void memory_matcher_acc_overlay_half(
    const int L,
    tapa::istreams<tapa::vec_t<ap_uint<48>, 8>, 16>& inbound_fifo, // interleave up and gate
    tapa::istream<ap_uint<64>>& scale_zero_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& rope_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& v_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    ap_uint<64> linear_out[MAX_SEQ_LEN][MAX_OUT_SIZE_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=256 dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_2P impl=URAM

    for (int round = 0; round < 2; round++) {

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
        
        if (round == 0) {
            for(int i = 0; i < TOTAL_HEADS; i++) {
                #pragma HLS pipeline II=1
                pack_dequant[i] = scale_zero_fifo.read();
            }
        } else {
            pack_dequant[0] = scale_zero_fifo.read(); 
        }

        int j_bound = (round == 0) ? (QKV_DIM >> 9) : (HIDDEN_DIM >> 9);

        // read indices and parallel match
        for (int r = 0; r < (HIDDEN_DIM_DIV_2 >> 4); r++) {

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
        } else { // write out projection
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
                    out_fifo.write(tmp);
                }
            }
        }
    }
}

void memory_matcher_w_vq_half_attn(
    const int L,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<8>, 64>>& lut_weight_idx_fifo,
    tapa::istreams<tapa::vec_t<ap_uint<48>, 8>, 16>& inbound_fifo,
    tapa::ostreams<tapa::vec_t<ap_uint<48>, 8>, 16>& outbound_fifo
) {
    memory_matcher_w_vq_half<HIDDEN_DIM_DIV_2, HIDDEN_DIM_DIV_2, QKV_DIM, HIDDEN_DIM>(
        L, idx_fifo, lut_weight_idx_fifo, inbound_fifo, outbound_fifo
    );
}


void memory_matcher_w_vq_half_dsp_attn(
    const int L,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<8>, 64>>& lut_weight_idx_fifo,
    tapa::istreams<tapa::vec_t<ap_uint<48>, 8>, 16>& inbound_fifo,
    tapa::ostreams<tapa::vec_t<ap_uint<48>, 8>, 16>& outbound_fifo
) {
    memory_matcher_w_vq_half_dsp<HIDDEN_DIM_DIV_2, HIDDEN_DIM_DIV_2, QKV_DIM, HIDDEN_DIM>(
        L, idx_fifo, lut_weight_idx_fifo, inbound_fifo, outbound_fifo
    );
}

void memory_matcher_w_vq_head_half_attn(
    const int L,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<8>, 64>>& lut_weight_idx_fifo,
    tapa::ostreams<tapa::vec_t<ap_uint<48>, 8>, 16>& outbound_fifo
) {
    memory_matcher_w_vq_head_half<HIDDEN_DIM_DIV_2, HIDDEN_DIM_DIV_2, QKV_DIM, HIDDEN_DIM>(
        L, idx_fifo, lut_weight_idx_fifo, outbound_fifo
    );
}

void scale_zero_reader_attn(
    tapa::async_mmap<ap_uint<64>>& scale_zero_buffer,
    tapa::ostream<ap_uint<64>>& scale_zero_fifo
) {
    scale_zero_reader<TOTAL_HEADS+1>(scale_zero_buffer, scale_zero_fifo);
}

void attn_cache(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& attn_in_fifo,
    tapa::ostreams<tapa::vec_t<float, 16>, 2>& attn_out_fifo
) {
    ap_uint<64> linear_out[MAX_SEQ_LEN][HIDDEN_DIM_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=16 dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_1P impl=URAM

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

void attention_block(
    const int L,
    tapa::mmaps<tapa::vec_t<float, 16>, 2> input_buffer,
    tapa::mmaps<tapa::vec_t<float, 16>, 2> centroid_buffer,
    tapa::mmaps<tapa::vec_t<ap_uint<8>, 64>, 16> lut_weight_idx_buffer,
    tapa::mmap<ap_uint<64>> scale_zero_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> sin_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> cos_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> attn_out_buffer,
    tapa::mmap<int> cycle_count
) {

    tapa::streams<tapa::vec_t<float, 16>, 2> input_fifo("input_fifo");

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
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");

    tapa::stream<tapa::vec_t<float, 16>> sin_fifo("sin_fifo");
    tapa::stream<tapa::vec_t<float, 16>> cos_fifo("cos_fifo");
    tapa::stream<tapa::vec_t<float, 16>> rope_in_fifo("rope_in_fifo");
    tapa::stream<tapa::vec_t<float, 16>, 8> input_fifo_qk("input_fifo_qk");
    tapa::stream<tapa::vec_t<float, 16>> input_fifo_av("input_fifo_av");
    tapa::stream<tapa::vec_t<float, 16>> attn_cache_fifo("attn_cache_fifo");
    tapa::streams<tapa::vec_t<float, 16>, 2> attn_out_fifo("attn_out_fifo");
    tapa::stream<tapa::vec_t<float, 16>> pre_softmax_fifo("pre_softmax_fifo");
    tapa::stream<tapa::vec_t<float, 16>> post_softmax_fifo("post_softmax_fifo");

    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join, 2>(input_reader_wide, L, HIDDEN_DIM_DIV_2, input_buffer, input_fifo)
        .invoke<tapa::join, 16>(lut_weight_idx_reader, ATTN_LUT_WEIGHT_SIZE, lut_weight_idx_buffer, lut_weight_idx_fifo)
        .invoke<tapa::join>(scale_zero_reader_attn, scale_zero_buffer, scale_zero_fifo)
        .invoke<tapa::join>(rope_input_reader, L, sin_buffer, sin_fifo)
        .invoke<tapa::join>(rope_input_reader, L, cos_buffer, cos_fifo)
        .invoke<tapa::join, 2>(input_splitter_attn, L, input_fifo, attn_out_fifo, input_split_fifo)
        .invoke<tapa::join, 2>(centroid_reader_split, ATTN_CENTROID_SIZE, centroid_buffer, centroid_fifo)
        .invoke<tapa::join, 16>(treeccu_fp32, L, ATTN_CENTROID_SIZE, input_split_fifo, centroid_fifo, idx_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_head_half_attn, L, idx_fifo, lut_weight_idx_fifo, psum_0_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_attn, L, idx_fifo, lut_weight_idx_fifo, psum_0_fifo, psum_1_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_attn, L, idx_fifo, lut_weight_idx_fifo, psum_1_fifo, psum_2_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_attn, L, idx_fifo, lut_weight_idx_fifo, psum_2_fifo, psum_3_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_attn, L, idx_fifo, lut_weight_idx_fifo, psum_3_fifo, psum_4_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_attn, L, idx_fifo, lut_weight_idx_fifo, psum_4_fifo, psum_5_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_attn, L, idx_fifo, lut_weight_idx_fifo, psum_5_fifo, psum_6_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_attn, L, idx_fifo, lut_weight_idx_fifo, psum_6_fifo, psum_7_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_attn, L, idx_fifo, lut_weight_idx_fifo, psum_7_fifo, psum_8_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_attn, L, idx_fifo, lut_weight_idx_fifo, psum_8_fifo, psum_9_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_attn, L, idx_fifo, lut_weight_idx_fifo, psum_9_fifo, psum_10_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_attn, L, idx_fifo, lut_weight_idx_fifo, psum_10_fifo, psum_11_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_attn, L, idx_fifo, lut_weight_idx_fifo, psum_11_fifo, psum_12_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_attn, L, idx_fifo, lut_weight_idx_fifo, psum_12_fifo, psum_13_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_attn, L, idx_fifo, lut_weight_idx_fifo, psum_13_fifo, psum_14_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_half_dsp_attn, L, idx_fifo, lut_weight_idx_fifo, psum_14_fifo, psum_15_fifo)
        .invoke<tapa::join>(memory_matcher_acc_overlay_half, L, psum_15_fifo, scale_zero_fifo, rope_in_fifo, input_fifo_av, out_fifo)
        .invoke<tapa::join>(apply_rope, L, rope_in_fifo, sin_fifo, cos_fifo, input_fifo_qk)
        .invoke<tapa::join>(gemm_gqa_qk, L, input_fifo_qk, pre_softmax_fifo)
        .invoke<tapa::join>(softmax, L, pre_softmax_fifo, post_softmax_fifo)
        .invoke<tapa::join>(gemm_gqa_av, L, input_fifo_av, post_softmax_fifo, attn_cache_fifo)
        .invoke<tapa::join>(attn_cache, L, attn_cache_fifo, attn_out_fifo)
        .invoke<tapa::join>(linear_out_writer, L, HIDDEN_DIM, out_fifo, attn_out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);

}


#endif // _ATTN_H_