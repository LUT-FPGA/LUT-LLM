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

void element_wise_mul(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& up_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& gate_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    ap_uint<64> linear_out[MAX_SEQ_LEN][INTERM_DIM_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=8 dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_2P impl=URAM

    for (int i = 0; i < (INTERM_DIM >> 4); i++) {
        for (int j = 0; j < L; j++){
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 16> tmp = up_fifo.read();
            for (int k = 0; k < 8; k++) {
                #pragma HLS unroll
                linear_out[j][i * 8 + k] = ap_uint<64>((tapa::bit_cast<ap_uint<32>>(tmp[k*2+1]), tapa::bit_cast<ap_uint<32>>(tmp[k*2])));
            }
        }
    }

    for(int i = 0; i < (INTERM_DIM >> 4); i++) {
        for(int j = 0; j < L; j++) {
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 16> tmp = gate_fifo.read();
            for (int k = 0; k < 8; k++) {
                #pragma HLS unroll
                float op1 = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 8 + k](31, 0)));
                float op2 = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 8 + k](63, 32)));
                op1 *= tmp[k*2];
                op2 *= tmp[k*2 + 1];
                linear_out[j][i*8+k] = ap_uint<64>((tapa::bit_cast<ap_uint<32>>(op2), tapa::bit_cast<ap_uint<32>>(op1)));
            }
        }
    }

    for (int i = 0; i < (INTERM_DIM >> 4); i++) {
        for (int j = 0; j < L; j++){
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 16> tmp;
            for (int k = 0; k < 8; k++) {
                #pragma HLS unroll
                tmp[k*2] = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 8 + k](31, 0)));
                tmp[k*2 + 1] = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * 8 + k](63, 32)));
            }
            
            out_fifo.write(tmp);
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
    memory_matcher<2, 128>(L, in_size, out_size, idx_fifo, lut_fifo, out_fifo);
}

void memory_matcher_down(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istream<ap_uint<8>>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    memory_matcher<16, 64>(L, in_size, out_size, idx_fifo, lut_fifo, out_fifo);
}

void memory_matcher_acc_overlay(
    const int L,
    tapa::istreams<tapa::vec_t<ap_uint<44>, 8>, 32>& inbound_fifo, // interleave up and gate
    tapa::istream<ap_uint<64>>& scale_zero_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& up_out_fifo, // stream to splitter
    tapa::ostream<tapa::vec_t<float, 16>>& gate_out_fifo, // stream to silu
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    ap_uint<64> linear_out[MAX_SEQ_LEN][MAX_OUT_SIZE_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=512 dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_2P impl=URAM

    for (int round = 0; round < 2; round++) {

        for (int i = 0; i < L; i++) {
            for (int j = 0; j < (INTERM_DIM >> 9); j++){
                #pragma HLS pipeline II=1
                for (int k = 0; k < 512; k++) {
                    #pragma HLS unroll
                    linear_out[i][(j << 9) + k] = ap_uint<64>(0); // Initialize output
                }
            }
        }
        
        ap_uint<64> pack_dequant_up;
        ap_uint<64> pack_dequant_gate;
        
        pack_dequant_up = scale_zero_fifo.read();
        float scale_up = tapa::bit_cast<float>(ap_uint<32>(pack_dequant_up(31, 0)));
        float zeropoint_up = tapa::bit_cast<float>(ap_uint<32>(pack_dequant_up(63, 32)));
        float scale_gate = 0.0f;
        float zeropoint_gate = 0.0f;

        if (round == 0) {
            pack_dequant_gate = scale_zero_fifo.read();
            scale_gate = tapa::bit_cast<float>(ap_uint<32>(pack_dequant_gate(31, 0)));
            zeropoint_gate = tapa::bit_cast<float>(ap_uint<32>(pack_dequant_gate(63, 32)));
        }

        int r_bound = (round == 0) ? (HIDDEN_DIM_DIV_2 >> 3) : (INTERM_DIM_DIV_2 >> 3);
        int j_bound = (round == 0) ? (INTERM_DIM >> 9) : (HIDDEN_DIM >> 10);

        // read indices and parallel match
        for (int r = 0; r < r_bound; r++) {

            for (int i = 0; i < L; i++) {

                for (int j = 0; j < j_bound; j++) {
                    #pragma HLS pipeline II=1
                    
                    for (int k = 0; k < 32; k++) {
                        #pragma HLS unroll
                        auto tmp_vec = inbound_fifo[k].read();
                        for (int m = 0; m < 8; m++) {
                            #pragma HLS unroll
                            ap_uint<44> op2 = tmp_vec[m];
                            for (int p = 0; p < 2; p++) {
                                #pragma HLS unroll
                                ap_uint<64> acc_reg = linear_out[i][j*512 + k*16 + m*2 + p];
                                ap_uint<44> simd_a = ap_uint<44>(acc_reg(43, 0));
                                ap_uint<44> simd_b = ap_uint<44>((ap_uint<22>(op2(p*22+21, p*22+11)),ap_uint<22>(op2(p*22+10, p*22))));
                                ap_uint<44> simd_out = simd_a + simd_b;
                                linear_out[i][j*512 + k*16 + m * 2 + p] = simd_out;
                            }
                        }
                    }
                }
            }
        }

        if (round == 0) {
            for (int i = 0; i < (INTERM_DIM >> 3); i++) {
                for (int j = 0; j < L; j++){
                    #pragma HLS pipeline II=1
                    tapa::vec_t<float, 16> tmp;
                    float scale = 0.0f;
                    float zeropoint = 0.0f;
                    if(i < (INTERM_DIM >> 4)) {
                        scale = scale_up;
                        zeropoint = zeropoint_up;
                    } else {
                        scale = scale_gate;
                        zeropoint = zeropoint_gate;
                    }
                    for (int k = 0; k < 8; k++) {
                        #pragma HLS unroll
                        tmp[k*2] = (float) (ap_uint<22>(linear_out[j][i * 8 + k](21, 0)).to_int()) * scale - zeropoint;
                        tmp[k*2 + 1] = (float) (ap_uint<22>(linear_out[j][i * 8 + k](43, 22)).to_int()) * scale - zeropoint;
                    }
                    if (i < (INTERM_DIM >> 4)) {
                        up_out_fifo.write(tmp);
                    } else {
                        gate_out_fifo.write(tmp);
                    }
                }
            }
        } else {
            for (int i = 0; i < (HIDDEN_DIM >> 4); i++) {
                for (int j = 0; j < L; j++){
                    #pragma HLS pipeline II=1
                    tapa::vec_t<float, 16> tmp;
                    for (int k = 0; k < 8; k++) {
                        #pragma HLS unroll
                        tmp[k*2] = (float) (ap_uint<22>(linear_out[j][i * 8 + k](21, 0)).to_int()) * scale_up - zeropoint_up;
                        tmp[k*2 + 1] = (float) (ap_uint<22>(linear_out[j][i * 8 + k](43, 22)).to_int()) * scale_up - zeropoint_up;
                    }
                    out_fifo.write(tmp);
                }
            }
        }
    }
}

void ffn_core(
    const int L,
    tapa::mmap<tapa::vec_t<float, 16>> input_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> centroid_buffer,
    tapa::mmaps<tapa::vec_t<ap_uint<8>, 64>, 8> lut_weight_idx_buffer,
    tapa::mmap<ap_uint<64>> scale_zero_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> ffn_out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 16>> input_fifo("input_fifo");

    tapa::streams<tapa::vec_t<float, 2>, 8> input_split_fifo("input_split_fifo");
    tapa::streams<tapa::vec_t<float, 2>, 8> centroid_fifo("centroid_fifo");
    tapa::streams<ap_uint<8>, 8, 16> idx_fifo("idx_fifo");
    tapa::streams<tapa::vec_t<ap_uint<8>, 64>, 8> lut_weight_idx_fifo("lut_weight_idx_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 32> psum_0_fifo("psum_0_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 32> psum_1_fifo("psum_1_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 32> psum_2_fifo("psum_2_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 32> psum_3_fifo("psum_3_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 32> psum_4_fifo("psum_4_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 32> psum_5_fifo("psum_5_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 32> psum_6_fifo("psum_6_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 32> psum_7_fifo("psum_7_fifo");
    tapa::stream<ap_uint<64>> scale_zero_fifo("scale_zero_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");

    tapa::stream<tapa::vec_t<float, 16>> up_gate_fifo("up_gate_fifo");
    tapa::stream<tapa::vec_t<float, 16>> gate_before_silu_fifo("gate_before_silu_fifo");
    tapa::stream<tapa::vec_t<float, 16>> gate_after_silu_fifo("gate_after_silu_fifo");
    tapa::stream<tapa::vec_t<float, 16>> up_out_fifo("up_out_fifo");

    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(input_reader_wide, L, HIDDEN_DIM_DIV_2, input_buffer, input_fifo)
        .invoke<tapa::join, 8>(lut_weight_idx_reader, FFN_LUT_WEIGHT_SIZE, lut_weight_idx_buffer, lut_weight_idx_fifo)
        .invoke<tapa::join>(scale_zero_reader, 3, scale_zero_buffer, scale_zero_fifo)
        .invoke<tapa::join>(input_splitter_ffn, L, input_fifo, up_gate_fifo, input_split_fifo)
        .invoke<tapa::join>(centroid_reader_split, CENTROID_SIZE, centroid_buffer, centroid_fifo)
        .invoke<tapa::join, 8>(ccu_fp32, L, CENTROID_SIZE, input_split_fifo, centroid_fifo, idx_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_head, L, idx_fifo, lut_weight_idx_fifo, psum_0_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_0_fifo, psum_1_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_1_fifo, psum_2_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_2_fifo, psum_3_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_3_fifo, psum_4_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_4_fifo, psum_5_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_5_fifo, psum_6_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_6_fifo, psum_7_fifo)
        .invoke<tapa::join>(memory_matcher_acc_overlay, L, psum_7_fifo, scale_zero_fifo, up_out_fifo, gate_before_silu_fifo, out_fifo)
        .invoke<tapa::join>(silu, L, gate_before_silu_fifo, gate_after_silu_fifo)
        .invoke<tapa::join>(element_wise_mul, L, up_out_fifo, gate_after_silu_fifo, up_gate_fifo)
        .invoke<tapa::join>(linear_out_writer, L, HIDDEN_DIM, out_fifo, ffn_out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}


#endif // _FFN_H_