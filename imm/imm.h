#ifndef _IMM_H_
#define _IMM_H_

#include <tapa.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <limits>

#include "../config/config.h"

typedef ap_uint<8> idx_t;

void index_reader(
    const int L,
    const int in_size,
    tapa::async_mmap<int>& idx_buffer,
    tapa::ostream<idx_t>& idx_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < (L * in_size >> 3);){
        #pragma HLS pipeline II=1
		if((i_req < (L * in_size >> 3)) & !idx_buffer.read_addr.full()){
            idx_buffer.read_addr.try_write(i_req);
            ++i_req;
		}
		if(!idx_buffer.read_data.empty()){
            int tmp;
            idx_buffer.read_data.try_read(tmp);
            idx_fifo.write(idx_t(tmp));
            ++i_resp;
		}
	}
}

void scale_zero_reader(
    const int scale_zero_size,
    tapa::async_mmap<ap_uint<64>>& scale_zero_buffer,
    tapa::ostream<ap_uint<64>>& scale_zero_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < scale_zero_size;){
        #pragma HLS pipeline II=1
        if((i_req < scale_zero_size) & !scale_zero_buffer.read_addr.full()){
            scale_zero_buffer.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!scale_zero_buffer.read_data.empty()){
            ap_uint<64> tmp;
            scale_zero_buffer.read_data.try_read(tmp);
            scale_zero_fifo.write(tmp);
            ++i_resp;
        }
    }
}


void lut_reader(
    const int lut_size,
    tapa::async_mmap<tapa::vec_t<ap_uint<8>, 64>>& lut_buffer,
    tapa::ostream<tapa::vec_t<ap_uint<8>, 64>>& lut_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((lut_size) >> 8);){
        #pragma HLS pipeline II=1
        if((i_req < ((lut_size) >> 8)) & !lut_buffer.read_addr.full()){
            lut_buffer.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!lut_buffer.read_data.empty()){
            tapa::vec_t<ap_uint<8>, 64> tmp;
            lut_buffer.read_data.try_read(tmp);
            lut_fifo.write(tmp);
            ++i_resp;
        }
    }
}

void weight_idx_reader(
    const int weight_idx_size,
    tapa::async_mmap<tapa::vec_t<ap_uint<8>, 64>>& lut_buffer,
    tapa::ostream<tapa::vec_t<ap_uint<8>, 64>>& lut_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((weight_idx_size) >> 10);){
        #pragma HLS pipeline II=1
        if((i_req < ((weight_idx_size) >> 10)) & !lut_buffer.read_addr.full()){
            lut_buffer.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!lut_buffer.read_data.empty()){
            tapa::vec_t<ap_uint<8>, 64> tmp;
            lut_buffer.read_data.try_read(tmp);
            lut_fifo.write(tmp);
            ++i_resp;
        }
    }
}

void lut_weight_idx_reader(
    const int total_size,
    tapa::async_mmap<tapa::vec_t<ap_uint<8>, 64>>& lut_weight_idx_buffer,
    tapa::ostream<tapa::vec_t<ap_uint<8>, 64>>& lut_weight_idx_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < total_size;){
        #pragma HLS pipeline II=1
        if((i_req < total_size) & !lut_weight_idx_buffer.read_addr.full()){
            lut_weight_idx_buffer.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!lut_weight_idx_buffer.read_data.empty()){
            tapa::vec_t<ap_uint<8>, 64> tmp;
            lut_weight_idx_buffer.read_data.try_read(tmp);
            lut_weight_idx_fifo.write(tmp);
            ++i_resp;
        }
    }
}

void linear_out_writer(
    const int L,
    const int out_size,
    tapa::istream<tapa::vec_t<float, 16>>& out_fifo,
    tapa::async_mmap<tapa::vec_t<float, 16>>& linear_out_buffer,
    tapa::ostream<bool>& fifo_fin
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((L * out_size) >> 4);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < ((L * out_size) >> 4)) & !out_fifo.empty() & !linear_out_buffer.write_addr.full() & !linear_out_buffer.write_data.full()){
            linear_out_buffer.write_addr.try_write(i_req);
            tapa::vec_t<float, 16> tmp; out_fifo.try_read(tmp);
            linear_out_buffer.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = linear_out_buffer.write_resp.read(success);
        if(success){
            i_resp += unsigned(resp)+1;
        }
    }
    fifo_fin.write(true);
}

template<int para_write = 16, int para_access_factor = 128>
void memory_matcher(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, para_write>>& out_fifo
) {
    // prefetch LUT for linear layer
    ap_uint<64> linear_lut[n_cent][MAX_OUT_SIZE_DIV_2];
    #pragma HLS array_partition variable=linear_lut cyclic factor=para_access_factor dim=2
    #pragma HLS bind_storage variable=linear_lut type=RAM_1P impl=URAM

    ap_uint<64> linear_out[MAX_SEQ_LEN][MAX_OUT_SIZE_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=para_access_factor dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_2P impl=URAM

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size/2; j+=para_access_factor){
            #pragma HLS pipeline II=1
            for (int k = 0; k < para_access_factor; k++) {
                #pragma HLS unroll
                linear_out[i][j + k] = ap_uint<64>(0); // Initialize output
            }
        }
    }

    // read indices and parallel match
    for (int r = 0; r < in_size; r++) {

        for(int i = 0; i < n_cent; i++) {
            for (int j = 0; j < (out_size >> 4);){
                #pragma HLS pipeline II=1
                if(!lut_fifo.empty()){
                    tapa::vec_t<ap_uint<64>, 8> tmp; lut_fifo.try_read(tmp);
                    for(int k = 0; k < 8; k++) {
                        #pragma HLS unroll
                        linear_lut[i][j * 8 + k] = tmp[k];
                    }
                    j++;
                }
            }
        }

        for (int i = 0; i < L; i++) {

            auto idx = idx_fifo.read();
            
            for (int j = 0; j < out_size/2; j+=para_access_factor){
                #pragma HLS pipeline II=1
                for (int k = 0; k < para_access_factor; k++) {
                    #pragma HLS unroll
                    auto pack_psum = linear_out[i][j + k];
                    auto pack_lut = linear_lut[idx][j + k];
                    float op1 = tapa::bit_cast<float>(ap_uint<32>(pack_psum(31, 0)));
                    float op2 = tapa::bit_cast<float>(ap_uint<32>(pack_psum(63, 32)));
                    float op1_l = tapa::bit_cast<float>(ap_uint<32>(pack_lut(31, 0)));
                    float op2_l = tapa::bit_cast<float>(ap_uint<32>(pack_lut(63, 32)));

                    op1 += op1_l;
                    op2 += op2_l;
                    //repack
                    linear_out[i][j + k] = ap_uint<64>((tapa::bit_cast<ap_uint<32>>(op2), tapa::bit_cast<ap_uint<32>>(op1)));
                }
            }
        }
    }

    // write out results
    for (int i = 0; i < out_size / para_write; i++) {
        for (int j = 0; j < L; j++){
            #pragma HLS pipeline II=1
            tapa::vec_t<float, para_write> tmp;
            for (int k = 0; k < (para_write >> 1); k++) {
                #pragma HLS unroll
                tmp[k*2] = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * (para_write >> 1) + k](31, 0)));
                tmp[k*2 + 1] = tapa::bit_cast<float>(ap_uint<32>(linear_out[j][i * (para_write >> 1) + k](63, 32)));
            }
            out_fifo.write(tmp);
        }
    }
}

void memory_matcher_w_vq(
    const int L,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<8>, 64>>& lut_weight_idx_fifo,
    tapa::istreams<tapa::vec_t<ap_uint<48>, 8>, 32>& inbound_fifo,
    tapa::ostreams<tapa::vec_t<ap_uint<48>, 8>, 32>& outbound_fifo
) {
    for (int round = 0; round < 2; round++){
    // read indices and parallel match
        const int in_size = (round == 0) ? HIDDEN_DIM_DIV_2 : INTERM_DIM_DIV_2;
        const int out_size = (round == 0) ? INTERM_DIM_MUL_2 : HIDDEN_DIM;
        for (int r = 0; r < (in_size >> 4); r++) {
            // prefetch LUT for linear layer
            ap_uint<8> linear_lut[n_cent][w_n_cent][MAX_OUT_SIZE_DIV_512];
            #pragma HLS array_partition variable=linear_lut complete dim=2
            #pragma HLS array_partition variable=linear_lut cyclic factor=4 dim=1
            #pragma HLS array_partition variable=linear_lut cyclic factor=2 dim=3
            ap_uint<4> weight_idx[MAX_OUT_SIZE];
            #pragma HLS array_partition variable=weight_idx cyclic factor=1024
            #pragma HLS bind_storage variable=weight_idx type=RAM_1P impl=LUTRAM

            for(int i = 0; i < (n_cent >> 2); i++) {
                for (int j = 0; j < (out_size >> 9);){
                    #pragma HLS pipeline II=1
                    if(!lut_weight_idx_fifo.empty()){
                        tapa::vec_t<ap_uint<8>, 64> tmp; lut_weight_idx_fifo.try_read(tmp);
                        for (int ii = 0; ii < 4; ii++) {
                            #pragma HLS unroll
                            for(int k = 0; k < 16; k++) {
                                #pragma HLS unroll
                                linear_lut[i*4+ii][k][j] = tmp[ii*16+k];
                            }
                        }
                        j++;
                    }
                }
            }

            for (int i = 0; i < (out_size >> 7);) {
                #pragma HLS pipeline II=1
                if(!lut_weight_idx_fifo.empty()){
                    tapa::vec_t<ap_uint<8>, 64> tmp; lut_weight_idx_fifo.try_read(tmp);
                    for(int k = 0; k < 64; k++) {
                        #pragma HLS unroll
                        weight_idx[i * 128 + k * 2] = ap_uint<4>(tmp[k](3, 0));
                        weight_idx[i * 128 + k * 2 + 1] = ap_uint<4>(tmp[k](7, 4));
                    }
                    i++;
                }
            }

            for (int i = 0; i < L; i++) {

                auto idx = idx_fifo.read();

                for (int j = 0; j < (out_size >> 10); j++) {
                    #pragma HLS pipeline II=1

                    ap_uint<8> linear_out_reg[1024];
                    #pragma HLS array_partition variable=linear_out_reg complete
                    ap_uint<8> lut_reg[2][16];
                    #pragma HLS array_partition variable=lut_reg complete dim=1
                    #pragma HLS array_partition variable=lut_reg complete dim=2

                    for(int lut_i = 0; lut_i < 2; lut_i++) {
                        #pragma HLS unroll
                        for (int k = 0; k < 16; k++) {
                            #pragma HLS unroll
                            lut_reg[lut_i][k] = linear_lut[idx][k][j*2+lut_i];
                        }
                    }
                    for (int k = 0; k < 1024; k++) {
                        #pragma HLS unroll
                        int w_idx = weight_idx[j * 1024 + k].to_int();
                        linear_out_reg[k] = lut_reg[k>>9][w_idx];
                    }
                    for (int k = 0; k < 32; k++) {
                        #pragma HLS unroll
                        auto tmp_vec = inbound_fifo[k].read();
                        tapa::vec_t<ap_uint<48>, 8> out_vec;
                        for (int m = 0; m < 8; m++) {
                            #pragma HLS unroll
                            ap_uint<48> simd_out;
                            ap_uint<48> simd_a = tmp_vec[m]; 
                            ap_uint<48> simd_b;
                            for(int p = 0; p < 4; p++) {
                                #pragma HLS unroll
                                simd_b(p * 12 + 11, p * 12) = ap_uint<12>(linear_out_reg[k * 32 + m * 4 + p]);
                            }
                            simd_out = simd_a + simd_b;
                            out_vec[m] = simd_out;
                        }
                        outbound_fifo[k].write(out_vec);
                    }
                }
            }
        }
    }
}

void memory_matcher_w_vq_dsp(
    const int L,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<8>, 64>>& lut_weight_idx_fifo,
    tapa::istreams<tapa::vec_t<ap_uint<48>, 8>, 32>& inbound_fifo,
    tapa::ostreams<tapa::vec_t<ap_uint<48>, 8>, 32>& outbound_fifo
) {
    for (int round = 0; round < 2; round++){
    // read indices and parallel match
        const int in_size = (round == 0) ? HIDDEN_DIM_DIV_2 : INTERM_DIM_DIV_2;
        const int out_size = (round == 0) ? INTERM_DIM_MUL_2 : HIDDEN_DIM;
        for (int r = 0; r < (in_size >> 4); r++) {
            // prefetch LUT for linear layer
            ap_uint<8> linear_lut[n_cent][w_n_cent][MAX_OUT_SIZE_DIV_512];
            #pragma HLS array_partition variable=linear_lut complete dim=2
            #pragma HLS array_partition variable=linear_lut cyclic factor=4 dim=1
            #pragma HLS array_partition variable=linear_lut cyclic factor=2 dim=3
            ap_uint<4> weight_idx[MAX_OUT_SIZE];
            #pragma HLS array_partition variable=weight_idx cyclic factor=1024
            #pragma HLS bind_storage variable=weight_idx type=RAM_1P impl=LUTRAM

            for(int i = 0; i < (n_cent >> 2); i++) {
                for (int j = 0; j < (out_size >> 9);){
                    #pragma HLS pipeline II=1
                    if(!lut_weight_idx_fifo.empty()){
                        tapa::vec_t<ap_uint<8>, 64> tmp; lut_weight_idx_fifo.try_read(tmp);
                        for (int ii = 0; ii < 4; ii++) {
                            #pragma HLS unroll
                            for(int k = 0; k < 16; k++) {
                                #pragma HLS unroll
                                linear_lut[i*4+ii][k][j] = tmp[ii*16+k];
                            }
                        }
                        j++;
                    }
                }
            }

            for (int i = 0; i < (out_size >> 7);) {
                #pragma HLS pipeline II=1
                if(!lut_weight_idx_fifo.empty()){
                    tapa::vec_t<ap_uint<8>, 64> tmp; lut_weight_idx_fifo.try_read(tmp);
                    for(int k = 0; k < 64; k++) {
                        #pragma HLS unroll
                        weight_idx[i * 128 + k * 2] = ap_uint<4>(tmp[k](3, 0));
                        weight_idx[i * 128 + k * 2 + 1] = ap_uint<4>(tmp[k](7, 4));
                    }
                    i++;
                }
            }

            for (int i = 0; i < L; i++) {

                auto idx = idx_fifo.read();

                for (int j = 0; j < (out_size >> 10); j++) {
                    #pragma HLS pipeline II=1

                    ap_uint<8> linear_out_reg[1024];
                    #pragma HLS array_partition variable=linear_out_reg complete
                    ap_uint<8> lut_reg[2][16];
                    #pragma HLS array_partition variable=lut_reg complete dim=1
                    #pragma HLS array_partition variable=lut_reg complete dim=2

                    for(int lut_i = 0; lut_i < 2; lut_i++) {
                        #pragma HLS unroll
                        for (int k = 0; k < 16; k++) {
                            #pragma HLS unroll
                            lut_reg[lut_i][k] = linear_lut[idx][k][j*2+lut_i];
                        }
                    }
                    for (int k = 0; k < 1024; k++) {
                        #pragma HLS unroll
                        int w_idx = weight_idx[j * 1024 + k].to_int();
                        linear_out_reg[k] = lut_reg[k>>9][w_idx];
                    }
                    for (int k = 0; k < 32; k++) {
                        #pragma HLS unroll
                        auto tmp_vec = inbound_fifo[k].read();
                        tapa::vec_t<ap_uint<48>, 8> out_vec;
                        for (int m = 0; m < 8; m++) {
                            #pragma HLS unroll
                            ap_uint<48> simd_out;
                            ap_uint<48> simd_a = tmp_vec[m]; 
                            ap_uint<48> simd_b;
                            #pragma HLS bind_op variable=simd_out op=add impl=dsp
                            for(int p = 0; p < 4; p++) {
                                #pragma HLS unroll
                                simd_b(p * 12 + 11, p * 12) = ap_uint<12>(linear_out_reg[k * 32 + m * 4 + p]);
                            }
                            simd_out = simd_a + simd_b;
                            out_vec[m] = simd_out;
                        }
                        outbound_fifo[k].write(out_vec);
                    }
                }
            }
        }
    }
}

void memory_matcher_w_vq_head(
    const int L,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<8>, 64>>& lut_weight_idx_fifo,
    tapa::ostreams<tapa::vec_t<ap_uint<48>, 8>, 32>& outbound_fifo
) {
    for (int round = 0; round < 2; round++) {
        // read indices and parallel match
        const int in_size = (round == 0) ? HIDDEN_DIM_DIV_2 : INTERM_DIM_DIV_2;
        const int out_size = (round == 0) ? INTERM_DIM_MUL_2 : HIDDEN_DIM;
        for (int r = 0; r < (in_size >> 4); r++) {
            // prefetch LUT for linear layer
            ap_uint<8> linear_lut[n_cent][w_n_cent][MAX_OUT_SIZE_DIV_512];
            #pragma HLS array_partition variable=linear_lut complete dim=2
            #pragma HLS array_partition variable=linear_lut cyclic factor=4 dim=1
            #pragma HLS array_partition variable=linear_lut cyclic factor=2 dim=3
            ap_uint<4> weight_idx[MAX_OUT_SIZE];
            #pragma HLS array_partition variable=weight_idx cyclic factor=1024
            #pragma HLS bind_storage variable=weight_idx type=RAM_1P impl=LUTRAM

            for(int i = 0; i < (n_cent >> 2); i++) {
                for (int j = 0; j < (out_size >> 9);){
                    #pragma HLS pipeline II=1
                    if(!lut_weight_idx_fifo.empty()){
                        tapa::vec_t<ap_uint<8>, 64> tmp; lut_weight_idx_fifo.try_read(tmp);
                        for (int ii = 0; ii < 4; ii++) {
                            #pragma HLS unroll
                            for(int k = 0; k < 16; k++) {
                                #pragma HLS unroll
                                linear_lut[i*4+ii][k][j] = tmp[ii*16+k];
                            }
                        }
                        j++;
                    }
                }
            }

            for (int i = 0; i < (out_size >> 7);) {
                #pragma HLS pipeline II=1
                if(!lut_weight_idx_fifo.empty()){
                    tapa::vec_t<ap_uint<8>, 64> tmp; lut_weight_idx_fifo.try_read(tmp);
                    for(int k = 0; k < 64; k++) {
                        #pragma HLS unroll
                        weight_idx[i * 128 + k * 2] = ap_uint<4>(tmp[k](3, 0));
                        weight_idx[i * 128 + k * 2 + 1] = ap_uint<4>(tmp[k](7, 4));
                    }
                    i++;
                }
            }

            for (int i = 0; i < L; i++) {

                auto idx = idx_fifo.read();

                for (int j = 0; j < (out_size >> 10); j++) {
                    #pragma HLS pipeline II=1

                    ap_uint<8> linear_out_reg[1024];
                    #pragma HLS array_partition variable=linear_out_reg complete
                    ap_uint<8> lut_reg[2][16];
                    #pragma HLS array_partition variable=lut_reg complete dim=1
                    #pragma HLS array_partition variable=lut_reg complete dim=2

                    for(int lut_i = 0; lut_i < 2; lut_i++) {
                        #pragma HLS unroll
                        for (int k = 0; k < 16; k++) {
                            #pragma HLS unroll
                            lut_reg[lut_i][k] = linear_lut[idx][k][j*2+lut_i];
                        }
                    }
                    for (int k = 0; k < 1024; k++) {
                        #pragma HLS unroll
                        int w_idx = weight_idx[j * 1024 + k].to_int();
                        linear_out_reg[k] = lut_reg[k>>9][w_idx];
                    }
                    for (int k = 0; k < 32; k++) {
                        #pragma HLS unroll
                        tapa::vec_t<ap_uint<48>, 8> out_vec;
                        for (int m = 0; m < 8; m++) {
                            #pragma HLS unroll
                            for (int p = 0; p < 4; p++){
                                #pragma HLS unroll
                                out_vec[m](p * 12 + 11, p * 12) = ap_uint<12>(linear_out_reg[k * 32 + m * 4 + p]);
                            }
                        }
                        outbound_fifo[k].write(out_vec);
                    }
                }
            }
        }
    }
}

void memory_matcher_tail_acc(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istreams<tapa::vec_t<ap_uint<48>, 8>, 8>& inbound_fifo,
    tapa::istream<ap_uint<64>>& scale_zero_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    ap_uint<64> linear_out[MAX_SEQ_LEN][MAX_OUT_SIZE_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=128 dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_2P impl=URAM

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < (out_size >> 8); j++){
            #pragma HLS pipeline II=1
            for (int k = 0; k < 128; k++) {
                #pragma HLS unroll
                linear_out[i][(j << 7) + k] = ap_uint<64>(0); // Initialize output
            }
        }
    }

    auto pack_dequant = scale_zero_fifo.read();
    float scale = tapa::bit_cast<float>(ap_uint<32>(pack_dequant(31, 0)));
    float zeropoint = tapa::bit_cast<float>(ap_uint<32>(pack_dequant(63, 32)));

    // read indices and parallel match
    for (int r = 0; r < (in_size >> 3); r++) {

        for (int i = 0; i < L; i++) {

            for (int j = 0; j < (out_size >> 8); j++) {
                #pragma HLS pipeline II=1
                
                for (int k = 0; k < 8; k++) {
                    #pragma HLS unroll
                    auto tmp_vec = inbound_fifo[k].read();
                    for (int m = 0; m < 8; m++) {
                        #pragma HLS unroll
                        ap_uint<44> op2 = tmp_vec[m];
                        for (int p = 0; p < 2; p++) {
                            #pragma HLS unroll
                            ap_uint<64> acc_reg = linear_out[i][j*128 + k*16 + m*2 + p];
                            ap_uint<44> simd_a = ap_uint<44>(acc_reg(43, 0));
                            ap_uint<44> simd_b = ap_uint<44>((ap_uint<22>(op2(p*22+21, p*22+11)),ap_uint<22>(op2(p*22+10, p*22))));
                            ap_uint<44> simd_out = simd_a + simd_b;
                            linear_out[i][j*128 + k*16 + m * 2 + p] = simd_out;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < (out_size >> 4); i++) {
        for (int j = 0; j < L; j++){
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

template<int dim_size = QK_DIM_DIV_2, int para_access_factor = 128>
void memory_matcher_attn(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    // prefetch LUT for linear layer
    ap_uint<64> linear_lut[n_cent][dim_size];
    #pragma HLS array_partition variable=linear_lut cyclic factor=para_access_factor dim=2
    #pragma HLS bind_storage variable=linear_lut type=RAM_1P impl=BRAM

    ap_uint<64> linear_out[MAX_SEQ_LEN][dim_size];
    #pragma HLS array_partition variable=linear_out cyclic factor=para_access_factor dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_2P impl=BRAM

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size/2; j+=para_access_factor){
            #pragma HLS pipeline II=1
            for (int k = 0; k < para_access_factor; k++) {
                #pragma HLS unroll
                linear_out[i][j + k] = ap_uint<64>(0); // Initialize output
            }
        }
    }

    // read indices and parallel match
    for (int r = 0; r < in_size; r++) {

        for(int i = 0; i < n_cent; i++) {
            for (int j = 0; j < (out_size >> 4);){
                #pragma HLS pipeline II=1
                if(!lut_fifo.empty()){
                    tapa::vec_t<ap_uint<64>, 8> tmp; lut_fifo.try_read(tmp);
                    for(int k = 0; k < 8; k++) {
                        #pragma HLS unroll
                        linear_lut[i][j * 8 + k] = tmp[k];
                    }
                    j++;
                }
            }
        }

        for (int i = 0; i < L; i++) {

            auto idx = idx_fifo.read();

            for (int j = 0; j < out_size/2; j+=para_access_factor){
                #pragma HLS pipeline II=1
                for (int k = 0; k < para_access_factor; k++) {
                    #pragma HLS unroll
                    auto pack_psum = linear_out[i][j + k];
                    auto pack_lut = linear_lut[idx][j + k];
                    float op1 = tapa::bit_cast<float>(ap_uint<32>(pack_psum(31, 0)));
                    float op2 = tapa::bit_cast<float>(ap_uint<32>(pack_psum(63, 32)));
                    float op1_l = tapa::bit_cast<float>(ap_uint<32>(pack_lut(31, 0)));
                    float op2_l = tapa::bit_cast<float>(ap_uint<32>(pack_lut(63, 32)));

                    op1 += op1_l;
                    op2 += op2_l;
                    //repack
                    linear_out[i][j + k] = ap_uint<64>((tapa::bit_cast<ap_uint<32>>(op2), tapa::bit_cast<ap_uint<32>>(op1)));
                }
            }
        }
    }

    // write out heads
    for (int r = 0; r < (out_size / HEAD_DIM); r++) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < (HEAD_DIM >> 4); j++){
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> tmp;
                for (int k = 0; k < 8; k++) {
                    #pragma HLS unroll
                    tmp[k*2] = tapa::bit_cast<float>(ap_uint<32>(linear_out[i][r * HEAD_DIM_DIV_2 + j * 8 + k](31, 0)));
                    tmp[k*2 + 1] = tapa::bit_cast<float>(ap_uint<32>(linear_out[i][r * HEAD_DIM_DIV_2 + j * 8 + k](63, 32)));
                }
                out_fifo.write(tmp);
            }
        }
    }
}

void memory_matcher_out_proj(
    const int L,
    const int in_size,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    // prefetch LUT for linear layer
    ap_uint<64> linear_lut[n_cent][HIDDEN_DIM_DIV_2];
    #pragma HLS array_partition variable=linear_lut cyclic factor=64 dim=2
    #pragma HLS bind_storage variable=linear_lut type=RAM_1P impl=BRAM

    ap_uint<64> linear_out[MAX_SEQ_LEN][HIDDEN_DIM_DIV_2];
    #pragma HLS array_partition variable=linear_out cyclic factor=64 dim=2
    #pragma HLS bind_storage variable=linear_out type=RAM_2P impl=BRAM

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM_DIV_2; j+=64){
            #pragma HLS pipeline II=1
            for (int k = 0; k < 64; k++) {
                #pragma HLS unroll
                linear_out[i][j + k] = ap_uint<64>(0); // Initialize output
            }
        }
    }

    // read indices and parallel match
    for (int r = 0; r < in_size; r++) {

        for(int i = 0; i < n_cent; i++) {
            for (int j = 0; j < (HIDDEN_DIM >> 4);){
                #pragma HLS pipeline II=1
                if(!lut_fifo.empty()){
                    tapa::vec_t<ap_uint<64>, 8> tmp; lut_fifo.try_read(tmp);
                    for(int k = 0; k < 8; k++) {
                        #pragma HLS unroll
                        linear_lut[i][j * 8 + k] = tmp[k];
                    }
                    j++;
                }
            }
        }

        for (int i = 0; i < L; i++) {

            auto idx = idx_fifo.read();

            for (int j = 0; j < HIDDEN_DIM_DIV_2; j+=64){
                #pragma HLS pipeline II=1
                for (int k = 0; k < 64; k++) {
                    #pragma HLS unroll
                    auto pack_psum = linear_out[i][j + k];
                    auto pack_lut = linear_lut[idx][j + k];
                    float op1 = tapa::bit_cast<float>(ap_uint<32>(pack_psum(31, 0)));
                    float op2 = tapa::bit_cast<float>(ap_uint<32>(pack_psum(63, 32)));
                    float op1_l = tapa::bit_cast<float>(ap_uint<32>(pack_lut(31, 0)));
                    float op2_l = tapa::bit_cast<float>(ap_uint<32>(pack_lut(63, 32)));

                    op1 += op1_l;
                    op2 += op2_l;
                    //repack
                    linear_out[i][j + k] = ap_uint<64>((tapa::bit_cast<ap_uint<32>>(op2), tapa::bit_cast<ap_uint<32>>(op1)));
                }
            }
        }
    }

    // write out heads
    for(int i = 0; i < (L >> 4); i++) {
        for (int j = 0; j < (HIDDEN_DIM >> 4); j++) {
            float transposed_reg[16][16];
            #pragma HLS array_partition variable=transposed_reg complete dim=1
            #pragma HLS array_partition variable=transposed_reg complete dim=2
            for (int ii = 0; ii < 16; ii++) {
                #pragma HLS pipeline II=1
                for (int jj = 0; jj < 8; jj++) {
                    #pragma HLS unroll
                    transposed_reg[ii][jj*2] = tapa::bit_cast<float>(ap_uint<32>(linear_out[i * 16 + ii][j * 8 + jj](31, 0)));
                    transposed_reg[ii][jj*2 + 1] = tapa::bit_cast<float>(ap_uint<32>(linear_out[i * 16 + ii][j * 8 + jj](63, 32)));
                }
            }
            for (int jj = 0; jj < 16; jj++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> tmp;
                for (int ii = 0; ii < 16; ii++) {
                    #pragma HLS unroll
                    tmp[ii] = transposed_reg[ii][jj];
                }
                out_fifo.write(tmp);
            }
        }
    }
}

void memory_matcher_test(
    const int L,
    const int in_size,
    const int out_size,
    tapa::istream<idx_t>& idx_fifo,
    tapa::istream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    memory_matcher<16, 32>(L, in_size, out_size, idx_fifo, lut_fifo, out_fifo);
}

#ifndef TIMING
#define TIMING

void measure_cycle(tapa::istream<bool>& fifo_fin, tapa::mmap<int> cycle_count){
    for(int cycle = 0;;cycle++){
        if(!fifo_fin.empty()){
            fifo_fin.read(nullptr);
            cycle_count[0] = cycle;
            break;
        }
    }
}
#endif // TIMING

//there are some problems with tapa fast cosim in axi interface modeling using ap_uint
//top function for testing
void imm(
    const int L,
    const int in_size,
    const int out_size,
    const int total_size,
    tapa::mmaps<int, 8> idx_buffer,
    tapa::mmaps<tapa::vec_t<ap_uint<8>, 64>, 8> lut_weight_idx_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> linear_out_buffer,
    tapa::mmap<ap_uint<64>> scale_zero_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::streams<idx_t, 8> idx_fifo("idx_fifo");
    tapa::streams<tapa::vec_t<ap_uint<8>, 64>, 8> lut_weight_idx_fifo("lut_weight_idx_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 8> psum_0_fifo("psum_0_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 8> psum_1_fifo("psum_1_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 8> psum_2_fifo("psum_2_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 8> psum_3_fifo("psum_3_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 8> psum_4_fifo("psum_4_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 8> psum_5_fifo("psum_5_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 8> psum_6_fifo("psum_6_fifo");
    tapa::streams<tapa::vec_t<ap_uint<48>, 8>, 8> psum_7_fifo("psum_7_fifo");
    tapa::stream<ap_uint<64>> scale_zero_fifo("scale_zero_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join, 8>(index_reader, L, in_size, idx_buffer, idx_fifo)
        .invoke<tapa::join, 8>(lut_reader, total_size, lut_weight_idx_buffer, lut_weight_idx_fifo)
        .invoke<tapa::join>(scale_zero_reader, 1, scale_zero_buffer, scale_zero_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_head, L, idx_fifo, lut_weight_idx_fifo, psum_0_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_0_fifo, psum_1_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_1_fifo, psum_2_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_2_fifo, psum_3_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_3_fifo, psum_4_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_4_fifo, psum_5_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_5_fifo, psum_6_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, idx_fifo, lut_weight_idx_fifo, psum_6_fifo, psum_7_fifo)
        .invoke<tapa::join>(memory_matcher_tail_acc, L, in_size, out_size, psum_7_fifo, scale_zero_fifo, out_fifo)
        .invoke<tapa::join>(linear_out_writer, L, out_size, out_fifo, linear_out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}

#endif