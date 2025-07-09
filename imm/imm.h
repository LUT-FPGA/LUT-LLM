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
    for(int i_req = 0, i_resp = 0; i_resp < (L * in_size);){
        #pragma HLS pipeline II=1
		if((i_req < (L * in_size)) & !idx_buffer.read_addr.full()){
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


void lut_reader(
    const int in_size,
    const int out_size,
    tapa::async_mmap<tapa::vec_t<float, 16>>& lut_buffer,
    tapa::ostream<tapa::vec_t<ap_uint<64>, 8>>& lut_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((out_size * in_size) << 2);){
        #pragma HLS pipeline II=1
        if((i_req < ((out_size * in_size) << 2)) & !lut_buffer.read_addr.full()){
            lut_buffer.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!lut_buffer.read_data.empty()){
            tapa::vec_t<float, 16> tmp;
            lut_buffer.read_data.try_read(tmp);
            tapa::vec_t<ap_uint<64>, 8> tmp64;
            for(int i = 0; i < 8; ++i) {
                #pragma HLS unroll
                tmp64[i] = ap_uint<64>((tapa::bit_cast<ap_uint<32>>(tmp[i*2 + 1]), tapa::bit_cast<ap_uint<32>>(tmp[i*2])));
            }
            lut_fifo.write(tmp64);
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
    tapa::mmap<int> idx_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> linear_out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<idx_t> idx_fifo("idx_fifo");
    tapa::stream<tapa::vec_t<ap_uint<64>, 8>> lut_fifo("lut_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(index_reader, L, in_size, idx_buffer, idx_fifo)
        .invoke<tapa::join>(lut_reader, in_size, out_size, lut_buffer, lut_fifo)
        .invoke<tapa::join>(memory_matcher_test, L, in_size, out_size, idx_fifo, lut_fifo, out_fifo)
        .invoke<tapa::join>(linear_out_writer, L, out_size, out_fifo, linear_out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}

#endif