#ifndef _RMS_NORM_H_
#define _RMS_NORM_H_

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

constexpr int HIDDEN_DIM = 896;
constexpr float EPSILON = 1e-6f;
constexpr float R_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

void input_reader(
    const int L,
    tapa::async_mmap<tapa::vec_t<float, 16>>& input_buffer,
    tapa::ostream<tapa::vec_t<float, 16>>& input_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((L * HIDDEN_DIM) >> 4);){
        #pragma HLS pipeline II=1
        if((i_req < ((L * HIDDEN_DIM) >> 4)) & !input_buffer.read_addr.full()){
            input_buffer.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!input_buffer.read_data.empty()){
            tapa::vec_t<float, 16> tmp;
            input_buffer.read_data.try_read(tmp);
            input_fifo.write(tmp);
            ++i_resp;
        }
    }
}

void weight_reader(
    tapa::async_mmap<tapa::vec_t<float, 16>>& weight_buffer,
    tapa::ostream<tapa::vec_t<float, 16>>& weight_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < (HIDDEN_DIM >> 4);){
        #pragma HLS pipeline II=1
        if((i_req < (HIDDEN_DIM >> 4)) & !weight_buffer.read_addr.full()){
            weight_buffer.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!weight_buffer.read_data.empty()){
            tapa::vec_t<float, 16> tmp;
            weight_buffer.read_data.try_read(tmp);
            weight_fifo.write(tmp);
            ++i_resp;
        }
    }
}

void out_writer(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& out_fifo,
    tapa::async_mmap<tapa::vec_t<float, 16>>& out_buffer,
    tapa::ostream<bool>& fifo_fin
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((L * HIDDEN_DIM) >> 4);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < ((L * HIDDEN_DIM) >> 4)) & !out_fifo.empty() & !out_buffer.write_addr.full() & !out_buffer.write_data.full()){
            out_buffer.write_addr.try_write(i_req);
            tapa::vec_t<float, 16> tmp; out_fifo.try_read(tmp);
            out_buffer.write_data.try_write(tmp);
            ++i_req;
        }
        bool success = false;
        auto resp = out_buffer.write_resp.read(success);
        if(success){
            i_resp += unsigned(resp)+1;
        }
    }
    fifo_fin.write(true);
}

void rms_norm(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& weight_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {

    //shared rms weight
    float weight[HIDDEN_DIM];
    #pragma HLS array_partition variable=weight cyclic factor=16
    for(int i = 0; i < (HIDDEN_DIM >> 4); i++){
        #pragma HLS pipeline II=1
        auto weight_vec = weight_fifo.read();
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            weight[i * 16 + j] = weight_vec[j];
        }
    }

    for(int i = 0; i < (L >> 4); i++){
        float variance[16];
        #pragma HLS array_partition variable=variance complete
        float input_buf[16][HIDDEN_DIM];
        #pragma HLS array_partition variable=input_buf complete dim=1
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            variance[j] = 0.0f;
        }

        for(int j = 0; j < HIDDEN_DIM; j++){
            #pragma HLS pipeline II=1
            auto input_vec = input_fifo.read();
            for(int k = 0; k < 16; k++){
                #pragma HLS unroll
                input_buf[k][j] = input_vec[k];
                variance[k] += input_vec[k] * input_vec[k];
            }
        }

        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            variance[j] = 1.0f / std::sqrt(variance[j] * R_HIDDEN_DIM + EPSILON);
        }

        for(int j = 0; j < HIDDEN_DIM; j++){
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 16> out_vec;
            float w = weight[j];
            for(int k = 0; k < 16; k++){
                #pragma HLS unroll
                out_vec[k] = input_buf[k][j] * variance[k] * w;
            }
            out_fifo.write(out_vec);
        }
        
    }
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
void rms_norm_top(
    const int L,
    tapa::mmap<tapa::vec_t<float, 16>> input_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> weight_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 16>> input_fifo("input_fifo");
    tapa::stream<tapa::vec_t<float, 16>> weight_fifo("weight_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(input_reader, L, input_buffer, input_fifo)
        .invoke<tapa::join>(weight_reader, weight_buffer, weight_fifo)
        .invoke<tapa::join>(rms_norm, L, input_fifo, weight_fifo, out_fifo)
        .invoke<tapa::join>(out_writer, L, out_fifo, out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}

#endif