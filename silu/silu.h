#ifndef _SILU_H_
#define _SILU_H_

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

constexpr int FFN_DIM = 4864;

void silu_input_reader(
    const int L,
    tapa::async_mmap<tapa::vec_t<float, 16>>& input_buffer,
    tapa::ostream<tapa::vec_t<float, 16>>& input_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((L * FFN_DIM) >> 4);){
        #pragma HLS pipeline II=1
        if((i_req < ((L * FFN_DIM) >> 4)) & !input_buffer.read_addr.full()){
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

void silu_out_writer(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& out_fifo,
    tapa::async_mmap<tapa::vec_t<float, 16>>& out_buffer,
    tapa::ostream<bool>& fifo_fin
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((L * FFN_DIM) >> 4);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < ((L * FFN_DIM) >> 4)) & !out_fifo.empty() & !out_buffer.write_addr.full() & !out_buffer.write_data.full()){
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

void silu(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& output_fifo
) {

    for(int r = 0; r < L; r++){
        for(int i = 0; i < (FFN_DIM >> 4); i++){
            #pragma HLS pipeline II=1
            tapa::vec_t<float, 16> input_vec = input_fifo.read();
            tapa::vec_t<float, 16> output_vec;
            for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                float slope = 0.0f;
                float intercept = 0.0f;
                // piecewise linear approximation of silu
                if (input_vec[j] < -8.000f) {
                    slope = 0.0f;
                    intercept = 0.0f;
                }
                else if (input_vec[j] < -4.000000f) {
                    slope = -0.017316f;
                    intercept = -0.141207f;
                }
                else if (input_vec[j] < -2.000000f) { // [-4.000000f, -2.000000f)
                    slope = -0.083231f;
                    intercept = -0.404867f;
                }
                else if (input_vec[j] < -1.000000f) { // [-2.000000f, -1.000000f)
                    slope = -0.030536f;
                    intercept = -0.299477f;
                }
                else if (input_vec[j] < 0.000000f) { // [-1.000000f, 0.000000f)
                    slope = 0.268941f;
                    intercept = 0.0f;
                }
                else if (input_vec[j] < 1.000000f) { // [0.000000f, 1.000000f)
                    slope = 0.731059f;
                    intercept = 0.0f;
                }
                else if (input_vec[j] < 2.000000f) { // [1.000000f, 2.000000f)
                    slope = 1.030536f;
                    intercept = -0.299477f;
                }
                else if (input_vec[j] < 4.000000f) { // [2.000000f, 4.000000f)
                    slope = 1.083231f;
                    intercept = -0.404867f;
                }
                else { // x >= 4.000000f
                    slope = 1.0f;
                    intercept = 0.0f;
                }
                output_vec[j] = slope * input_vec[j] + intercept;
            }
            output_fifo.write(output_vec);
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
void silu_top(
    const int L,
    tapa::mmap<tapa::vec_t<float, 16>> input_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 16>> input_fifo("input_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(silu_input_reader, L, input_buffer, input_fifo)
        .invoke<tapa::join>(silu, L, input_fifo, out_fifo)
        .invoke<tapa::join>(silu_out_writer, L, out_fifo, out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}

#endif