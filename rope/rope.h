#ifndef _ROPE_H_
#define _ROPE_H_

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

void rope_input_reader(
    const int L,
    tapa::async_mmap<tapa::vec_t<float, 16>>& input_buffer,
    tapa::ostream<tapa::vec_t<float, 16>>& input_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((L * HEAD_DIM) >> 4);){
        #pragma HLS pipeline II=1
        if((i_req < ((L * HEAD_DIM) >> 4)) & !input_buffer.read_addr.full()){
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

void rope_out_writer(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& out_fifo,
    tapa::async_mmap<tapa::vec_t<float, 16>>& out_buffer,
    tapa::ostream<bool>& fifo_fin
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((L * HEAD_DIM) >> 4);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < ((L * HEAD_DIM) >> 4)) & !out_fifo.empty() & !out_buffer.write_addr.full() & !out_buffer.write_data.full()){
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

template <int iter = 1>
void apply_rotary_pos_emb(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& sin_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& cos_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    //prefetch rope embeddings
    float sin[MAX_SEQ_LEN][HEAD_DIM];
    float cos[MAX_SEQ_LEN][HEAD_DIM];

    #pragma HLS array_partition variable=sin cyclic factor=16 dim=2
    #pragma HLS array_partition variable=cos cyclic factor=16 dim=2

    for(int i = 0; i < L; i++){
        for(int j = 0; j < (HEAD_DIM >> 4); j++){
            #pragma HLS pipeline II=1
            auto sin_vec = sin_fifo.read();
            auto cos_vec = cos_fifo.read();
            for(int k = 0; k < 16; k++){
                #pragma HLS unroll
                sin[i][j*16+k] = sin_vec[k];
                cos[i][j*16+k] = cos_vec[k];
            }
        }
    }

    for(int r = 0; r < iter; r++) {
        for (int i = 0; i < L; i++){
            // #pragma HLS dataflow
            #pragma HLS loop_tripcount min=32 max=128
             //read input and apply embeddings
            float input_buf_sin[HEAD_DIM];
            #pragma HLS array_partition variable=input_buf_sin cyclic factor=16
            float input_buf_cos[HEAD_DIM];
            #pragma HLS array_partition variable=input_buf_cos cyclic factor=16

            for(int j = 0; j < (HEAD_DIM >> 4); j++){
                #pragma HLS pipeline II=1
                auto input_vec = input_fifo.read();
                for(int k = 0; k < 16; k++){
                    #pragma HLS unroll
                    input_buf_sin[j*16+k] = input_vec[k] * sin[i][j*16+k];
                    input_buf_cos[j*16+k] = input_vec[k] * cos[i][j*16+k];
                }
            }

            for(int j = 0; j < (HEAD_DIM >> 4); j++){
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> out_vec;
                if(j < (HEAD_DIM_DIV_2 >> 4)){
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        out_vec[k] = input_buf_cos[j*16+k] - input_buf_sin[j*16+HEAD_DIM_DIV_2+k];
                    }
                } else {
                    for(int k = 0; k < 16; k++){
                        #pragma HLS unroll
                        out_vec[k] = input_buf_cos[j*16+k] + input_buf_sin[j*16-HEAD_DIM_DIV_2+k];
                    }
                }
                out_fifo.write(out_vec);
            }

        }
    }
}

void apply_rotary_pos_emb_inst(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& sin_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& cos_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    apply_rotary_pos_emb<1>(L, input_fifo, sin_fifo, cos_fifo, out_fifo);
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
void rope(
    const int L,
    tapa::mmap<tapa::vec_t<float, 16>> input_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> sin_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> cos_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 16>> input_fifo("input_fifo");
    tapa::stream<tapa::vec_t<float, 16>> sin_fifo("sin_fifo");
    tapa::stream<tapa::vec_t<float, 16>> cos_fifo("cos_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(rope_input_reader, L, input_buffer, input_fifo)
        .invoke<tapa::join>(rope_input_reader, L, sin_buffer, sin_fifo)
        .invoke<tapa::join>(rope_input_reader, L, cos_buffer, cos_fifo)
        .invoke<tapa::join>(apply_rotary_pos_emb_inst, L, input_fifo, sin_fifo, cos_fifo, out_fifo)
        .invoke<tapa::join>(rope_out_writer, L, out_fifo, out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}

#endif