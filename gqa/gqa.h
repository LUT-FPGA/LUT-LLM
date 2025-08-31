#ifndef _GQA_H_
#define _GQA_H_

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


void gqa_input_reader( 
    const int L,
    tapa::async_mmap<tapa::vec_t<float, 16>>& input_buffer,
    tapa::ostream<tapa::vec_t<float, 16>>& input_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((L * QKV_DIM) >> 4);){
        #pragma HLS pipeline II=1
        if((i_req < ((L * QKV_DIM) >> 4)) & !input_buffer.read_addr.full()){
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

void gqa_arbiter(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& qk_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& v_fifo
) {
    // the order is:
    // v[0], k[0], q[0:1], v[1], k[1], q[2:3], ..., v[7], k[7], q[14:15]
    // each block is L * HEAD_DIM / 16
    for (int i = 0; i < (L * QKV_DIM) >> 4; i++) {
        #pragma HLS pipeline II=1
        tapa::vec_t<float, 16> tmp = input_fifo.read();
        int idx = i / (L * HEAD_DIM >> 4);
        if (idx % (HEAD_PER_GROUP + 2) == 0) {
            v_fifo.write(tmp);
        } else {
            qk_fifo.write(tmp);
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

void pe_16x16(
    const tapa::vec_t<float, 16>& q,
    const tapa::vec_t<float, 16>& k,
    float qk_reg[16][16]
) {
    #pragma HLS inline off
    for (int i = 0; i < 16; i++) {
        #pragma HLS unroll
        for (int j = 0; j < 16; j++) {
            #pragma HLS unroll
            qk_reg[i][j] += q[i] * k[j];
        }
    }
}

void pe_16x16_2x128_simd(
    const float op1_reg[16][16],
    const float op2_reg[16],
    float output_reg[16][16]
) {
    #pragma HLS inline off
    // 16x16 array if prefill, otherwise 1x256
    for (int i = 0; i < 16; i++) {
        #pragma HLS unroll
        for (int j = 0; j < 16; j++) {
            #pragma HLS unroll
            output_reg[i][j] += op1_reg[i][j] * op2_reg[j];
        }
    }
}

void kv_cache_reader(
    const ap_uint<10> L_inst,
    tapa::async_mmap<tapa::vec_t<float, 16>>& kv_cache_buffer,
    tapa::ostream<tapa::vec_t<float, 16>>& kv_cache_fifo
) {
    const int L_prefill = ap_uint<9>(L_inst(8, 0)).to_int();
    if(L_inst[9] == 1) {
        for(int i_req = 0, i_resp = 0; i_resp < (((L_prefill-1) * KV_CACHE_DIM) >> 4);){
            #pragma HLS pipeline II=1
            if((i_req < (((L_prefill-1) * KV_CACHE_DIM) >> 4)) & !kv_cache_buffer.read_addr.full()){
                kv_cache_buffer.read_addr.try_write(i_req);
                ++i_req;
            }
            if(!kv_cache_buffer.read_data.empty()){
                tapa::vec_t<float, 16> tmp;
                kv_cache_buffer.read_data.try_read(tmp);
                kv_cache_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

void gemm_gqa_qk(
    tapa::istream<ap_uint<10>>& L_in_fifo,
    tapa::ostream<ap_uint<10>>& L_out_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& k_cache_fifo,
    tapa::istream<tapa::vec_t<float, 32>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& pre_softmax_fifo
) {
    // compute grouped query attention

    const ap_uint<10> L_inst = L_in_fifo.read();
    L_out_fifo.write(L_inst);

    const int L_prefill = ap_uint<9>(L_inst(8, 0)).to_int();
    const int L = (L_inst[9] == 1) ? 1 : L_prefill;

    float k_buf[MAX_SEQ_LEN][KV_CACHE_DIM];
    #pragma HLS array_partition variable=k_buf cyclic factor=16 dim=1
    #pragma HLS array_partition variable=k_buf cyclic factor=32 dim=2
    #pragma HLS bind_storage variable=k_buf type=RAM_1P impl=BRAM

    if(L_inst[9] == 1) {
        // load kv cache
        for (int i = 0; i < (L_prefill-1); i++) {
            for (int j = 0; j < (KV_CACHE_DIM >> 4); j++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> tmp = k_cache_fifo.read(); 
                for (int k = 0; k < 16; k++) {
                    #pragma HLS unroll
                    k_buf[i][j*16+k] = tmp[k];
                }
            }
        }
    }

    for (int g = 0; g < NUM_GROUPS; g++){ // groups

        // step 1: load k

        // 16 x 16 x 16 pe array, reconfigurable to parallel group, same size
        const int offset = g * HEAD_DIM;

        load_kv: for (int i = 0; i < L; i++) {
            const int acc_idx = (L_inst[9] == 0) ? i : (L_prefill-1);
            for (int j = 0; j < (HEAD_DIM >> 5); j++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 32> tmp = input_fifo.read(); 
                for (int k = 0; k < 32; k++) {
                    #pragma HLS unroll
                    k_buf[acc_idx][offset+j*32+k] = tmp[k];
                }
            }
        }


        for (int r = 0; r < HEAD_PER_GROUP; r++) {
            
            // step 2: compute QK^T

            for(int i = 0; i < L; i++) {

                float q_buf_row[HEAD_DIM];
                #pragma HLS array_partition variable=q_buf_row cyclic factor=32
                
                for (int j = 0; j < (L_prefill >> 4); j++) {
                    #pragma HLS loop_tripcount min=32 max=128

                    float qk_reg_row[16][32];
                    #pragma HLS array_partition variable=qk_reg_row complete dim=1
                    #pragma HLS array_partition variable=qk_reg_row complete dim=2

                    init_qk: for (int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        for (int l = 0; l < 32; l++) {
                            #pragma HLS unroll
                            qk_reg_row[k][l] = 0.0f;
                        }
                    }

                    compute_macc: for (int k = 0; k < (HEAD_DIM >> 5); k++) {
                        #pragma HLS pipeline II=1
                        #pragma HLS loop_tripcount min=4 max=4

                        tapa::vec_t<float, 32> q_vec;

                        if (j == 0){
                            // assign to q_buf
                            q_vec = input_fifo.read();
                            for(int kk = 0; kk < 32; kk++) {
                                #pragma HLS unroll
                                q_buf_row[k*32+kk] = q_vec[kk];
                            }
                        } else {
                            // assign to q_vec
                            for(int kk = 0; kk < 32; kk++) {
                                #pragma HLS unroll
                                q_vec[kk] = q_buf_row[k*32+kk];
                            }
                        }

                        //macc
                        for(int jj = 0; jj < 16; jj++) {
                            #pragma HLS unroll
                            for(int kk = 0; kk < 32; kk++) {
                                #pragma HLS unroll
                                qk_reg_row[jj][kk] += q_vec[kk] * k_buf[j*16+jj][offset+k*32+kk];
                            }
                        }

                    }

                    // reduction
                    reduction: for (int k = 1; k < 8; k++) {
                        #pragma HLS pipeline II=1
                        for (int kk = 0; kk < 4; kk++){
                            #pragma HLS unroll
                            for (int l = 0; l < 16; l++) {
                                #pragma HLS unroll
                                qk_reg_row[l][kk*8] += qk_reg_row[l][kk*8+k];
                            }
                        }
                    }

                    tapa::vec_t<float, 16> qk_pre_softmax;

                    // final reduction and assignment
                    for (int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        qk_pre_softmax[k] = (qk_reg_row[k][0] + qk_reg_row[k][8] + qk_reg_row[k][16] + qk_reg_row[k][24]);
                    }

                    pre_softmax_fifo.write(qk_pre_softmax);
                }
            }
        }
    }
}

void gemm_gqa_av(
    tapa::istream<ap_uint<10>>& L_in_fifo,
    tapa::ostream<ap_uint<10>>& L_out_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& v_cache_fifo,
    tapa::istream<tapa::vec_t<float, 32>>& input_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& post_softmax_fifo,
    tapa::ostream<tapa::vec_t<float, 32>>& output_fifo
){

    const ap_uint<10> L_inst = L_in_fifo.read();
    L_out_fifo.write(L_inst);

    const int L_prefill = ap_uint<9>(L_inst(8, 0)).to_int();
    const int L = (L_inst[9] == 1) ? 1 : L_prefill;

    float v_buf[MAX_SEQ_LEN][KV_CACHE_DIM];
    #pragma HLS array_partition variable=v_buf cyclic factor=16 dim=1
    #pragma HLS array_partition variable=v_buf cyclic factor=32 dim=2
    #pragma HLS bind_storage variable=v_buf type=RAM_1P impl=BRAM

    if(L_inst[9] == 1) {
        // load kv cache
        for (int i = 0; i < (L_prefill-1); i++) {
            for (int j = 0; j < (KV_CACHE_DIM >> 4); j++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> tmp = v_cache_fifo.read(); 
                for (int k = 0; k < 16; k++) {
                    #pragma HLS unroll
                    v_buf[i][j*16+k] = tmp[k];
                }
            }
        }
    }

    for (int g = 0; g < NUM_GROUPS; g++){ // groups

        // step 1: load v

        // 16 x 16 x 16 pe array, reconfigurable to parallel group, same size
        const int offset = g * HEAD_DIM;

        load_kv: for (int i = 0; i < L; i++) {
            const int acc_idx = (L_inst[9] == 0) ? i : (L_prefill-1);
            for (int j = 0; j < (HEAD_DIM >> 5); j++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 32> tmp = input_fifo.read(); 
                for (int k = 0; k < 32; k++) {
                    #pragma HLS unroll
                    v_buf[acc_idx][offset+j*32+k] = tmp[k];
                }
            }
        }

        for (int r = 0; r < HEAD_PER_GROUP; r++) {

            // step 3: write batch of rows for softmax and compute AV
            for (int i = 0; i < L; i++) {

                float qk_buf_row[MAX_SEQ_LEN];
                #pragma HLS array_partition variable=qk_buf_row cyclic factor=16

                for (int j = 0; j < (HEAD_DIM >> 5); j++) {
                    #pragma HLS dataflow
                    #pragma HLS loop_tripcount min=8 max=8

                    float av_reg_row[32][16];
                    #pragma HLS array_partition variable=av_reg_row complete dim=1
                    #pragma HLS array_partition variable=av_reg_row complete dim=2

                    init_qk: for (int k = 0; k < 32; k++) {
                        #pragma HLS unroll
                        for (int l = 0; l < 16; l++) {
                            #pragma HLS unroll
                            av_reg_row[k][l] = 0.0f;
                        }
                    }

                    compute_macc: for (int k = 0; k < (L_prefill >> 4); k++) {
                        #pragma HLS pipeline II=1
                        #pragma HLS loop_tripcount min=2 max=8
                        
                        tapa::vec_t<float, 16> qk_vec;

                        if (j == 0){
                            // assign to q_buf
                            qk_vec = post_softmax_fifo.read();
                            for(int kk = 0; kk < 16; kk++) {
                                #pragma HLS unroll
                                qk_buf_row[k*16+kk] = qk_vec[kk];
                            }
                        } else {
                            // assign to q_vec
                            for(int kk = 0; kk < 16; kk++) {
                                #pragma HLS unroll
                                qk_vec[kk] = qk_buf_row[k*16+kk];
                            }
                        }

                        //macc
                        for(int jj = 0; jj < 32; jj++) {
                            #pragma HLS unroll
                            for(int kk = 0; kk < 16; kk++) {
                                #pragma HLS unroll
                                av_reg_row[jj][kk] += qk_vec[kk] * v_buf[k*16+kk][offset+j*32+jj];
                            }
                        }

                    }

                    // reduction
                    reduction: for (int k = 1; k < 8; k++) {
                        #pragma HLS pipeline II=1
                        for (int kk = 0; kk < 2; kk++){
                            #pragma HLS unroll
                            for (int l = 0; l < 32; l++) {
                                #pragma HLS unroll
                                av_reg_row[l][kk*8] += av_reg_row[l][kk*8+k];
                            }
                        }
                    }

                    tapa::vec_t<float, 32> output_vec;

                    // final reduction and assignment
                    for (int k = 0; k < 32; k++) {
                        #pragma HLS unroll
                        output_vec[k] = (av_reg_row[k][0] + av_reg_row[k][8]);
                    }

                    output_fifo.write(output_vec);
                }
            }
        }
    }
}

void softmax(
    tapa::istream<ap_uint<10>>& L_in_fifo,
    tapa::ostream<ap_uint<10>>& L_out_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& pre_softmax_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& post_softmax_fifo
) {

    const ap_uint<10> L_inst = L_in_fifo.read();
    L_out_fifo.write(L_inst);

    const int L_prefill = ap_uint<9>(L_inst(8, 0)).to_int();
    const int L = (L_inst[9] == 1) ? 1 : L_prefill; 

    for(int g = 0; g < NUM_GROUPS; g++) {
        for (int r = 0; r < HEAD_PER_GROUP; r++) {
            for (int i = 0; i < L; i++) {
                #pragma HLS dataflow
                #pragma HLS loop_tripcount min=32 max=128

                float softmax_buf[MAX_SEQ_LEN];
                #pragma HLS array_partition variable=softmax_buf cyclic factor=16 dim=1
                float sum = 0.0f;

                exp_sum: for (int j = 0; j < (L_prefill >> 4); j++) {
                    #pragma HLS pipeline II=1
                    tapa::vec_t<float, 16> pre_softmax_vec = pre_softmax_fifo.read();
                    float exp_buf[16];
                    #pragma HLS array_partition variable=exp_buf complete

                    for (int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        int col = j * 16 + k;
                        float exp_val = std::exp((float)pre_softmax_vec[k] * (float)0.0883883476);
                        if (i < col && L_inst[9] != 1) exp_val = 0.0f; // zero out upper triangular part
                        softmax_buf[col] = exp_val;
                        exp_buf[k] = exp_val;
                    }

                    //binary reduction
                    for (int k = 0; k < 8; k++) {
                        #pragma HLS unroll
                        exp_buf[k] += exp_buf[k + 8];
                    }

                    for (int k = 0; k < 4; k++) {
                        #pragma HLS unroll
                        exp_buf[k] += exp_buf[k + 4];
                    }

                    for (int k = 0; k < 2; k++) {
                        #pragma HLS unroll
                        exp_buf[k] += exp_buf[k + 2];
                    }

                    sum += exp_buf[0] + exp_buf[1];
                    
                }

                sum = 1.0 / sum; // compute the inverse of the sum

                send_exp: for (int j = 0; j < (L_prefill >> 4); j++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS loop_tripcount min=2 max=8
                    tapa::vec_t<float, 16> post_softmax_vec;
                    for (int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        post_softmax_vec[k] = softmax_buf[j*16+k] * sum;
                    }
                    post_softmax_fifo.write(post_softmax_vec);
                }
            }
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


//top function for testing
// void gqa(
//     const int L,
//     tapa::mmap<tapa::vec_t<float, 16>> input_buffer,
//     tapa::mmap<tapa::vec_t<float, 16>> out_buffer,
//     tapa::mmap<int> cycle_count
// ) {
//     tapa::stream<tapa::vec_t<float, 16>> input_fifo("input_fifo");
//     tapa::stream<tapa::vec_t<float, 16>> input_fifo_qk("input_fifo_qk");
//     tapa::stream<tapa::vec_t<float, 16>> input_fifo_av("input_fifo_av");
//     tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");
//     tapa::stream<tapa::vec_t<float, 16>> pre_softmax_fifo("pre_softmax_fifo");
//     tapa::stream<tapa::vec_t<float, 16>> post_softmax_fifo("post_softmax_fifo");
//     tapa::stream<bool> fifo_fin("fifo_fin");

//     tapa::task()
//         .invoke<tapa::join>(gqa_input_reader, L, input_buffer, input_fifo)
//         .invoke<tapa::join>(gqa_arbiter, L, input_fifo, input_fifo_qk, input_fifo_av)
//         .invoke<tapa::join>(gemm_gqa_qk, L, input_fifo_qk, pre_softmax_fifo)
//         .invoke<tapa::join>(softmax, L, pre_softmax_fifo, post_softmax_fifo)
//         .invoke<tapa::join>(gemm_gqa_av, L, input_fifo_av, post_softmax_fifo, out_fifo)
//         .invoke<tapa::join>(out_writer, L, out_fifo, out_buffer, fifo_fin)
//         .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
// }

#endif