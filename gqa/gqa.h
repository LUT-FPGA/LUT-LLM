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

constexpr int MAX_SEQ_LEN = 128;
constexpr int HIDDEN_DIM = 896;
constexpr int HEAD_DIM = 64;
constexpr int HEAD_PER_GROUP = 7;
constexpr int QKV_DIM = 896 + 64 * 4;


void input_reader( // k[0], v[0], q[0:6], k[1], v[1], q[7:13]
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

void gemm_gqa(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& pre_softmax_fifo,
    tapa::istream<tapa::vec_t<float, 16>>& post_softmax_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& out_fifo
) {
    #pragma HLS allocation function instances=pe_16x16 limit=1
    // compute grouped query attention
    
    for (int g = 0; g < 2; g++){ // groups
        
        // step 1: load k, v
        float k_buf[MAX_SEQ_LEN][HEAD_DIM];
        float v_buf[MAX_SEQ_LEN][HEAD_DIM];
        #pragma HLS array_partition variable=k_buf cyclic factor=16 dim=1
        #pragma HLS array_partition variable=v_buf cyclic factor=16 dim=2
        
        load_k: for (int i = 0; i < HEAD_DIM; i++) {
            for (int j = 0; j < (L >> 4); j++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> tmp = input_fifo.read(); 
                for (int k = 0; k < 16; k++) {
                    #pragma HLS unroll
                    k_buf[j*16+k][i] = tmp[k];
                }
            }
        }

        load_v: for (int i = 0; i < L; i++) {
            for (int j = 0; j < (HEAD_DIM >> 4); j++) {
                #pragma HLS pipeline II=1
                tapa::vec_t<float, 16> tmp = input_fifo.read(); 
                for (int k = 0; k < 16; k++) {
                    #pragma HLS unroll
                    v_buf[i][j*16+k] = tmp[k];
                }
            }
        }

        for (int r = 0; r < HEAD_PER_GROUP; r++) {
            
            // step 2: compute QK^T
            float qk[MAX_SEQ_LEN][MAX_SEQ_LEN];
            #pragma HLS array_partition variable=qk cyclic factor=16 dim=1
            #pragma HLS array_partition variable=qk cyclic factor=16 dim=2
            #pragma HLS bind_storage variable=qk type=RAM_2P impl=BRAM

            // initialize qk
            init_qk: for (int i = 0; i < (L >> 4); i++) {
                for (int j = 0; j < (L >> 4); j++) {
                    #pragma HLS pipeline II=1
                    for (int ii = 0; ii < 16; ii++) {
                        #pragma HLS unroll
                        for (int jj = 0; jj < 16; jj++) {
                            #pragma HLS unroll
                            qk[i*16+ii][j*16+jj] = 0.0f;
                        }
                    }
                }
            }

            // compute macc
            for (int k = 0; k < HEAD_DIM; k++) {
                for (int i = 0; i < (L >> 4); i++) {
                    auto q_vec = input_fifo.read();
                    
                    compute_qk: for (int j = 0; j < (L >> 4); j++) {
                        #pragma HLS pipeline II=1
                        float qk_reg[16][16];
                        #pragma HLS array_partition variable=qk_reg complete dim=1
                        #pragma HLS array_partition variable=qk_reg complete dim=2
                        // assign qk_reg
                        for (int ii = 0; ii < 16; ii++) {
                            #pragma HLS unroll
                            for (int jj = 0; jj < 16; jj++) {
                                #pragma HLS unroll
                                qk_reg[ii][jj] = qk[i*16+ii][j*16+jj];
                            }
                        }

                        tapa::vec_t<float, 16> k_vec;
                        for(int jj = 0; jj < 16; jj++) {
                            #pragma HLS unroll
                            k_vec[jj] = k_buf[j*16+jj][k];
                        }
                        pe_16x16(q_vec, k_vec, qk_reg);
                        for (int ii = 0; ii < 16; ii++) {
                            #pragma HLS unroll
                            for (int jj = 0; jj < 16; jj++) {
                                #pragma HLS unroll
                                qk[i*16+ii][j*16+jj] = qk_reg[ii][jj];
                            }
                        }

                    }
                }
            }

            // step 3: write batch of rows for softmax and compute AV
            for (int i = 0; i < (L >> 4); i++) {
                send_qk: for (int j = 0; j < L; j++){
                    #pragma HLS pipeline II=1
                    tapa::vec_t<float, 16> pre_softmax_qk;
                    for(int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        pre_softmax_qk[k] = qk[i*16+k][j];
                    }
                    pre_softmax_fifo.write(pre_softmax_qk);
                    for(int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        qk[i*16+k][j] = 0.0f; // reset qk for next round
                    }
                }

                if (L < HEAD_DIM) {
                    // If L is less than HEAD_DIM, we need to zero out the remaining elements
                    reset_qk: for (int j = L; j < HEAD_DIM; j++) {
                        #pragma HLS pipeline II=1
                        for(int k = 0; k < 16; k++) {
                            #pragma HLS unroll
                            qk[i*16+k][j] = 0.0f; // reset qk for next round
                        }
                    }
                }
                
                for (int j = 0; j < L; j++) {
                    #pragma HLS loop_tripcount min=16 max=128
                    auto qk_softmax_vec = post_softmax_fifo.read();
                    
                    compute_av: for (int k = 0; k < (HEAD_DIM >> 4); k++) {
                        #pragma HLS pipeline II=1
                        #pragma HLS loop_tripcount min=4 max=4

                        float av_reg[16][16];
                        #pragma HLS array_partition variable=av_reg complete dim=1
                        #pragma HLS array_partition variable=av_reg complete dim=2
                        
                        for (int ii = 0; ii < 16; ii++) {
                            #pragma HLS unroll
                            for (int jj = 0; jj < 16; jj++) {
                                #pragma HLS unroll
                                av_reg[ii][jj] = qk[i*16+ii][k*16+jj];
                            }
                        }
                        
                        tapa::vec_t<float, 16> v_vec;
                        for(int jj = 0; jj < 16; jj++) {
                            #pragma HLS unroll
                            v_vec[jj] = v_buf[j][k*16+jj];
                        }
                        
                        pe_16x16(qk_softmax_vec, v_vec, av_reg);
                        
                        for (int ii = 0; ii < 16; ii++) {
                            #pragma HLS unroll
                            for (int jj = 0; jj < 16; jj++) {
                                #pragma HLS unroll
                                qk[i*16+ii][k*16+jj] = av_reg[ii][jj];
                            }
                        }    
                    }
                }

                // write out results
                for (int ii = 0; ii < 16; ii++) {
                    tapa::vec_t<float, 16> out_vec;
                    write_av: for (int jj = 0; jj < (HEAD_DIM >> 4); jj++) {
                        #pragma HLS pipeline II=1
                        for (int kk = 0; kk < 16; kk++) {
                            #pragma HLS unroll
                            out_vec[kk] = qk[i*16+ii][jj*16+kk];
                        }
                        out_fifo.write(out_vec);
                    }
                }
            }
        }
    }
}

void softmax(
    const int L,
    tapa::istream<tapa::vec_t<float, 16>>& pre_softmax_fifo,
    tapa::ostream<tapa::vec_t<float, 16>>& post_softmax_fifo
) {
    for(int g = 0; g < 2; g++) {
        for (int r = 0; r < HEAD_PER_GROUP; r++) {
            for (int i = 0; i < (L >> 4); i++) {
                float softmax_buf[16][MAX_SEQ_LEN];
                #pragma HLS array_partition variable=softmax_buf complete dim=1
                float sum[16];
                #pragma HLS array_partition variable=sum complete

                // Initialize sum to 0
                for (int k = 0; k < 16; k++) {
                    #pragma HLS unroll
                    sum[k] = 0.0f;
                }

                exp_sum: for (int j = 0; j < L; j++) {
                    #pragma HLS pipeline II=1
                    tapa::vec_t<float, 16> pre_softmax_vec = pre_softmax_fifo.read();
                    
                    for (int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        int row = i * 16 + k;
                        float exp_val = std::exp((float)pre_softmax_vec[k] * (float)0.125);
                        if (row < j) exp_val = 0.0f; // zero out upper triangular part
                        softmax_buf[k][j] = exp_val;
                        sum[k] += exp_val;
                    }
                }

                for (int j = 0; j < 16; j++) {
                    #pragma HLS unroll
                    sum[j] = 1.0f / sum[j]; // compute reciprocal for normalization
                }

                for (int j = 0; j < L; j++) {
                    #pragma HLS pipeline II=1
                    tapa::vec_t<float, 16> post_softmax_vec;
                    for (int k = 0; k < 16; k++) {
                        #pragma HLS unroll
                        post_softmax_vec[k] = softmax_buf[k][j] * sum[k];
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
void gqa(
    const int L,
    tapa::mmap<tapa::vec_t<float, 16>> input_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 16>> input_fifo("input_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");
    tapa::stream<tapa::vec_t<float, 16>> pre_softmax_fifo("pre_softmax_fifo");
    tapa::stream<tapa::vec_t<float, 16>> post_softmax_fifo("post_softmax_fifo");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(input_reader, L, input_buffer, input_fifo)
        .invoke<tapa::join>(gemm_gqa, L, input_fifo, pre_softmax_fifo, post_softmax_fifo, out_fifo)
        .invoke<tapa::join>(softmax, L, pre_softmax_fifo, post_softmax_fifo)
        .invoke<tapa::join>(out_writer, L, out_fifo, out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);
}

#endif