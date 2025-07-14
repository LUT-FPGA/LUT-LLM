#ifndef _CCU_FP32_H_
#define _CCU_FP32_H_

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

template<int vec_len = 2, int log_vec_len = 1, int idx = 0>
void distance_pe(
    hls::stream<tapa::vec_t<float, vec_len>>& inp,
    hls::stream<tapa::vec_t<float, vec_len>>& centroid,
    hls::stream<float>& d_in,
    hls::stream<ap_uint<8>>& idx_in,
    hls::stream<float>& d_out,
    hls::stream<ap_uint<8>>& idx_out,
    hls::stream<tapa::vec_t<float, vec_len>>& carry,
    const int L
){
    // compute distance in parallel
    float diff[vec_len];
    #pragma HLS array_partition variable=diff complete
    auto centroid_vec = centroid.read();

    for (int r = 0; r < L; r++) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=1 max=100

        auto input_vec = inp.read();
        carry.write(input_vec);
        auto d_best = d_in.read();
        auto idx_best = idx_in.read();
        comp_dist: for (int i = 0; i < vec_len; i++) {
            #pragma HLS unroll
            auto diff_real = input_vec[i] - centroid_vec[i];
            #pragma HLS bind_op variable=diff_real op=fsub impl=primitivedsp
            // Set sign bit to zero (take absolute value)
            uint32_t diff_real_bits = *reinterpret_cast<uint32_t*>(&diff_real);
            diff_real_bits &= 0x7FFFFFFF;  // Clear the sign bit
            diff[i] = *reinterpret_cast<float*>(&diff_real_bits);
        }
        // norm type: chebyshev
        // tree reduction, select the largest value from diff[]
        reduction:for (int i = 0; i < log_vec_len; i++) {
            for (int j = 0; j < vec_len; j+=((1 << i)+1)) {
                #pragma HLS unroll
                if (diff[j] < diff[j + (1 << i)]) {
                    diff[j] = diff[j + (1 << i)];
                }
            }
        }
        // check distance with previous centroid
        if (diff[0] < d_best) {
            d_best = diff[0];
            idx_best = ap_uint<8>(idx);
        }
        d_out.write(d_best);
        idx_out.write(idx_best);
    }
}

void input_splitter(
    const int L,
    const int in_size,
    tapa::istream<tapa::vec_t<float, 16>>& input_fifo,
    tapa::ostreams<tapa::vec_t<float, 2>, 8>& output_fifo
) {
    for(int i = 0; i < (L * in_size >> 3); i++){
        #pragma HLS pipeline II=1
        auto input_vec = input_fifo.read();
        for (int j = 0; j < 8; j++) {
            #pragma HLS unroll
            tapa::vec_t<float, 2> tmp;
            for (int k = 0; k < 2; k++) {
                #pragma HLS unroll
                tmp[k] = input_vec[j * 2 + k];
            }
            output_fifo[j].write(tmp);
        }
    }
}

void ccu_fp32(
    const int L, // sequence length
    const int in_size, // number of 2-element positions
    tapa::istream<tapa::vec_t<float, 2>>& inp,
    tapa::istream<tapa::vec_t<float, 2>>& centroid,
    tapa::ostream<ap_uint<8>>& idx_out
) {

    for(int r = 0; r < in_size; r++){
        #pragma HLS dataflow disable_start_propagation

        // Streams for carrying inp vectors between PEs
        hls::stream<tapa::vec_t<float, 2>> input_carry[65];
        #pragma HLS stream variable=input_carry depth=2
        #pragma HLS BIND_STORAGE variable=input_carry type=fifo impl=srl
        
        // Streams for carrying distances between PEs
        hls::stream<float> distance_carry[65];
        #pragma HLS stream variable=distance_carry depth=2
        #pragma HLS BIND_STORAGE variable=distance_carry type=fifo impl=srl
        
        // Streams for carrying indices between PEs
        hls::stream<ap_uint<8>> index_carry[65];
        #pragma HLS stream variable=index_carry depth=2
        #pragma HLS BIND_STORAGE variable=index_carry type=fifo impl=srl
        
        // Streams for centroid data to each PE
        hls::stream<tapa::vec_t<float, 2>> centroid_streams[64];
        #pragma HLS stream variable=centroid_streams depth=2
        #pragma HLS BIND_STORAGE variable=centroid_streams type=fifo impl=srl
        
        // Read centroid data and distribute to PEs
        // Each centroid read provides 16 floats, each PE needs 2 floats
        // So we get 8 centroids per read, need 8 reads for 64 centroids
        read_centroid: for (int c = 0; c < 64; c++) {
            #pragma HLS pipeline II=1
            auto centroid_vec = centroid.read();
            centroid_streams[c].write(centroid_vec);
        }

        // Read inp vector and initialize the chain
        loop_fill_inp: for(int i = 0; i < L; i++){
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=1 max=100
            auto input_vec = inp.read();
            input_carry[0].write(input_vec);
            distance_carry[0].write(3.0e20f);
            index_carry[0].write(ap_uint<8>(0));
        }
        
        // Instantiate 64 distance_pe units
        distance_pe<2, 1, 0>(input_carry[0], centroid_streams[0], distance_carry[0], index_carry[0], distance_carry[1], index_carry[1], input_carry[1], L);
        distance_pe<2, 1, 1>(input_carry[1], centroid_streams[1], distance_carry[1], index_carry[1], distance_carry[2], index_carry[2], input_carry[2], L);
        distance_pe<2, 1, 2>(input_carry[2], centroid_streams[2], distance_carry[2], index_carry[2], distance_carry[3], index_carry[3], input_carry[3], L);
        distance_pe<2, 1, 3>(input_carry[3], centroid_streams[3], distance_carry[3], index_carry[3], distance_carry[4], index_carry[4], input_carry[4], L);
        distance_pe<2, 1, 4>(input_carry[4], centroid_streams[4], distance_carry[4], index_carry[4], distance_carry[5], index_carry[5], input_carry[5], L);
        distance_pe<2, 1, 5>(input_carry[5], centroid_streams[5], distance_carry[5], index_carry[5], distance_carry[6], index_carry[6], input_carry[6], L);
        distance_pe<2, 1, 6>(input_carry[6], centroid_streams[6], distance_carry[6], index_carry[6], distance_carry[7], index_carry[7], input_carry[7], L);
        distance_pe<2, 1, 7>(input_carry[7], centroid_streams[7], distance_carry[7], index_carry[7], distance_carry[8], index_carry[8], input_carry[8], L);
        distance_pe<2, 1, 8>(input_carry[8], centroid_streams[8], distance_carry[8], index_carry[8], distance_carry[9], index_carry[9], input_carry[9], L);
        distance_pe<2, 1, 9>(input_carry[9], centroid_streams[9], distance_carry[9], index_carry[9], distance_carry[10], index_carry[10], input_carry[10], L);
        distance_pe<2, 1, 10>(input_carry[10], centroid_streams[10], distance_carry[10], index_carry[10], distance_carry[11], index_carry[11], input_carry[11], L);
        distance_pe<2, 1, 11>(input_carry[11], centroid_streams[11], distance_carry[11], index_carry[11], distance_carry[12], index_carry[12], input_carry[12], L);
        distance_pe<2, 1, 12>(input_carry[12], centroid_streams[12], distance_carry[12], index_carry[12], distance_carry[13], index_carry[13], input_carry[13], L);
        distance_pe<2, 1, 13>(input_carry[13], centroid_streams[13], distance_carry[13], index_carry[13], distance_carry[14], index_carry[14], input_carry[14], L);
        distance_pe<2, 1, 14>(input_carry[14], centroid_streams[14], distance_carry[14], index_carry[14], distance_carry[15], index_carry[15], input_carry[15], L);
        distance_pe<2, 1, 15>(input_carry[15], centroid_streams[15], distance_carry[15], index_carry[15], distance_carry[16], index_carry[16], input_carry[16], L);
        distance_pe<2, 1, 16>(input_carry[16], centroid_streams[16], distance_carry[16], index_carry[16], distance_carry[17], index_carry[17], input_carry[17], L);
        distance_pe<2, 1, 17>(input_carry[17], centroid_streams[17], distance_carry[17], index_carry[17], distance_carry[18], index_carry[18], input_carry[18], L);
        distance_pe<2, 1, 18>(input_carry[18], centroid_streams[18], distance_carry[18], index_carry[18], distance_carry[19], index_carry[19], input_carry[19], L);
        distance_pe<2, 1, 19>(input_carry[19], centroid_streams[19], distance_carry[19], index_carry[19], distance_carry[20], index_carry[20], input_carry[20], L);
        distance_pe<2, 1, 20>(input_carry[20], centroid_streams[20], distance_carry[20], index_carry[20], distance_carry[21], index_carry[21], input_carry[21], L);
        distance_pe<2, 1, 21>(input_carry[21], centroid_streams[21], distance_carry[21], index_carry[21], distance_carry[22], index_carry[22], input_carry[22], L);
        distance_pe<2, 1, 22>(input_carry[22], centroid_streams[22], distance_carry[22], index_carry[22], distance_carry[23], index_carry[23], input_carry[23], L);
        distance_pe<2, 1, 23>(input_carry[23], centroid_streams[23], distance_carry[23], index_carry[23], distance_carry[24], index_carry[24], input_carry[24], L);
        distance_pe<2, 1, 24>(input_carry[24], centroid_streams[24], distance_carry[24], index_carry[24], distance_carry[25], index_carry[25], input_carry[25], L);
        distance_pe<2, 1, 25>(input_carry[25], centroid_streams[25], distance_carry[25], index_carry[25], distance_carry[26], index_carry[26], input_carry[26], L);
        distance_pe<2, 1, 26>(input_carry[26], centroid_streams[26], distance_carry[26], index_carry[26], distance_carry[27], index_carry[27], input_carry[27], L);
        distance_pe<2, 1, 27>(input_carry[27], centroid_streams[27], distance_carry[27], index_carry[27], distance_carry[28], index_carry[28], input_carry[28], L);
        distance_pe<2, 1, 28>(input_carry[28], centroid_streams[28], distance_carry[28], index_carry[28], distance_carry[29], index_carry[29], input_carry[29], L);
        distance_pe<2, 1, 29>(input_carry[29], centroid_streams[29], distance_carry[29], index_carry[29], distance_carry[30], index_carry[30], input_carry[30], L);
        distance_pe<2, 1, 30>(input_carry[30], centroid_streams[30], distance_carry[30], index_carry[30], distance_carry[31], index_carry[31], input_carry[31], L);
        distance_pe<2, 1, 31>(input_carry[31], centroid_streams[31], distance_carry[31], index_carry[31], distance_carry[32], index_carry[32], input_carry[32], L);
        distance_pe<2, 1, 32>(input_carry[32], centroid_streams[32], distance_carry[32], index_carry[32], distance_carry[33], index_carry[33], input_carry[33], L);
        distance_pe<2, 1, 33>(input_carry[33], centroid_streams[33], distance_carry[33], index_carry[33], distance_carry[34], index_carry[34], input_carry[34], L);
        distance_pe<2, 1, 34>(input_carry[34], centroid_streams[34], distance_carry[34], index_carry[34], distance_carry[35], index_carry[35], input_carry[35], L);
        distance_pe<2, 1, 35>(input_carry[35], centroid_streams[35], distance_carry[35], index_carry[35], distance_carry[36], index_carry[36], input_carry[36], L);
        distance_pe<2, 1, 36>(input_carry[36], centroid_streams[36], distance_carry[36], index_carry[36], distance_carry[37], index_carry[37], input_carry[37], L);
        distance_pe<2, 1, 37>(input_carry[37], centroid_streams[37], distance_carry[37], index_carry[37], distance_carry[38], index_carry[38], input_carry[38], L);
        distance_pe<2, 1, 38>(input_carry[38], centroid_streams[38], distance_carry[38], index_carry[38], distance_carry[39], index_carry[39], input_carry[39], L);
        distance_pe<2, 1, 39>(input_carry[39], centroid_streams[39], distance_carry[39], index_carry[39], distance_carry[40], index_carry[40], input_carry[40], L);
        distance_pe<2, 1, 40>(input_carry[40], centroid_streams[40], distance_carry[40], index_carry[40], distance_carry[41], index_carry[41], input_carry[41], L);
        distance_pe<2, 1, 41>(input_carry[41], centroid_streams[41], distance_carry[41], index_carry[41], distance_carry[42], index_carry[42], input_carry[42], L);
        distance_pe<2, 1, 42>(input_carry[42], centroid_streams[42], distance_carry[42], index_carry[42], distance_carry[43], index_carry[43], input_carry[43], L);
        distance_pe<2, 1, 43>(input_carry[43], centroid_streams[43], distance_carry[43], index_carry[43], distance_carry[44], index_carry[44], input_carry[44], L);
        distance_pe<2, 1, 44>(input_carry[44], centroid_streams[44], distance_carry[44], index_carry[44], distance_carry[45], index_carry[45], input_carry[45], L);
        distance_pe<2, 1, 45>(input_carry[45], centroid_streams[45], distance_carry[45], index_carry[45], distance_carry[46], index_carry[46], input_carry[46], L);
        distance_pe<2, 1, 46>(input_carry[46], centroid_streams[46], distance_carry[46], index_carry[46], distance_carry[47], index_carry[47], input_carry[47], L);
        distance_pe<2, 1, 47>(input_carry[47], centroid_streams[47], distance_carry[47], index_carry[47], distance_carry[48], index_carry[48], input_carry[48], L);
        distance_pe<2, 1, 48>(input_carry[48], centroid_streams[48], distance_carry[48], index_carry[48], distance_carry[49], index_carry[49], input_carry[49], L);
        distance_pe<2, 1, 49>(input_carry[49], centroid_streams[49], distance_carry[49], index_carry[49], distance_carry[50], index_carry[50], input_carry[50], L);
        distance_pe<2, 1, 50>(input_carry[50], centroid_streams[50], distance_carry[50], index_carry[50], distance_carry[51], index_carry[51], input_carry[51], L);
        distance_pe<2, 1, 51>(input_carry[51], centroid_streams[51], distance_carry[51], index_carry[51], distance_carry[52], index_carry[52], input_carry[52], L);
        distance_pe<2, 1, 52>(input_carry[52], centroid_streams[52], distance_carry[52], index_carry[52], distance_carry[53], index_carry[53], input_carry[53], L);
        distance_pe<2, 1, 53>(input_carry[53], centroid_streams[53], distance_carry[53], index_carry[53], distance_carry[54], index_carry[54], input_carry[54], L);
        distance_pe<2, 1, 54>(input_carry[54], centroid_streams[54], distance_carry[54], index_carry[54], distance_carry[55], index_carry[55], input_carry[55], L);
        distance_pe<2, 1, 55>(input_carry[55], centroid_streams[55], distance_carry[55], index_carry[55], distance_carry[56], index_carry[56], input_carry[56], L);
        distance_pe<2, 1, 56>(input_carry[56], centroid_streams[56], distance_carry[56], index_carry[56], distance_carry[57], index_carry[57], input_carry[57], L);
        distance_pe<2, 1, 57>(input_carry[57], centroid_streams[57], distance_carry[57], index_carry[57], distance_carry[58], index_carry[58], input_carry[58], L);
        distance_pe<2, 1, 58>(input_carry[58], centroid_streams[58], distance_carry[58], index_carry[58], distance_carry[59], index_carry[59], input_carry[59], L);
        distance_pe<2, 1, 59>(input_carry[59], centroid_streams[59], distance_carry[59], index_carry[59], distance_carry[60], index_carry[60], input_carry[60], L);
        distance_pe<2, 1, 60>(input_carry[60], centroid_streams[60], distance_carry[60], index_carry[60], distance_carry[61], index_carry[61], input_carry[61], L);
        distance_pe<2, 1, 61>(input_carry[61], centroid_streams[61], distance_carry[61], index_carry[61], distance_carry[62], index_carry[62], input_carry[62], L);
        distance_pe<2, 1, 62>(input_carry[62], centroid_streams[62], distance_carry[62], index_carry[62], distance_carry[63], index_carry[63], input_carry[63], L);
        distance_pe<2, 1, 63>(input_carry[63], centroid_streams[63], distance_carry[63], index_carry[63], distance_carry[64], index_carry[64], input_carry[64], L);
        
        // Output the final result
        epilogue: for(int i = 0; i < L; i++){
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=1 max=100
            distance_carry[64].read();
            input_carry[64].read();
            idx_out.write(index_carry[64].read());
        }
    }

}

void input_reader(
    const int L,
    const int in_size,
    tapa::async_mmap<tapa::vec_t<float, 2>>& inp,
    tapa::ostream<tapa::vec_t<float, 2>>& input_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < (L * in_size);){
        #pragma HLS pipeline II=1
		if((i_req < (L * in_size)) & !inp.read_addr.full()){
            inp.read_addr.try_write(i_req);
            ++i_req;
		}
		if(!inp.read_data.empty()){
            tapa::vec_t<float, 2> tmp;
            inp.read_data.try_read(tmp);
            input_fifo.write(tmp);
            ++i_resp;
		}
	}
}

void input_reader_wide(
    const int L,
    const int in_size,
    tapa::async_mmap<tapa::vec_t<float, 16>>& inp,
    tapa::ostream<tapa::vec_t<float, 16>>& input_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < ((L * in_size) >> 4);){
        #pragma HLS pipeline II=1
		if((i_req < ((L * in_size) >> 4)) & !inp.read_addr.full()){
            inp.read_addr.try_write(i_req);
            ++i_req;
		}
		if(!inp.read_data.empty()){
            tapa::vec_t<float, 16> tmp;
            inp.read_data.try_read(tmp);
            input_fifo.write(tmp);
            ++i_resp;
		}
	}
}

void centroid_reader(
    const int in_size,
    tapa::async_mmap<tapa::vec_t<float, 16>>& centroid,
    tapa::ostream<tapa::vec_t<float, 16>>& centroid_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < (8 * in_size);){
        #pragma HLS pipeline II=1
        if((i_req < (8 * in_size)) & !centroid.read_addr.full()){
            centroid.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!centroid.read_data.empty()){
            tapa::vec_t<float, 16> tmp;
            centroid.read_data.try_read(tmp);
            centroid_fifo.write(tmp);
            ++i_resp;
        }
    }
}

void centroid_reader_split(
    const int in_size,
    tapa::async_mmap<tapa::vec_t<float, 16>>& centroid,
    tapa::ostreams<tapa::vec_t<float, 2>, 8>& centroid_fifo
) {
    for(int i_req = 0, i_resp = 0; i_resp < (8 * in_size);){
        #pragma HLS pipeline II=1
        if((i_req < (8 * in_size)) & !centroid.read_addr.full()){
            centroid.read_addr.try_write(i_req);
            ++i_req;
        }
        if(!centroid.read_data.empty()){
            tapa::vec_t<float, 16> tmp;
            centroid.read_data.try_read(tmp);
            for(int j = 0; j < 8; j++){
                #pragma HLS unroll
                tapa::vec_t<float, 2> tmp_sub;
                for(int k = 0; k < 2; k++){
                    #pragma HLS unroll
                    tmp_sub[k] = tmp[j * 2 + k];
                }
                centroid_fifo[j].write(tmp_sub);
            }
            ++i_resp;
        }
    }
}

void idx_out_writer(
    const int L,
    const int in_size,
    tapa::istream<ap_uint<8>>& idx_out_fifo,
    tapa::async_mmap<int>& idx_out
) {
    for(int i_req = 0, i_resp = 0; i_resp < (L * in_size);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (L * in_size)) & !idx_out_fifo.empty() & !idx_out.write_addr.full() & !idx_out.write_data.full()){
            idx_out.write_addr.try_write(i_req);
            ap_uint<8> tmp; idx_out_fifo.try_read(tmp);
            idx_out.write_data.try_write(tmp.to_int());
            ++i_req;
        }
        bool success = false;
        auto resp = idx_out.write_resp.read(success);
        if(success){
            i_resp += unsigned(resp)+1;
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

#endif

//top function for testing
void ccu_fp32_top(
    const int L,
    tapa::mmap<tapa::vec_t<float, 16>> inp,
    tapa::mmap<tapa::vec_t<float, 16>> centroid,
    tapa::mmaps<int, 8> idx_out
) {
    tapa::stream<tapa::vec_t<float, 16>> input_fifo;
    tapa::streams<tapa::vec_t<float, 2>, 8> input_split_fifo;
    tapa::streams<tapa::vec_t<float, 2>, 8> centroid_fifo;
    tapa::streams<ap_uint<8>, 8> idx_out_fifo;

    tapa::task()
        .invoke<tapa::join>(input_reader_wide, L, 16, inp, input_fifo)
        .invoke<tapa::join>(input_splitter, L, 8, input_fifo, input_split_fifo)
        .invoke<tapa::join>(centroid_reader_split, 8, centroid, centroid_fifo)
        .invoke<tapa::join, 8>(ccu_fp32, L, 1, input_split_fifo, centroid_fifo, idx_out_fifo)
        .invoke<tapa::join, 8>(idx_out_writer, L, 1, idx_out_fifo, idx_out);

}

#endif