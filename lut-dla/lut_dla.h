#ifndef _LUT_DLA_H_
#define _LUT_DLA_H_

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
#include "../imm/imm.h"
#include "../ccu/ccu_fp32.h"



void lut_dla_core(
    const int L,
    const int in_size,
    const int out_size,
    tapa::mmap<tapa::vec_t<float, 16>> inp,
    tapa::mmap<tapa::vec_t<float, 16>> centroid,
    tapa::mmaps<tapa::vec_t<ap_uint<8>, 64>, 8> lut_buffer,
    tapa::mmaps<tapa::vec_t<ap_uint<8>, 64>, 8> weight_idx_buffer,
    tapa::mmap<ap_uint<64>> scale_zero_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> linear_out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 16>> input_fifo("input_fifo");
    tapa::streams<tapa::vec_t<float, 2>, 8> input_split_fifo("input_split_fifo");
    tapa::streams<tapa::vec_t<float, 2>, 8> centroid_fifo("centroid_fifo");
    tapa::streams<ap_uint<8>, 8, 16> idx_fifo("idx_fifo");
    tapa::streams<tapa::vec_t<ap_uint<8>, 64>, 8> lut_fifo("lut_fifo");
    tapa::streams<tapa::vec_t<ap_uint<8>, 64>, 8> weight_idx_fifo("weight_idx_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 8> psum_0_fifo("psum_0_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 8> psum_1_fifo("psum_1_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 8> psum_2_fifo("psum_2_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 8> psum_3_fifo("psum_3_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 8> psum_4_fifo("psum_4_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 8> psum_5_fifo("psum_5_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 8> psum_6_fifo("psum_6_fifo");
    tapa::streams<tapa::vec_t<ap_uint<44>, 8>, 8> psum_7_fifo("psum_7_fifo");
    tapa::stream<ap_uint<64>> scale_zero_fifo("scale_zero_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(input_reader_wide, L, in_size, inp, input_fifo)
        .invoke<tapa::join, 8>(lut_reader, in_size, out_size, lut_buffer, lut_fifo)
        .invoke<tapa::join>(scale_zero_reader, scale_zero_buffer, scale_zero_fifo)
        .invoke<tapa::join, 8>(weight_idx_reader, in_size, out_size, weight_idx_buffer, weight_idx_fifo)
        .invoke<tapa::join>(input_splitter, L, in_size, input_fifo, input_split_fifo)
        .invoke<tapa::join>(centroid_reader_split, in_size, centroid, centroid_fifo)
        .invoke<tapa::join, 8>(ccu_fp32, L, in_size, input_split_fifo, centroid_fifo, idx_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq_head, L, in_size, out_size, idx_fifo, lut_fifo, weight_idx_fifo, psum_0_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, in_size, out_size, idx_fifo, lut_fifo, weight_idx_fifo, psum_0_fifo, psum_1_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, in_size, out_size, idx_fifo, lut_fifo, weight_idx_fifo, psum_1_fifo, psum_2_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, in_size, out_size, idx_fifo, lut_fifo, weight_idx_fifo, psum_2_fifo, psum_3_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, in_size, out_size, idx_fifo, lut_fifo, weight_idx_fifo, psum_3_fifo, psum_4_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, in_size, out_size, idx_fifo, lut_fifo, weight_idx_fifo, psum_4_fifo, psum_5_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, in_size, out_size, idx_fifo, lut_fifo, weight_idx_fifo, psum_5_fifo, psum_6_fifo)
        .invoke<tapa::join>(memory_matcher_w_vq, L, in_size, out_size, idx_fifo, lut_fifo, weight_idx_fifo, psum_6_fifo, psum_7_fifo)
        .invoke<tapa::join>(memory_matcher_tail_acc, L, in_size, out_size, psum_7_fifo, scale_zero_fifo, out_fifo)
        .invoke<tapa::join>(linear_out_writer, L, out_size, out_fifo, linear_out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);

}


#endif // _LUT_DLA_H_