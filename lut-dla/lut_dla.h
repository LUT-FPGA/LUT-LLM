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
    tapa::mmap<tapa::vec_t<float, 2>> inp,
    tapa::mmap<tapa::vec_t<float, 16>> centroid,
    tapa::mmap<tapa::vec_t<float, 16>> lut_buffer,
    tapa::mmap<tapa::vec_t<float, 16>> linear_out_buffer,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<tapa::vec_t<float, 2>> input_fifo("input_fifo");
    tapa::stream<tapa::vec_t<float, 16>> centroid_fifo("centroid_fifo");
    tapa::stream<ap_uint<8>, 16> idx_fifo("idx_fifo");
    tapa::stream<tapa::vec_t<ap_uint<64>, 8>> lut_fifo("lut_fifo");
    tapa::stream<tapa::vec_t<float, 16>> out_fifo("out_fifo");
    tapa::stream<bool> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(input_reader, L, in_size, inp, input_fifo)
        .invoke<tapa::join>(centroid_reader, in_size, centroid, centroid_fifo)
        .invoke<tapa::join>(lut_reader, in_size, out_size, lut_buffer, lut_fifo)
        .invoke<tapa::join>(ccu_fp32, L, in_size, input_fifo, centroid_fifo, idx_fifo)
        .invoke<tapa::join>(memory_matcher_test, L, in_size, out_size, idx_fifo, lut_fifo, out_fifo)
        .invoke<tapa::join>(linear_out_writer, L, out_size, out_fifo, linear_out_buffer, fifo_fin)
        .invoke<tapa::join>(measure_cycle, fifo_fin, cycle_count);

}


#endif // _LUT_DLA_H_