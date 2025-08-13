#ifndef CONFIG_H
#define CONFIG_H

#include <tapa.h>
#include <ap_int.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <limits>

constexpr int MAX_SEQ_LEN = 128;
constexpr int MAX_SEQ_LEN_MUL_2 = MAX_SEQ_LEN * 2;
constexpr int MAX_OUT_SIZE = 6144 * 2;
constexpr int MAX_OUT_SIZE_DIV_2 = MAX_OUT_SIZE >> 1;
constexpr int MAX_OUT_SIZE_DIV_256 = MAX_OUT_SIZE / 256;
constexpr int MAX_OUT_SIZE_DIV_512 = MAX_OUT_SIZE / 512;
constexpr int n_cent = 64;
constexpr int w_n_cent = 16;
constexpr int NUM_GROUPS = 8;
constexpr int QK_DIM = 2048 + 128 * NUM_GROUPS;
constexpr int QK_DIM_DIV_2 = QK_DIM >> 1;
constexpr int V_DIM = 128 * NUM_GROUPS;
constexpr int V_DIM_DIV_2 = V_DIM >> 1;
constexpr int HEAD_DIM = 128;
constexpr int HEAD_DIM_DIV_2 = HEAD_DIM >> 1;
constexpr int HIDDEN_DIM = 2048;
constexpr int HIDDEN_DIM_DIV_2 = HIDDEN_DIM >> 1;
constexpr int HIDDEN_DIM_DIV_4 = HIDDEN_DIM >> 2;
constexpr int HEAD_PER_GROUP = 2;
constexpr int NUM_HEADS = NUM_GROUPS * HEAD_PER_GROUP;
constexpr int NUM_HEADS_MUL_2 = NUM_HEADS * 2;
constexpr int QKV_DIM = 2048 + 128 * NUM_GROUPS * 2;
constexpr int TOTAL_HEADS = QKV_DIM / HEAD_DIM;
constexpr float EPSILON = 1e-6f;
constexpr float R_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);
constexpr int FFN_DIM = 6144;
constexpr int INTERM_DIM = 6144;
constexpr int INTERM_DIM_MUL_2 = INTERM_DIM * 2;
constexpr int INTERM_DIM_DIV_2 = INTERM_DIM / 2;
constexpr int INTERM_DIM_DIV_16 = INTERM_DIM / 16;
constexpr int NUM_ROPE_HEADS = NUM_HEADS + NUM_GROUPS;

constexpr int FFN_LUT_SIZE = HIDDEN_DIM_DIV_2 * INTERM_DIM_MUL_2 + INTERM_DIM_DIV_2 * HIDDEN_DIM;
constexpr int FFN_LUT_WEIGHT_SIZE = (FFN_LUT_SIZE >> 9) + (FFN_LUT_SIZE >> 11);
constexpr int FFN_CENTROID_SIZE = HIDDEN_DIM_DIV_2 + INTERM_DIM_DIV_2;

constexpr int ATTN_LUT_SIZE = HIDDEN_DIM_DIV_2 * QKV_DIM + HIDDEN_DIM_DIV_2 * HIDDEN_DIM;
constexpr int ATTN_LUT_WEIGHT_SIZE = (ATTN_LUT_SIZE >> 9) + (ATTN_LUT_SIZE >> 11);
constexpr int ATTN_CENTROID_SIZE = HIDDEN_DIM;

constexpr int TOTAL_LUT_WEIGHT_SIZE = FFN_LUT_WEIGHT_SIZE + ATTN_LUT_WEIGHT_SIZE;
constexpr int TOTAL_CENTROID_SIZE = FFN_CENTROID_SIZE + ATTN_CENTROID_SIZE;

void repeater(
    const int L,
    tapa::istream<tapa::vec_t<float, 2>>& input_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& qk_in_fifo,
    tapa::ostream<tapa::vec_t<float, 2>>& v_in_fifo
) {
    for(int i = 0; i < HIDDEN_DIM_DIV_2; i++) {
        for (int j = 0; j < L;) {
            #pragma HLS pipeline II=1
            if (!input_fifo.empty()) {
                tapa::vec_t<float, 2> tmp; input_fifo.try_read(tmp);
                qk_in_fifo.write(tmp);
                v_in_fifo.write(tmp);
                j++;
            }
        }
    }
}

#endif // CONFIG_H