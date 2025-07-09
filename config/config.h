#ifndef CONFIG_H
#define CONFIG_H

constexpr int MAX_SEQ_LEN = 128;
constexpr int MAX_OUT_SIZE = 4864;
constexpr int MAX_OUT_SIZE_DIV_2 = MAX_OUT_SIZE >> 1;
constexpr int n_cent = 64;
constexpr int QK_DIM = 896 + 64 * 2;
constexpr int QK_DIM_DIV_2 = QK_DIM >> 1;
constexpr int V_DIM = 64 * 2;
constexpr int V_DIM_DIV_2 = V_DIM >> 1;
constexpr int HEAD_DIM = 64;
constexpr int HEAD_DIM_DIV_2 = HEAD_DIM >> 1;
constexpr int HIDDEN_DIM = 896;
constexpr int HIDDEN_DIM_DIV_2 = HIDDEN_DIM >> 1;
constexpr int HEAD_PER_GROUP = 7;
constexpr int QKV_DIM = 896 + 64 * 4;

#endif // CONFIG_H