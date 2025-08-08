#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include "attention_block.h"

typedef ap_uint<8> idx_t;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// Helper function to compute Chebyshev distance (L-infinity norm)
float chebyshev_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

// Reference implementation for finding closest centroid
int find_closest_centroid(const std::vector<float>& point, 
                         const std::vector<std::vector<float>>& centroids) {
    int closest_idx = 0;
    float min_distance = chebyshev_distance(point, centroids[0]);
    
    for (size_t i = 1; i < centroids.size(); ++i) {
        float distance = chebyshev_distance(point, centroids[i]);
        if (distance < min_distance) {
            min_distance = distance;
            closest_idx = i;
        }
    }
    return closest_idx;
}

// Quantization helper functions
std::pair<float, float> compute_scale_zeropoint(const std::vector<std::vector<std::vector<std::vector<float>>>>& lut_2d,
                                                int in_size, int num_submatrices, int num_act_centroids, int num_weight_centroids) {
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    // Find min and max across all LUT values
    for (int pos = 0; pos < in_size; pos++) {
        for (int sub = 0; sub < num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    float val = lut_2d[pos][sub][act_idx][weight_idx];
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
            }
        }
    }
    
    float scale = (max_val - min_val) / 255.0f;
    float zeropoint = -min_val / scale;

    return {scale, zeropoint};
}

uint8_t quantize_value(float value, float scale, float zeropoint) {
    int quantized = std::round(value / scale + zeropoint);
    // Clamp to uint8 range
    quantized = std::max(0, std::min(255, quantized));
    return static_cast<uint8_t>(quantized);
}

// Reference rotary position embedding
void apply_rotary_pos_emb_ref(
    const std::vector<std::vector<float>>& input,
    std::vector<std::vector<float>>& output,
    const std::vector<std::vector<float>>& cos_table,
    const std::vector<std::vector<float>>& sin_table,
    int seq_len, int head_dim
) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float x = input[pos][i];
            float y = input[pos][i + head_dim / 2];
            float cos_val = cos_table[pos][i];
            float sin_val = sin_table[pos][i];
            
            output[pos][i] = x * cos_val - y * sin_val;
            output[pos][i + head_dim / 2] = x * sin_val + y * cos_val;
        }
    }
}

void reference_linear_quantized_lut(
    const std::vector<std::vector<std::vector<float>>>& input_vectors,  // in_size x L x vector_dim
    const std::vector<std::vector<std::vector<float>>>& act_centroids,  // in_size x num_act_centroids x vector_dim
    const std::vector<std::vector<std::vector<int>>>& weight_indices,  // in_size x (out_size/256) x 256
    const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& lut_2d_quantized,  // in_size x num_submatrices x num_act_centroids x num_weight_centroids
    float scale, float zeropoint,
    std::vector<std::vector<float>>& output,                             // L x out_size
    int L, int in_size, int out_size
) {
    int num_submatrices = (out_size + 511) / 512;

    // Initialize output
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size; j++) {
            output[i][j] = 0.0f;
        }
    }
    
    // For each sequence
    for (int i = 0; i < L; i++) {
        for (int pos = 0; pos < in_size; pos++) {
            // Find closest activation centroid for this position
            std::vector<float> input_vec = input_vectors[pos][i];
            int act_centroid_idx = find_closest_centroid(input_vec, act_centroids[pos]);
            
            // For each weight submatrix
            for (int sub = 0; sub < num_submatrices; sub++) {
                int sub_out_size = std::min(512, out_size - sub * 512);
                for (int j = 0; j < sub_out_size; j++) {
                    // Get weight centroid index for this output position
                    int weight_centroid_idx = weight_indices[pos][sub][j];
                    
                    // Get quantized LUT value and dequantize
                    uint8_t quantized_val = lut_2d_quantized[pos][sub][act_centroid_idx][weight_centroid_idx];
                    float lut_val = (float(quantized_val) - zeropoint) * scale;
                    
                    output[i][sub * 512 + j] += lut_val;
                }
            }
        }
    }
}

// Reference grouped query attention implementation with weight quantized LUT
void reference_attention_block(
    const std::vector<std::vector<float>>& input,  // L x HIDDEN_DIM
    const std::vector<std::vector<std::vector<float>>>& act_centroids,    // HIDDEN_DIM_DIV_2 x num_centroids x 2
    const std::vector<std::vector<std::vector<int>>>& qkv_weight_indices,  // HIDDEN_DIM_DIV_2 x (QKV_DIM/512) x 512
    const std::vector<std::vector<std::vector<int>>>& out_weight_indices,  // HIDDEN_DIM_DIV_2 x (HIDDEN_DIM/512) x 512
    const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& qkv_lut_2d_quantized,  // HIDDEN_DIM_DIV_2 x qkv_submatrices x num_act_centroids x num_weight_centroids
    const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& out_lut_2d_quantized,  // HIDDEN_DIM_DIV_2 x out_submatrices x num_act_centroids x num_weight_centroids
    const std::vector<float>& qkv_scales, const std::vector<float>& qkv_zeropoints,  // per head scale/zeropoint for QKV
    float out_scale, float out_zeropoint,  // scale/zeropoint for output projection
    const std::vector<std::vector<float>>& sin_table,  // L x HEAD_DIM
    const std::vector<std::vector<float>>& cos_table,  // L x HEAD_DIM
    std::vector<std::vector<float>>& output,       // L x HIDDEN_DIM
    int L
) {
    // Step 1: Compute QKV projection using weight quantized LUT
    // The order is: v[0], k[0], q[0:1], v[1], k[1], q[2:3], ..., v[7], k[7], q[14:15]
    std::vector<std::vector<float>> qkv_proj(L, std::vector<float>(QKV_DIM));
    
    // Reshape 2D input to 3D format expected by reference_linear_quantized_lut
    // input: L x HIDDEN_DIM → input_3d: HIDDEN_DIM_DIV_2 x L x 2
    std::vector<std::vector<std::vector<float>>> input_3d(HIDDEN_DIM_DIV_2, 
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < L; i++) {
            input_3d[pos][i][0] = input[i][pos * 2];
            input_3d[pos][i][1] = input[i][pos * 2 + 1];
        }
    }
    
    reference_linear_quantized_lut(
        input_3d, act_centroids, qkv_weight_indices, qkv_lut_2d_quantized,
        qkv_scales[0], qkv_zeropoints[0], qkv_proj, L, HIDDEN_DIM_DIV_2, QKV_DIM
    );
    
    // Step 2: Extract V, K, Q heads according to the arbiter pattern
    // v[0], k[0], q[0:1], v[1], k[1], q[2:3], ..., v[7], k[7], q[14:15]
    std::vector<std::vector<std::vector<float>>> v_heads(L, 
        std::vector<std::vector<float>>(NUM_GROUPS, std::vector<float>(HEAD_DIM)));
    std::vector<std::vector<std::vector<float>>> k_heads(L, 
        std::vector<std::vector<float>>(NUM_GROUPS, std::vector<float>(HEAD_DIM)));
    std::vector<std::vector<std::vector<float>>> q_heads(L, 
        std::vector<std::vector<float>>(NUM_HEADS, std::vector<float>(HEAD_DIM)));
    
    int head_idx = 0;
    for (int g = 0; g < NUM_GROUPS; g++) {
        // V head for group g
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                v_heads[i][g][d] = qkv_proj[i][head_idx * HEAD_DIM + d];
            }
        }
        head_idx++;
        
        // K head for group g
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                k_heads[i][g][d] = qkv_proj[i][head_idx * HEAD_DIM + d];
            }
        }
        head_idx++;
        
        // Q heads for group g (HEAD_PER_GROUP = 2)
        for (int h = 0; h < HEAD_PER_GROUP; h++) {
            int q_head_idx = g * HEAD_PER_GROUP + h;
            for (int i = 0; i < L; i++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    q_heads[i][q_head_idx][d] = qkv_proj[i][head_idx * HEAD_DIM + d];
                }
            }
            head_idx++;
        }
    }
    
    // Step 3: Apply RoPE to Q and K heads
    for (int h = 0; h < NUM_HEADS; h++) {
        std::vector<std::vector<float>> q_head_2d(L, std::vector<float>(HEAD_DIM));
        std::vector<std::vector<float>> q_head_2d_out(L, std::vector<float>(HEAD_DIM));
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                q_head_2d[i][d] = q_heads[i][h][d];
            }
        }
        apply_rotary_pos_emb_ref(q_head_2d, q_head_2d_out, cos_table, sin_table, L, HEAD_DIM);
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                q_heads[i][h][d] = q_head_2d_out[i][d];
            }
        }
    }
    
    for (int h = 0; h < NUM_GROUPS; h++) {
        std::vector<std::vector<float>> k_head_2d(L, std::vector<float>(HEAD_DIM));
        std::vector<std::vector<float>> k_head_2d_out(L, std::vector<float>(HEAD_DIM));
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                k_head_2d[i][d] = k_heads[i][h][d];
            }
        }
        apply_rotary_pos_emb_ref(k_head_2d, k_head_2d_out, cos_table, sin_table, L, HEAD_DIM);
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                k_heads[i][h][d] = k_head_2d_out[i][d];
            }
        }
    }
    
    // Step 4: Compute attention for each head group (GQA)
    std::vector<std::vector<std::vector<float>>> attn_output(L, 
        std::vector<std::vector<float>>(NUM_HEADS, std::vector<float>(HEAD_DIM)));
    
    for (int g = 0; g < NUM_GROUPS; g++) {
        // For each query head in this group
        for (int h = 0; h < HEAD_PER_GROUP; h++) {
            int q_head_idx = g * HEAD_PER_GROUP + h;
            
            // Compute attention scores: Q @ K^T
            std::vector<std::vector<float>> scores(L, std::vector<float>(L));
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        sum += q_heads[i][q_head_idx][d] * k_heads[j][g][d];
                    }
                    scores[i][j] = sum * 0.125f;  // Scale factor from hardware
                }
            }
            
            // Apply causal mask and softmax
            for (int i = 0; i < L; i++) {
                float sum = 0.0f;
                // Apply causal mask and compute exp
                for (int j = 0; j < L; j++) {
                    if (i >= j) {
                        scores[i][j] = std::exp(scores[i][j]);
                        sum += scores[i][j];
                    } else {
                        scores[i][j] = 0.0f;
                    }
                }
                // Normalize
                for (int j = 0; j < L; j++) {
                    if (i >= j && sum > 0.0f) {
                        scores[i][j] /= sum;
                    }
                }
            }
            
            // Compute attention output: attention_weights @ V
            for (int i = 0; i < L; i++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < L; j++) {
                        sum += scores[i][j] * v_heads[j][g][d];
                    }
                    attn_output[i][q_head_idx][d] = sum;
                }
            }
        }
    }
    
    // Step 5: Concatenate attention outputs and reshape
    std::vector<std::vector<float>> attn_concat(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                attn_concat[i][h * HEAD_DIM + d] = attn_output[i][h][d];
            }
        }
    }
    
    // Step 6: Apply output projection using weight quantized LUT
    // Reshape 2D attn_concat to 3D format expected by reference_linear_quantized_lut
    // attn_concat: L x HIDDEN_DIM → attn_concat_3d: HIDDEN_DIM_DIV_2 x L x 2
    std::vector<std::vector<std::vector<float>>> attn_concat_3d(HIDDEN_DIM_DIV_2, 
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < L; i++) {
            attn_concat_3d[pos][i][0] = attn_concat[i][pos * 2];
            attn_concat_3d[pos][i][1] = attn_concat[i][pos * 2 + 1];
        }
    }
    
    reference_linear_quantized_lut(
        attn_concat_3d, act_centroids, out_weight_indices, out_lut_2d_quantized,
        out_scale, out_zeropoint, output, L, HIDDEN_DIM_DIV_2, HIDDEN_DIM
    );
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters
    const int L = 32;              // Sequence length
    const int num_act_centroids = 64;   // Number of activation centroids per position
    const int num_weight_centroids = 16; // Number of weight centroids per position  
    const int vector_dim = 2;       // Dimension of each centroid
    
    std::cout << "Testing Complete Attention Block with:" << std::endl;
    std::cout << "  L (sequence length): " << L << std::endl;
    std::cout << "  Hidden dimension: " << HIDDEN_DIM << std::endl;
    std::cout << "  QKV dimension: " << QKV_DIM << std::endl;
    std::cout << "  Total heads: " << TOTAL_HEADS << std::endl;
    std::cout << "  Number of groups: " << NUM_GROUPS << std::endl;
    std::cout << "  Heads per group: " << HEAD_PER_GROUP << std::endl;
    std::cout << "  Head dimension: " << HEAD_DIM << std::endl;
    std::cout << "  Number of activation centroids per position: " << num_act_centroids << std::endl;
    std::cout << "  Number of weight centroids per position: " << num_weight_centroids << std::endl;
    std::cout << "  Vector dimension: " << vector_dim << std::endl;
    
    // Initialize random number generator
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> centroid_dis(-0.2f, 0.2f);
    std::uniform_real_distribution<float> input_dis(-0.2f, 0.2f);
    std::uniform_int_distribution<int> weight_idx_dis(0, num_weight_centroids - 1);
    
    // Generate random input
    std::vector<std::vector<float>> input(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            input[i][j] = input_dis(gen);
        }
    }
    
    // Pack input into hardware format for 2 channels (follows CCU input reader pattern)
    // Channel 0: first half, Channel 1: second half
    std::vector<std::vector<tapa::vec_t<float, 16>>> input_hw(2);
    input_hw[0].resize(L * HIDDEN_DIM / 32);  // First half
    input_hw[1].resize(L * HIDDEN_DIM / 32);  // Second half
    
    for (int i = 0; i < (HIDDEN_DIM / 32); i++) {
        for (int j = 0; j < L; j++) {
            int hw_idx = i * L + j;
            // First channel gets first half of dimensions
            for (int k = 0; k < 16; k++) {
                input_hw[0][hw_idx][k] = input[j][i * 32 + k];
            }
            // Second channel gets second half of dimensions 
            for (int k = 0; k < 16; k++) {
                input_hw[1][hw_idx][k] = input[j][i * 32 + 16 + k];
            }
        }
    }
    
    // Generate activation centroids (same for both QKV and output projections)
    std::vector<std::vector<std::vector<float>>> act_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(vector_dim)));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                act_centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }
    
    // Generate weight centroids and indices for QKV projection
    int qkv_submatrices = (QKV_DIM + 511) / 512;
    std::vector<std::vector<std::vector<std::vector<float>>>> qkv_weight_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(qkv_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> qkv_weight_indices(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<int>>(qkv_submatrices, std::vector<int>(512)));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < qkv_submatrices; sub++) {
            for (int wc = 0; wc < num_weight_centroids; wc++) {
                for (int d = 0; d < vector_dim; d++) {
                    qkv_weight_centroids[pos][sub][wc][d] = centroid_dis(gen);
                }
            }
            for (int idx = 0; idx < 512; idx++) {
                qkv_weight_indices[pos][sub][idx] = weight_idx_dis(gen);
            }
        }
    }
    
    // Generate weight centroids and indices for output projection
    int out_submatrices = (HIDDEN_DIM + 511) / 512;
    std::vector<std::vector<std::vector<std::vector<float>>>> out_weight_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(out_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> out_weight_indices(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<int>>(out_submatrices, std::vector<int>(512)));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < out_submatrices; sub++) {
            for (int wc = 0; wc < num_weight_centroids; wc++) {
                for (int d = 0; d < vector_dim; d++) {
                    out_weight_centroids[pos][sub][wc][d] = centroid_dis(gen);
                }
            }
            for (int idx = 0; idx < 512; idx++) {
                out_weight_indices[pos][sub][idx] = weight_idx_dis(gen);
            }
        }
    }
    
    // Generate 2D LUTs with dot products between activation and weight centroids
    std::vector<std::vector<std::vector<std::vector<float>>>> qkv_lut_2d(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(qkv_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    std::vector<std::vector<std::vector<std::vector<float>>>> out_lut_2d(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(out_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    // Compute QKV LUT values
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < qkv_submatrices; sub++) {
            for (int ac = 0; ac < num_act_centroids; ac++) {
                for (int wc = 0; wc < num_weight_centroids; wc++) {
                    float dot_product = 0.0f;
                    for (int d = 0; d < vector_dim; d++) {
                        dot_product += act_centroids[pos][ac][d] * qkv_weight_centroids[pos][sub][wc][d];
                    }
                    qkv_lut_2d[pos][sub][ac][wc] = dot_product;
                }
            }
        }
    }
    
    // Compute output LUT values
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < out_submatrices; sub++) {
            for (int ac = 0; ac < num_act_centroids; ac++) {
                for (int wc = 0; wc < num_weight_centroids; wc++) {
                    float dot_product = 0.0f;
                    for (int d = 0; d < vector_dim; d++) {
                        dot_product += act_centroids[pos][ac][d] * out_weight_centroids[pos][sub][wc][d];
                    }
                    out_lut_2d[pos][sub][ac][wc] = dot_product;
                }
            }
        }
    }
    
    // Quantize LUTs and compute scale/zeropoint for each head
    std::vector<float> qkv_scales(TOTAL_HEADS);
    std::vector<float> qkv_zeropoints(TOTAL_HEADS);
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> qkv_lut_2d_quantized(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<uint8_t>>>(qkv_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    // For simplicity, use same scale/zeropoint for all QKV heads
    auto [qkv_scale, qkv_zeropoint] = compute_scale_zeropoint(qkv_lut_2d, HIDDEN_DIM_DIV_2, qkv_submatrices, num_act_centroids, num_weight_centroids);
    for (int h = 0; h < TOTAL_HEADS; h++) {
        qkv_scales[h] = qkv_scale;
        qkv_zeropoints[h] = qkv_zeropoint;
    }
    
    // Quantize QKV LUT
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < qkv_submatrices; sub++) {
            for (int ac = 0; ac < num_act_centroids; ac++) {
                for (int wc = 0; wc < num_weight_centroids; wc++) {
                    qkv_lut_2d_quantized[pos][sub][ac][wc] = quantize_value(qkv_lut_2d[pos][sub][ac][wc], qkv_scale, qkv_zeropoint);
                }
            }
        }
    }
    
    // Quantize output LUT
    auto [out_scale, out_zeropoint] = compute_scale_zeropoint(out_lut_2d, HIDDEN_DIM_DIV_2, out_submatrices, num_act_centroids, num_weight_centroids);
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> out_lut_2d_quantized(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<uint8_t>>>(out_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < out_submatrices; sub++) {
            for (int ac = 0; ac < num_act_centroids; ac++) {
                for (int wc = 0; wc < num_weight_centroids; wc++) {
                    out_lut_2d_quantized[pos][sub][ac][wc] = quantize_value(out_lut_2d[pos][sub][ac][wc], out_scale, out_zeropoint);
                }
            }
        }
    }
    
    // Generate RoPE sin/cos tables
    std::cout << "Generating RoPE sin/cos tables..." << std::endl;
    const float theta = 1e6f;  // RoPE theta parameter
    std::vector<std::vector<float>> sin_table(L, std::vector<float>(HEAD_DIM));
    std::vector<std::vector<float>> cos_table(L, std::vector<float>(HEAD_DIM));
    
    // Generate frequency for each dimension pair
    std::vector<float> inv_freq(HEAD_DIM / 2);
    for (int i = 0; i < HEAD_DIM / 2; i++) {
        inv_freq[i] = 1.0f / std::pow(theta, 2.0f * i / HEAD_DIM);
    }
    
    // Generate sin and cos for each position
    for (int pos = 0; pos < L; pos++) {
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            float angle = pos * inv_freq[i];
            sin_table[pos][i] = std::sin(angle);
            cos_table[pos][i] = std::cos(angle);
            sin_table[pos][i + HEAD_DIM / 2] = std::sin(angle);
            cos_table[pos][i + HEAD_DIM / 2] = std::cos(angle);
        }
    }
    
    int total_centroid_positions = HIDDEN_DIM; // CENTROID_SIZE
    std::vector<std::vector<tapa::vec_t<float, 16>>> centroid_hw(2);
    centroid_hw[0].resize(total_centroid_positions * num_act_centroids / 16);  // First channel
    centroid_hw[1].resize(total_centroid_positions * num_act_centroids / 16);  // Second channel

    std::vector<tapa::vec_t<float, 16>> centroid_hw_tmp(total_centroid_positions * num_act_centroids / 8);
    
    // Copy up centroids first
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroid_hw_tmp[(pos/8)*num_act_centroids+i][(pos % 8)*vector_dim+j] = act_centroids[pos][i][j];
            }
        }
    }
    
    // Pack down centroids second
    int down_offset = (HIDDEN_DIM_DIV_2 / 8) * num_act_centroids;
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroid_hw_tmp[down_offset+(pos/8)*num_act_centroids+i][(pos % 8)*vector_dim+j] = act_centroids[pos][i][j];
            }
        }
    }

    for (int i = 0; i < centroid_hw_tmp.size(); i++) {
        centroid_hw[(i/num_act_centroids)%2][((i/num_act_centroids)/2)*num_act_centroids+(i%num_act_centroids)] = centroid_hw_tmp[i];
    }
    
    // Pack quantized LUT into hardware format (following FFN pattern)
    std::cout << "Packing quantized LUT into hardware format..." << std::endl;
    
    // Calculate total LUT size: QKV LUT + Output LUT
    int qkv_lut_size = HIDDEN_DIM_DIV_2 * qkv_submatrices * num_act_centroids * num_weight_centroids;
    int out_lut_size = HIDDEN_DIM_DIV_2 * out_submatrices * num_act_centroids * num_weight_centroids;
    int total_lut_size = qkv_lut_size + out_lut_size;
    
    std::cout << "  QKV LUT size: " << qkv_lut_size << std::endl;
    std::cout << "  Output LUT size: " << out_lut_size << std::endl;
    std::cout << "  Total LUT size: " << total_lut_size << std::endl;
    
    // Pack into 16 hardware buffers
    // Calculate total LUT vectors needed for hardware format
    int qkv_lut_vectors = (HIDDEN_DIM_DIV_2 / 16) * qkv_submatrices * (num_act_centroids / 4);
    int out_lut_vectors = (HIDDEN_DIM_DIV_2 / 16) * out_submatrices * (num_act_centroids / 4);
    int total_lut_vectors = qkv_lut_vectors + out_lut_vectors;
    
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> lut_hw(16);
    for (int buffer_idx = 0; buffer_idx < 16; buffer_idx++) {
        lut_hw[buffer_idx].resize(total_lut_vectors);
    }
    
    std::cout << "  QKV LUT vectors: " << qkv_lut_vectors << std::endl;
    std::cout << "  Output LUT vectors: " << out_lut_vectors << std::endl;
    std::cout << "  Total LUT vectors: " << total_lut_vectors << std::endl;
    
    int vector_offset = 0;
    
    // Pack QKV LUT first (following FFN pattern)
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < qkv_submatrices; sub++) {
            // Process groups of 4 activation centroids at a time (as expected by kernel)
            for (int act_group = 0; act_group < num_act_centroids / 4; act_group++) {
                int hw_idx = vector_offset + local_pos * qkv_submatrices * (num_act_centroids / 4) + act_group * qkv_submatrices + sub;
                
                // Pack 64 elements: 4 activation centroids x 16 weight centroids
                for (int k = 0; k < 16; k++) {  // 16 weight centroids
                    for (int ii = 0; ii < 4; ii++) {  // 4 activation centroids
                        int act_idx = act_group * 4 + ii;
                        if (act_idx < num_act_centroids && k < num_weight_centroids) {
                            int elem_idx = ii * 16 + k;  // Matches kernel: tmp[ii*16+k]
                            lut_hw[buffer_idx][hw_idx][elem_idx] = qkv_lut_2d_quantized[pos][sub][act_idx][k];
                        } else {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = 0;  // Padding
                        }
                    }
                }
            }
        }
    }
    
    // Update vector offset for output LUT
    vector_offset += qkv_lut_vectors;
    
    // Pack output LUT second (following FFN pattern)
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < out_submatrices; sub++) {
            // Process groups of 4 activation centroids at a time (as expected by kernel)
            for (int act_group = 0; act_group < num_act_centroids / 4; act_group++) {
                int hw_idx = vector_offset + local_pos * out_submatrices * (num_act_centroids / 4) + act_group * out_submatrices + sub;
                
                // Pack 64 elements: 4 activation centroids x 16 weight centroids
                for (int k = 0; k < 16; k++) {  // 16 weight centroids
                    for (int ii = 0; ii < 4; ii++) {  // 4 activation centroids
                        int act_idx = act_group * 4 + ii;
                        if (act_idx < num_act_centroids && k < num_weight_centroids) {
                            int elem_idx = ii * 16 + k;  // Matches kernel: tmp[ii*16+k]
                            lut_hw[buffer_idx][hw_idx][elem_idx] = out_lut_2d_quantized[pos][sub][act_idx][k];
                        } else {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = 0;  // Padding
                        }
                    }
                }
            }
        }
    }
    
    // Pack weight indices into hardware format
    std::cout << "Packing weight indices into hardware format..." << std::endl;
    
    // Calculate weight index vectors
    int qkv_weight_vectors = (HIDDEN_DIM_DIV_2 / 16) * (qkv_submatrices * 512 / 64 / 2);
    int out_weight_vectors = (HIDDEN_DIM_DIV_2 / 16) * (out_submatrices * 512 / 64 / 2);
    int total_weight_vectors = qkv_weight_vectors + out_weight_vectors;
    
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> weight_idx_hw(16);
    for (int buffer_idx = 0; buffer_idx < 16; buffer_idx++) {
        weight_idx_hw[buffer_idx].resize(total_weight_vectors);
    }
    
    vector_offset = 0;
    
    // Pack QKV weight indices first
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < qkv_submatrices; sub++) {
            for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                int hw_idx = vector_offset + local_pos * qkv_submatrices * 4 + sub * 4 + vec_idx;
                for (int k = 0; k < 64; k++) {
                    int col = vec_idx * 128 + k * 2;
                    if (col < 512) {
                        ap_uint<8> tmp_idx;
                        tmp_idx(3, 0) = qkv_weight_indices[pos][sub][col];
                        tmp_idx(7, 4) = qkv_weight_indices[pos][sub][col + 1];
                        weight_idx_hw[buffer_idx][hw_idx][k] = tmp_idx;
                    } else {
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0; // Padding
                    }
                }
            }
        }
    }
    
    // Update vector offset for output weight indices
    vector_offset += qkv_weight_vectors;
    
    // Pack output weight indices
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < out_submatrices; sub++) {
            for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                int hw_idx = vector_offset + local_pos * out_submatrices * 4 + sub * 4 + vec_idx;
                for (int k = 0; k < 64; k++) {
                    int col = vec_idx * 128 + k * 2;
                    if (col < 512) {
                        ap_uint<8> tmp_idx;
                        tmp_idx(3, 0) = out_weight_indices[pos][sub][col];
                        tmp_idx(7, 4) = out_weight_indices[pos][sub][col + 1];
                        weight_idx_hw[buffer_idx][hw_idx][k] = tmp_idx;
                    } else {
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0; // Padding
                    }
                }
            }
        }
    }
    
    // Combine LUT and weight indices following FFN pattern
    std::cout << "Combining LUT and weight indices..." << std::endl;
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> lut_weight_idx_hw(16);
    for (int buffer_idx = 0; buffer_idx < 16; buffer_idx++) {
        lut_weight_idx_hw[buffer_idx].resize(lut_hw[0].size() + weight_idx_hw[0].size());
    }
    
    // Calculate bounds for interleaving pattern (adapted for attention dimensions)
    const int round_0_lut_bound = (num_act_centroids >> 2) * (QKV_DIM >> 9);
    const int round_1_lut_bound = (num_act_centroids >> 2) * (HIDDEN_DIM >> 9);
    const int round_0_weight_bound = (QKV_DIM >> 7);
    const int round_1_weight_bound = (HIDDEN_DIM >> 7);
    const int round_0_bound = (HIDDEN_DIM_DIV_2 >> 4);
    const int round_1_bound = (HIDDEN_DIM_DIV_2 >> 4);  // Only one layer for attention
    
    for(int buffer_idx = 0; buffer_idx < 16; buffer_idx++) {
        int vec_idx = 0;
        // Round 0: QKV projection (input dim = HIDDEN_DIM_DIV_2, output dim = QKV_DIM)
        for(int r = 0; r < round_0_bound; r++){
            for(int i = 0; i < round_0_lut_bound; i++) {
                lut_weight_idx_hw[buffer_idx][vec_idx] = lut_hw[buffer_idx][i + r * round_0_lut_bound];
                vec_idx++;
            }
            for(int i = 0; i < round_0_weight_bound; i++) {
                lut_weight_idx_hw[buffer_idx][vec_idx] = weight_idx_hw[buffer_idx][i + r * round_0_weight_bound];
                vec_idx++;
            }
        }
        
        // Round 1: Output projection (input dim = HIDDEN_DIM_DIV_2, output dim = HIDDEN_DIM)
        for(int r = 0; r < round_1_bound; r++){
            for(int i = 0; i < round_1_lut_bound; i++) {
                lut_weight_idx_hw[buffer_idx][vec_idx] = lut_hw[buffer_idx][i + r * round_1_lut_bound + round_0_bound * round_0_lut_bound];
                vec_idx++;
            }
            for(int i = 0; i < round_1_weight_bound; i++) {
                lut_weight_idx_hw[buffer_idx][vec_idx] = weight_idx_hw[buffer_idx][i + r * round_1_weight_bound + round_0_bound * round_0_weight_bound];
                vec_idx++;
            }
        }
    }
    
    // Pack scale/zeropoint values
    std::vector<ap_uint<64>> scale_zero_hw(TOTAL_HEADS + 1);
    for (int h = 0; h < TOTAL_HEADS; h++) {
        float zero_hw = qkv_zeropoints[h] * qkv_scales[h] * HIDDEN_DIM_DIV_2;
        ap_uint<32> scale_bits = tapa::bit_cast<ap_uint<32>>(qkv_scales[h]);
        ap_uint<32> zero_bits = tapa::bit_cast<ap_uint<32>>(zero_hw);
        scale_zero_hw[h] = ap_uint<64>((zero_bits, scale_bits));
    }
    // Output projection scale/zeropoint
    float out_zero_hw = out_zeropoint * out_scale * HIDDEN_DIM_DIV_2;
    ap_uint<32> out_scale_bits = tapa::bit_cast<ap_uint<32>>(out_scale);
    ap_uint<32> out_zero_bits = tapa::bit_cast<ap_uint<32>>(out_zero_hw);
    scale_zero_hw[TOTAL_HEADS] = ap_uint<64>((out_zero_bits, out_scale_bits));
    
    // Pack RoPE tables into hardware format
    std::cout << "Packing RoPE tables into hardware format..." << std::endl;
    int rope_vectors = (L * HEAD_DIM) / 16;
    std::vector<tapa::vec_t<float, 16>> sin_hw(rope_vectors);
    std::vector<tapa::vec_t<float, 16>> cos_hw(rope_vectors);
    
    int vec_idx = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HEAD_DIM; j += 16) {
            for (int k = 0; k < 16; k++) {
                sin_hw[vec_idx][k] = sin_table[i][j + k];
                cos_hw[vec_idx][k] = cos_table[i][j + k];
            }
            vec_idx++;
        }
    }
    
    // Allocate output arrays
    int output_vectors = (L * HIDDEN_DIM) / 16;
    std::vector<tapa::vec_t<float, 16>> output_hw_raw(output_vectors);
    std::vector<int> cycle_count_hw(1);
    
    // Compute reference results
    std::cout << "Computing reference results..." << std::endl;
    std::vector<std::vector<float>> output_ref(L, std::vector<float>(HIDDEN_DIM));
    reference_attention_block(input, act_centroids, qkv_weight_indices, out_weight_indices, 
                             qkv_lut_2d_quantized, out_lut_2d_quantized,
                             qkv_scales, qkv_zeropoints, out_scale, out_zeropoint,
                             sin_table, cos_table, output_ref, L);
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(attention_block, FLAGS_bitstream,
                L,
                tapa::read_only_mmaps<tapa::vec_t<float, 16>, 2>(input_hw),
                tapa::read_only_mmaps<tapa::vec_t<float, 16>, 2>(centroid_hw),
                tapa::read_only_mmaps<tapa::vec_t<ap_uint<8>, 64>, 16>(lut_weight_idx_hw),
                tapa::read_only_mmap<ap_uint<64>>(scale_zero_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(sin_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(cos_hw),
                tapa::write_only_mmap<tapa::vec_t<float, 16>>(output_hw_raw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Convert hardware output from tapa::vec_t<float, 16> vectors to 2D array
    std::cout << "Converting hardware output..." << std::endl;
    std::vector<std::vector<float>> output_hw(L, std::vector<float>(HIDDEN_DIM));
    
    vec_idx = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < (HIDDEN_DIM / 16); j++) {
            for (int k = 0; k < 16; k++) {
                output_hw[i][j * 16 + k] = output_hw_raw[vec_idx][k];
            }
            vec_idx++;
        }
    }
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    float max_error = 0.0f;
    float tolerance = 1e-1f;  // Relaxed tolerance for quantized attention block
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float error = std::abs(output_hw[i][j] - output_ref[i][j]);
            max_error = std::max(max_error, error);
            if (error > tolerance) {
                errors++;
                if (errors <= 10) {  // Print first 10 errors
                    std::cout << "Error at [" << i << "][" << j << "]: HW=" 
                             << output_hw[i][j] << ", REF=" << output_ref[i][j] 
                             << ", diff=" << error << std::endl;
                }
            }
        }
    }
    
    std::cout << "Maximum error: " << max_error << std::endl;
    
    if (errors == 0) {
        std::cout << "SUCCESS: All " << (L * HIDDEN_DIM) 
                 << " results match reference within tolerance!" << std::endl;
    } else {
        std::cout << "NOTICE: " << errors << " out of " << (L * HIDDEN_DIM) 
                 << " results don't match reference within strict tolerance." << std::endl;
        std::cout << "This may be expected due to quantization and accumulated floating point errors." << std::endl;
    }
    
    // Print some sample results for debugging
    std::cout << "\nSample results (first sequence, first 10 outputs):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < std::min(10, HIDDEN_DIM); j++) {
        std::cout << "Output [0][" << j << "]: HW=" << output_hw[0][j] 
                 << ", REF=" << output_ref[0][j] 
                 << ", diff=" << std::abs(output_hw[0][j] - output_ref[0][j]) << std::endl;
    }
    
    
    // Test attention pattern (should be causal)
    std::cout << "\n=== Attention Pattern Analysis ===" << std::endl;
    // Simulate what the attention would look like using first group's K,V and first Q
    // This is conceptual since the hardware does full attention, but helps verify causality
    std::cout << "Note: Full attention verification requires complete hardware run" << std::endl;
    std::cout << "Hardware attention should show causal pattern where position i only attends to positions <= i" << std::endl;
    
    // Print statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total input elements: " << (L * HIDDEN_DIM) << std::endl;
    std::cout << "  Total activation centroids: " << (HIDDEN_DIM_DIV_2 * num_act_centroids) << std::endl;
    std::cout << "  Total weight centroids (QKV): " << (HIDDEN_DIM_DIV_2 * qkv_submatrices * num_weight_centroids) << std::endl;
    std::cout << "  Total weight centroids (Output): " << (HIDDEN_DIM_DIV_2 * out_submatrices * num_weight_centroids) << std::endl;
    std::cout << "  QKV LUT size: " << (HIDDEN_DIM_DIV_2 * qkv_submatrices * num_act_centroids * num_weight_centroids) << std::endl;
    std::cout << "  Output LUT size: " << (HIDDEN_DIM_DIV_2 * out_submatrices * num_act_centroids * num_weight_centroids) << std::endl;
    std::cout << "  Memory bandwidth:" << std::endl;
    std::cout << "    Input: " << (2 * HIDDEN_DIM_DIV_2 * L * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Centroids: " << (2 * HIDDEN_DIM_DIV_2 * 8 * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    LUT indices: " << (16 * ATTN_LUT_WEIGHT_SIZE * 64) << " bytes" << std::endl;
    std::cout << "    RoPE tables: " << (2 * rope_vectors * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Output: " << (output_vectors * 16 * sizeof(float)) << " bytes" << std::endl;
    
    return errors == 0 ? 0 : 1;
}
