#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include "qwen_block.h"

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
            closest_idx = static_cast<int>(i);
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

// Helper function to check if two floating point numbers are close
bool isClose(float a, float b, float tolerance = 1e-3) {
    return std::abs(a - b) < tolerance;
}

// Direct SiLU activation function
float silu_direct(float x) {
    return x / (1.0f + std::exp(-x));
}

// Piece-wise SiLU activation function (for testing against hardware)
float silu_piecewise(float x) {
    // This should match the hardware implementation's piece-wise approximation
    float slope = 0.0f;
    float intercept = 0.0f;
    // piecewise linear approximation of silu
    if (x < -8.000f) {
        slope = 0.0f;
        intercept = 0.0f;
    }
    else if (x < -4.000000f) {
        slope = -0.017316f;
        intercept = -0.141207f;
    }
    else if (x < -2.000000f) { // [-4.000000f, -2.000000f)
        slope = -0.083231f;
        intercept = -0.404867f;
    }
    else if (x < -1.000000f) { // [-2.000000f, -1.000000f)
        slope = -0.030536f;
        intercept = -0.299477f;
    }
    else if (x < 0.000000f) { // [-1.000000f, 0.000000f)
        slope = 0.268941f;
        intercept = 0.0f;
    }
    else if (x < 1.000000f) { // [0.000000f, 1.000000f)
        slope = 0.731059f;
        intercept = 0.0f;
    }
    else if (x < 2.000000f) { // [1.000000f, 2.000000f)
        slope = 1.030536f;
        intercept = -0.299477f;
    }
    else if (x < 4.000000f) { // [2.000000f, 4.000000f)
        slope = 1.083231f;
        intercept = -0.404867f;
    }
    else { // x >= 4.000000f
        slope = 1.0f;
        intercept = 0.0f;
    }
    return slope * x + intercept;
}

// Reference RMS normalization
void reference_rms_norm(
    const std::vector<std::vector<float>>& input,  // L x HIDDEN_DIM
    const std::vector<float>& weight,              // HIDDEN_DIM
    std::vector<std::vector<float>>& output,       // L x HIDDEN_DIM
    int L
) {
    const float epsilon = EPSILON;
    const float r_hidden_dim = R_HIDDEN_DIM;
    
    for (int i = 0; i < L; i++) {
        // Compute variance (mean square)
        float variance = 0.0f;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            variance += input[i][j] * input[i][j];
        }
        variance = variance * r_hidden_dim + epsilon;
        
        // Compute RMS normalization factor
        float rms_scale = 1.0f / std::sqrt(variance);
        
        // Apply normalization and weight scaling
        for (int j = 0; j < HIDDEN_DIM; j++) {
            output[i][j] = input[i][j] * rms_scale * weight[j];
        }
    }
}

// Reference rotary position embedding implementation
void apply_rotary_pos_emb_ref(
    const std::vector<std::vector<float>>& tensor,  // [L, head_dim]
    std::vector<std::vector<float>>& tensor_out,  // [L, head_dim]
    const std::vector<std::vector<float>>& cos,  // [L, head_dim]
    const std::vector<std::vector<float>>& sin,  // [L, head_dim]
    int L, int head_dim
) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < head_dim; j++) {
            float x = tensor[i][j];
            float sin_val = sin[i][j];
            float cos_val = cos[i][j];

            // Apply RoPE transformation: rotate_half operation
            float x_rotated;
            if (j < head_dim / 2) {
                // First half: -x[j + head_dim/2]
                x_rotated = -tensor[i][j + head_dim / 2];
            } else {
                // Second half: x[j - head_dim/2]
                x_rotated = tensor[i][j - head_dim / 2];
            }
            
            // RoPE formula: x * cos + rotate_half(x) * sin
            tensor_out[i][j] = x * cos_val + x_rotated * sin_val;
        }
    }
}

// Reference implementation for quantized linear layer (matches hardware behavior)
void reference_linear_quantized_lut(
    const std::vector<std::vector<std::vector<float>>>& input_vectors,  // in_size x L x vector_dim
    const std::vector<std::vector<std::vector<float>>>& act_centroids,  // in_size x num_act_centroids x vector_dim
    const std::vector<std::vector<std::vector<int>>>& weight_indices,  // in_size x (out_size/512) x 512
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

// Reference softmax implementation (matching hardware behavior)
void softmax_ref(std::vector<std::vector<float>>& scores, int L) {
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
}

// Reference grouped query attention implementation
void reference_grouped_query_attention(
    const std::vector<std::vector<std::vector<float>>>& q_heads,  // L x NUM_HEADS x HEAD_DIM
    const std::vector<std::vector<std::vector<float>>>& k_heads,  // L x NUM_GROUPS x HEAD_DIM
    const std::vector<std::vector<std::vector<float>>>& v_heads,  // L x NUM_GROUPS x HEAD_DIM
    std::vector<std::vector<float>>& output,                      // L x HIDDEN_DIM
    int L
) {
    
    // Initialize output
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            output[i][j] = 0.0f;
        }
    }
    
    // Process each query head
    for (int h = 0; h < NUM_HEADS; h++) {
        int group_idx = h / HEAD_PER_GROUP;  // Which group this head belongs to
        
        // Compute attention scores
        std::vector<std::vector<float>> scores(L, std::vector<float>(L, 0.0f));
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                float dot_product = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot_product += q_heads[i][h][d] * k_heads[j][group_idx][d];
                }
                scores[i][j] = dot_product * 0.0883883476f;  // Scale factor from hardware
            }
        }

        // Debug: Print 16x16 region of scores for h=0
        // if (h == 0) {
        //     std::cout << "\n=== DEBUG: Attention Scores (h=0, 16x16 region) ===" << std::endl;
        //     for (int i = 0; i < std::min(16, L); i++) {
        //         std::string row_str = "";
        //         for (int j = 0; j < std::min(16, L); j++) {
        //             std::cout << std::fixed << std::setprecision(4) << std::setw(8) << scores[i][j] / 0.125f << " ";
        //         }
        //         std::cout << "Row " << std::setw(2) << i << ": " << row_str << std::endl;
        //     }
        //     std::cout << "=== END DEBUG ===" << std::endl;
        // }
        
        // Apply softmax
        softmax_ref(scores, L);
        
        // Compute output for this head
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                float weighted_sum = 0.0f;
                for (int j = 0; j < L; j++) {  // causal mask
                    weighted_sum += scores[i][j] * v_heads[j][group_idx][d];
                }
                output[i][h * HEAD_DIM + d] = weighted_sum;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters
    const int L = 32;              // Sequence length
    const int num_act_centroids = 64;   // Number of activation centroids per position
    const int num_weight_centroids = 16; // Number of weight centroids per position
    const int vector_dim = 2;       // Dimension of each centroid
    const int num_streams = 16;      // Number of parallel streams
    
    std::cout << "Testing Qwen Block kernel with weight vector quantization:" << std::endl;
    std::cout << "  L (sequence length): " << L << std::endl;
    std::cout << "  Hidden dimension: " << HIDDEN_DIM << std::endl;
    std::cout << "  Intermediate dimension: " << INTERM_DIM << std::endl;
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
    std::uniform_real_distribution<float> weight_dis(-0.05f, 0.05f);
    std::uniform_real_distribution<float> input_dis(-0.2f, 0.2f);
    std::uniform_real_distribution<float> norm_weight_dis(0.5f, 1.5f);
    std::uniform_int_distribution<int> weight_idx_dis(0, num_weight_centroids - 1);

    const ap_uint<10> L_hw = ap_uint<10>((ap_uint<1>(0), ap_uint<9>(L)));
    
    // Generate random input
    std::cout << "Generating random input..." << std::endl;
    std::vector<std::vector<float>> input(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            input[i][j] = input_dis(gen);
        }
    }
    
    // Pack input into hardware format (wide format: L * HIDDEN_DIM_DIV_2 vectors of 16 floats)
    // Following input_reader_wide pattern from qwen_block.h
    std::vector<std::vector<tapa::vec_t<float, 16>>> input_hw(2, std::vector<tapa::vec_t<float, 16>>(L * HIDDEN_DIM / 32));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM / 32; j++) {
            for (int k = 0; k < 16; k++) {
                input_hw[0][i * HIDDEN_DIM / 32 + j][k] = input[i][j * 32 + k];
            }
            for (int k = 0; k < 16; k++) {
                input_hw[1][i * HIDDEN_DIM / 32 + j][k] = input[i][j * 32 + 16 + k];
            }
        }
    }
    
    // Generate RMS normalization weights
    std::cout << "Generating RMS normalization weights..." << std::endl;
    std::vector<float> rms_weight(HIDDEN_DIM);
    for (int i = 0; i < HIDDEN_DIM; i++) {
        rms_weight[i] = norm_weight_dis(gen);
    }
    
    // Pack RMS weights into hardware format
    std::vector<tapa::vec_t<float, 16>> rms_weight_hw(HIDDEN_DIM / 16);
    for (int i = 0; i < HIDDEN_DIM / 16; i++) {
        for (int j = 0; j < 16; j++) {
            rms_weight_hw[i][j] = rms_weight[i * 16 + j];
        }
    }
    
    // Generate activation centroids for combined attention and FFN
    std::cout << "Generating activation centroids..." << std::endl;

    std::vector<std::vector<std::vector<float>>> qkv_act_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(vector_dim)));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                qkv_act_centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }

    std::vector<std::vector<std::vector<float>>> out_act_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(vector_dim)));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                out_act_centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }

    std::vector<std::vector<std::vector<float>>> up_act_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(vector_dim)));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                up_act_centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }

    std::vector<std::vector<std::vector<float>>> down_act_centroids(INTERM_DIM_DIV_2,
        std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(vector_dim)));
    
    for (int pos = 0; pos < INTERM_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                down_act_centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }
    
    // Pack activation centroids into hardware format for 2 channels
    // Following centroid_reader_split pattern: split by alternating every 8 vectors between channels
    std::vector<std::vector<tapa::vec_t<float, 16>>> centroid_hw(2);
    centroid_hw[0].resize(TOTAL_CENTROID_SIZE * num_act_centroids / 16);
    centroid_hw[1].resize(TOTAL_CENTROID_SIZE * num_act_centroids / 16);
    
    std::vector<tapa::vec_t<float, 16>> centroid_hw_tmp(TOTAL_CENTROID_SIZE * num_act_centroids / 8);
    
    int offset = 0;
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroid_hw_tmp[offset+(pos/8)*num_act_centroids+i][(pos % 8)*vector_dim+j] = qkv_act_centroids[pos][i][j];
            }
        }
    }

    offset+=(HIDDEN_DIM_DIV_2 / 8) * num_act_centroids;
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroid_hw_tmp[offset+(pos/8)*num_act_centroids+i][(pos % 8)*vector_dim+j] = out_act_centroids[pos][i][j];
            }
        }
    }

    offset+=(HIDDEN_DIM_DIV_2 / 8) * num_act_centroids;
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroid_hw_tmp[offset+(pos/8)*num_act_centroids+i][(pos % 8)*vector_dim+j] = up_act_centroids[pos][i][j];
            }
        }
    }

    offset+=(HIDDEN_DIM_DIV_2 / 8) * num_act_centroids;
    for (int pos = 0; pos < INTERM_DIM_DIV_2; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroid_hw_tmp[offset+(pos/8)*num_act_centroids+i][(pos % 8)*vector_dim+j] = down_act_centroids[pos][i][j];
            }
        }
    }
    
    for (int i = 0; i < centroid_hw_tmp.size(); i++) {
        centroid_hw[(i/num_act_centroids)%2][((i/num_act_centroids)/2)*num_act_centroids+(i%num_act_centroids)] = centroid_hw_tmp[i];
    }
    
    // Generate sin/cos tables for RoPE
    std::cout << "Generating RoPE sin/cos tables..." << std::endl;
    std::vector<std::vector<float>> sin_table(L, std::vector<float>(HEAD_DIM));
    std::vector<std::vector<float>> cos_table(L, std::vector<float>(HEAD_DIM));
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            float freq = 1.0f / std::pow(10000.0f, static_cast<float>(2 * (j / 2)) / HEAD_DIM);
            float angle = i * freq;
            sin_table[i][j] = std::sin(angle);
            cos_table[i][j] = std::cos(angle);
        }
    }

    // Print sin and cos tables for the first 4 sequences
    // std::cout << "\n=== Sin Table (first 4 sequences) ===" << std::endl;
    // for (int i = 0; i < std::min(4, L); i++) {
    //     std::cout << "Seq " << i << ": ";
    //     for (int j = 0; j < HEAD_DIM; j++) {
    //         std::cout << std::fixed << std::setprecision(6) << sin_table[i][j];
    //         if (j < HEAD_DIM - 1) std::cout << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // std::cout << "\n=== Cos Table (first 4 sequences) ===" << std::endl;
    // for (int i = 0; i < std::min(4, L); i++) {
    //     std::cout << "Seq " << i << ": ";
    //     for (int j = 0; j < HEAD_DIM; j++) {
    //         std::cout << std::fixed << std::setprecision(6) << cos_table[i][j];
    //         if (j < HEAD_DIM - 1) std::cout << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // Pack sin/cos tables into hardware format
    std::vector<tapa::vec_t<float, 16>> sin_hw(L * HEAD_DIM / 16);
    std::vector<tapa::vec_t<float, 16>> cos_hw(L * HEAD_DIM / 16);
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HEAD_DIM / 16; j++) {
            for (int k = 0; k < 16; k++) {
                if(j < HEAD_DIM / 32) {
                    sin_hw[i * (HEAD_DIM / 16) + j][k] = sin_table[i][(HEAD_DIM / 32 + j) * 16 + k];
                } else {
                    sin_hw[i * (HEAD_DIM / 16) + j][k] = sin_table[i][(j - HEAD_DIM/32) * 16 + k];
                }
                cos_hw[i * (HEAD_DIM / 16) + j][k] = cos_table[i][j * 16 + k];
            }
        }
    }
    
    // Generate weight centroids and indices for all projections
    std::cout << "Generating weight centroids and indices..." << std::endl;
    
    // QKV projection
    int qkv_submatrices = (QKV_DIM + 511) / 512;
    std::vector<std::vector<std::vector<std::vector<float>>>> qkv_weight_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(qkv_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> qkv_weight_indices(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<int>>(qkv_submatrices, std::vector<int>(512)));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < qkv_submatrices; sub++) {
            // Generate weight centroids
            for (int i = 0; i < num_weight_centroids; i++) {
                for (int j = 0; j < vector_dim; j++) {
                    qkv_weight_centroids[pos][sub][i][j] = weight_dis(gen);
                }
            }
            // Generate weight indices by finding closest centroids to random weight vectors
            for (int col = 0; col < 512; col++) {
                // Generate random weight vector
                std::vector<float> weight_vec(vector_dim);
                for (int j = 0; j < vector_dim; j++) {
                    weight_vec[j] = weight_dis(gen);
                }
                // Find closest weight centroid
                qkv_weight_indices[pos][sub][col] = find_closest_centroid(weight_vec, qkv_weight_centroids[pos][sub]);
            }
        }
    }
    
    // Attention output projection  
    int attn_out_submatrices = (HIDDEN_DIM + 511) / 512;
    std::vector<std::vector<std::vector<std::vector<float>>>> attn_out_weight_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(attn_out_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> attn_out_weight_indices(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<int>>(attn_out_submatrices, std::vector<int>(512)));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < attn_out_submatrices; sub++) {
            // Generate weight centroids
            for (int i = 0; i < num_weight_centroids; i++) {
                for (int j = 0; j < vector_dim; j++) {
                    attn_out_weight_centroids[pos][sub][i][j] = weight_dis(gen);
                }
            }
            // Generate weight indices by finding closest centroids to random weight vectors
            for (int col = 0; col < 512; col++) {
                // Generate random weight vector
                std::vector<float> weight_vec(vector_dim);
                for (int j = 0; j < vector_dim; j++) {
                    weight_vec[j] = weight_dis(gen);
                }
                // Find closest weight centroid
                attn_out_weight_indices[pos][sub][col] = find_closest_centroid(weight_vec, attn_out_weight_centroids[pos][sub]);
            }
        }
    }
    
    // FFN projections (up, gate, down)
    int up_submatrices = (INTERM_DIM + 511) / 512;
    int down_submatrices = (HIDDEN_DIM + 511) / 512;
    
    std::vector<std::vector<std::vector<std::vector<float>>>> up_weight_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(up_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> up_weight_indices(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<int>>(up_submatrices, std::vector<int>(512)));
    
    std::vector<std::vector<std::vector<std::vector<float>>>> gate_weight_centroids(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(up_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> gate_weight_indices(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<int>>(up_submatrices, std::vector<int>(512)));
    
    std::vector<std::vector<std::vector<std::vector<float>>>> down_weight_centroids(INTERM_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(down_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> down_weight_indices(INTERM_DIM_DIV_2,
        std::vector<std::vector<int>>(down_submatrices, std::vector<int>(512)));
    
    // Generate FFN weight data
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < up_submatrices; sub++) {
            // Generate up weight centroids
            for (int i = 0; i < num_weight_centroids; i++) {
                for (int j = 0; j < vector_dim; j++) {
                    up_weight_centroids[pos][sub][i][j] = weight_dis(gen);
                    gate_weight_centroids[pos][sub][i][j] = weight_dis(gen);
                }
            }
            // Generate up weight indices
            for (int col = 0; col < 512; col++) {
                // Generate random weight vector for up projection
                std::vector<float> up_weight_vec(vector_dim);
                for (int j = 0; j < vector_dim; j++) {
                    up_weight_vec[j] = weight_dis(gen);
                }
                up_weight_indices[pos][sub][col] = find_closest_centroid(up_weight_vec, up_weight_centroids[pos][sub]);
                
                // Generate random weight vector for gate projection
                std::vector<float> gate_weight_vec(vector_dim);
                for (int j = 0; j < vector_dim; j++) {
                    gate_weight_vec[j] = weight_dis(gen);
                }
                gate_weight_indices[pos][sub][col] = find_closest_centroid(gate_weight_vec, gate_weight_centroids[pos][sub]);
            }
        }
    }
    
    for (int pos = 0; pos < INTERM_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < down_submatrices; sub++) {
            // Generate down weight centroids
            for (int i = 0; i < num_weight_centroids; i++) {
                for (int j = 0; j < vector_dim; j++) {
                    down_weight_centroids[pos][sub][i][j] = weight_dis(gen);
                }
            }
            // Generate down weight indices
            for (int col = 0; col < 512; col++) {
                // Generate random weight vector
                std::vector<float> weight_vec(vector_dim);
                for (int j = 0; j < vector_dim; j++) {
                    weight_vec[j] = weight_dis(gen);
                }
                down_weight_indices[pos][sub][col] = find_closest_centroid(weight_vec, down_weight_centroids[pos][sub]);
            }
        }
    }
    
    // Precompute floating-point 2D LUTs for all projections (dot products between activation and weight centroids)
    std::cout << "Precomputing 2D lookup tables..." << std::endl;
    
    // QKV projection LUT
    std::vector<std::vector<std::vector<std::vector<float>>>> qkv_lut_2d(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(qkv_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < qkv_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    // Compute dot product between activation and weight centroids
                    float dot_product = 0.0f;
                    for (int k = 0; k < vector_dim; k++) {
                        dot_product += qkv_act_centroids[pos][act_idx][k] * 
                                     qkv_weight_centroids[pos][sub][weight_idx][k];
                    }
                    qkv_lut_2d[pos][sub][act_idx][weight_idx] = dot_product;
                }
            }
        }
    }
    
    // Attention output projection LUT
    std::vector<std::vector<std::vector<std::vector<float>>>> attn_out_lut_2d(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(attn_out_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < attn_out_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    float dot_product = 0.0f;
                    for (int k = 0; k < vector_dim; k++) {
                        dot_product += out_act_centroids[pos][act_idx][k] * 
                                     attn_out_weight_centroids[pos][sub][weight_idx][k];
                    }
                    attn_out_lut_2d[pos][sub][act_idx][weight_idx] = dot_product;
                }
            }
        }
    }
    
    // FFN Up projection LUT  
    std::vector<std::vector<std::vector<std::vector<float>>>> up_lut_2d(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(up_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < up_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    float dot_product = 0.0f;
                    for (int k = 0; k < vector_dim; k++) {
                        // Use FFN centroids (starting at ATTN_CENTROID_SIZE)
                        dot_product += up_act_centroids[pos][act_idx][k] * 
                                     up_weight_centroids[pos][sub][weight_idx][k];
                    }
                    up_lut_2d[pos][sub][act_idx][weight_idx] = dot_product;
                }
            }
        }
    }
    
    // FFN Gate projection LUT
    std::vector<std::vector<std::vector<std::vector<float>>>> gate_lut_2d(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(up_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < up_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    float dot_product = 0.0f;
                    for (int k = 0; k < vector_dim; k++) {
                        dot_product += up_act_centroids[pos][act_idx][k] * 
                                     gate_weight_centroids[pos][sub][weight_idx][k];
                    }
                    gate_lut_2d[pos][sub][act_idx][weight_idx] = dot_product;
                }
            }
        }
    }
    
    // FFN Down projection LUT
    std::vector<std::vector<std::vector<std::vector<float>>>> down_lut_2d(INTERM_DIM_DIV_2,
        std::vector<std::vector<std::vector<float>>>(down_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    for (int pos = 0; pos < INTERM_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < down_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    float dot_product = 0.0f;
                    for (int k = 0; k < vector_dim; k++) {
                        // Use down FFN centroids (starting at ATTN_CENTROID_SIZE + HIDDEN_DIM_DIV_2)
                        dot_product += down_act_centroids[pos][act_idx][k] * 
                                     down_weight_centroids[pos][sub][weight_idx][k];
                    }
                    down_lut_2d[pos][sub][act_idx][weight_idx] = dot_product;
                }
            }
        }
    }
    
    
    // Compute scale and zero-point for quantization (separate for each projection)
    std::cout << "Computing quantization parameters..." << std::endl;
    auto [qkv_scale, qkv_zeropoint] = compute_scale_zeropoint(qkv_lut_2d, HIDDEN_DIM_DIV_2, qkv_submatrices, num_act_centroids, num_weight_centroids);
    auto [attn_out_scale, attn_out_zeropoint] = compute_scale_zeropoint(attn_out_lut_2d, HIDDEN_DIM_DIV_2, attn_out_submatrices, num_act_centroids, num_weight_centroids);
    auto [up_scale, up_zeropoint] = compute_scale_zeropoint(up_lut_2d, HIDDEN_DIM_DIV_2, up_submatrices, num_act_centroids, num_weight_centroids);
    auto [gate_scale, gate_zeropoint] = compute_scale_zeropoint(gate_lut_2d, HIDDEN_DIM_DIV_2, up_submatrices, num_act_centroids, num_weight_centroids);
    auto [down_scale, down_zeropoint] = compute_scale_zeropoint(down_lut_2d, INTERM_DIM_DIV_2, down_submatrices, num_act_centroids, num_weight_centroids);
    
    std::cout << "  QKV scale: " << qkv_scale << ", zeropoint: " << qkv_zeropoint << std::endl;
    std::cout << "  Attn Out scale: " << attn_out_scale << ", zeropoint: " << attn_out_zeropoint << std::endl;
    std::cout << "  Up scale: " << up_scale << ", zeropoint: " << up_zeropoint << std::endl;
    std::cout << "  Gate scale: " << gate_scale << ", zeropoint: " << gate_zeropoint << std::endl;
    std::cout << "  Down scale: " << down_scale << ", zeropoint: " << down_zeropoint << std::endl;
    
    // Quantize LUTs
    std::cout << "Quantizing LUTs..." << std::endl;
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> qkv_lut_2d_quantized(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<uint8_t>>>(qkv_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> attn_out_lut_2d_quantized(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<uint8_t>>>(attn_out_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> up_lut_2d_quantized(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<uint8_t>>>(up_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> gate_lut_2d_quantized(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<uint8_t>>>(up_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> down_lut_2d_quantized(INTERM_DIM_DIV_2,
        std::vector<std::vector<std::vector<uint8_t>>>(down_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    // Quantize all LUTs
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < qkv_submatrices; sub++) {
            for (int act = 0; act < num_act_centroids; act++) {
                for (int weight = 0; weight < num_weight_centroids; weight++) {
                    qkv_lut_2d_quantized[pos][sub][act][weight] = quantize_value(qkv_lut_2d[pos][sub][act][weight], qkv_scale, qkv_zeropoint);
                }
            }
        }
        for (int sub = 0; sub < attn_out_submatrices; sub++) {
            for (int act = 0; act < num_act_centroids; act++) {
                for (int weight = 0; weight < num_weight_centroids; weight++) {
                    attn_out_lut_2d_quantized[pos][sub][act][weight] = quantize_value(attn_out_lut_2d[pos][sub][act][weight], attn_out_scale, attn_out_zeropoint);
                }
            }
        }
        for (int sub = 0; sub < up_submatrices; sub++) {
            for (int act = 0; act < num_act_centroids; act++) {
                for (int weight = 0; weight < num_weight_centroids; weight++) {
                    up_lut_2d_quantized[pos][sub][act][weight] = quantize_value(up_lut_2d[pos][sub][act][weight], up_scale, up_zeropoint);
                    gate_lut_2d_quantized[pos][sub][act][weight] = quantize_value(gate_lut_2d[pos][sub][act][weight], gate_scale, gate_zeropoint);
                }
            }
        }
    }
    
    for (int pos = 0; pos < INTERM_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < down_submatrices; sub++) {
            for (int act = 0; act < num_act_centroids; act++) {
                for (int weight = 0; weight < num_weight_centroids; weight++) {
                    down_lut_2d_quantized[pos][sub][act][weight] = quantize_value(down_lut_2d[pos][sub][act][weight], down_scale, down_zeropoint);
                }
            }
        }
    }
    
    // Pack quantized LUT into hardware format following FFN testbench pattern
    // Order: QKV LUT, Attn Out LUT, Up+Gate LUT (concatenated), Down LUT
    std::cout << "Packing quantized LUT into hardware format..." << std::endl;
    
    // Calculate total LUT vectors needed for hardware format
    int qkv_lut_vectors = (HIDDEN_DIM_DIV_2 / 16) * qkv_submatrices * (num_act_centroids / 4);
    int attn_out_lut_vectors = (HIDDEN_DIM_DIV_2 / 16) * attn_out_submatrices * (num_act_centroids / 4);
    int up_gate_lut_vectors = (HIDDEN_DIM_DIV_2 / 16) * up_submatrices * 2 * (num_act_centroids / 4);  // *2 for up+gate concatenation
    int down_lut_vectors = (INTERM_DIM_DIV_2 / 16) * down_submatrices * (num_act_centroids / 4);
    int total_lut_vectors = qkv_lut_vectors + attn_out_lut_vectors + up_gate_lut_vectors + down_lut_vectors;
    
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> lut_hw(16);
    for (int buffer_idx = 0; buffer_idx < 16; buffer_idx++) {
        lut_hw[buffer_idx].resize(total_lut_vectors);
    }
    
    std::cout << "  QKV LUT vectors: " << qkv_lut_vectors << std::endl;
    std::cout << "  Attn Out LUT vectors: " << attn_out_lut_vectors << std::endl;
    std::cout << "  Up+Gate LUT vectors: " << up_gate_lut_vectors << std::endl;
    std::cout << "  Down LUT vectors: " << down_lut_vectors << std::endl;
    std::cout << "  Total LUT vectors: " << total_lut_vectors << std::endl;
    
    int vector_offset = 0;
    
    // Pack QKV LUT first
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < qkv_submatrices; sub++) {
            for (int act_group = 0; act_group < num_act_centroids / 4; act_group++) {
                int hw_idx = vector_offset + local_pos * qkv_submatrices * (num_act_centroids / 4) + act_group * qkv_submatrices + sub;
                
                // Pack 64 elements: 4 activation centroids x 16 weight centroids
                for (int k = 0; k < 16; k++) {
                    for (int ii = 0; ii < 4; ii++) {
                        int act_idx = act_group * 4 + ii;
                        if (act_idx < num_act_centroids && k < num_weight_centroids) {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = qkv_lut_2d_quantized[pos][sub][act_idx][k];
                        } else {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = 0;
                        }
                    }
                }
            }
        }
    }
    vector_offset += qkv_lut_vectors;
    
    // Pack Attention Output LUT second
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < attn_out_submatrices; sub++) {
            for (int act_group = 0; act_group < num_act_centroids / 4; act_group++) {
                int hw_idx = vector_offset + local_pos * attn_out_submatrices * (num_act_centroids / 4) + act_group * attn_out_submatrices + sub;
                
                for (int k = 0; k < 16; k++) {
                    for (int ii = 0; ii < 4; ii++) {
                        int act_idx = act_group * 4 + ii;
                        if (act_idx < num_act_centroids && k < num_weight_centroids) {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = attn_out_lut_2d_quantized[pos][sub][act_idx][k];
                        } else {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = 0;
                        }
                    }
                }
            }
        }
    }
    vector_offset += attn_out_lut_vectors;
    
    // Create concatenated up+gate LUT (following FFN pattern - up first, then gate)
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> up_gate_lut_2d_quantized(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<std::vector<uint8_t>>>(up_submatrices * 2,  // Concatenated dimension
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    // Fill up LUT first (index 0 to up_submatrices-1)
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < up_submatrices; sub++) {
            for (int act = 0; act < num_act_centroids; act++) {
                for (int weight = 0; weight < num_weight_centroids; weight++) {
                    up_gate_lut_2d_quantized[pos][sub][act][weight] = up_lut_2d_quantized[pos][sub][act][weight];
                }
            }
        }
    }
    
    // Fill gate LUT second (index up_submatrices to 2*up_submatrices-1)
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < up_submatrices; sub++) {
            for (int act = 0; act < num_act_centroids; act++) {
                for (int weight = 0; weight < num_weight_centroids; weight++) {
                    up_gate_lut_2d_quantized[pos][sub + up_submatrices][act][weight] = gate_lut_2d_quantized[pos][sub][act][weight];
                }
            }
        }
    }
    
    // Pack up+gate LUT (concatenated)
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < up_submatrices * 2; sub++) {  // Iterate through all concatenated submatrices
            for (int act_group = 0; act_group < num_act_centroids / 4; act_group++) {
                int hw_idx = vector_offset + local_pos * up_submatrices * 2 * (num_act_centroids / 4) + act_group * up_submatrices * 2 + sub;
                
                for (int k = 0; k < 16; k++) {
                    for (int ii = 0; ii < 4; ii++) {
                        int act_idx = act_group * 4 + ii;
                        if (act_idx < num_act_centroids && k < num_weight_centroids) {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = up_gate_lut_2d_quantized[pos][sub][act_idx][k];
                        } else {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = 0;
                        }
                    }
                }
            }
        }
    }
    vector_offset += up_gate_lut_vectors;
    
    // Pack down LUT last
    for (int pos = 0; pos < INTERM_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < down_submatrices; sub++) {
            for (int act_group = 0; act_group < num_act_centroids / 4; act_group++) {
                int hw_idx = vector_offset + local_pos * down_submatrices * (num_act_centroids / 4) + act_group * down_submatrices + sub;
                
                for (int k = 0; k < 16; k++) {
                    for (int ii = 0; ii < 4; ii++) {
                        int act_idx = act_group * 4 + ii;
                        if (act_idx < num_act_centroids && k < num_weight_centroids) {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = down_lut_2d_quantized[pos][sub][act_idx][k];
                        } else {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = 0;
                        }
                    }
                }
            }
        }
    }
    
    
    // Pack weight indices into hardware format following FFN testbench pattern
    // Order: QKV indices, Attn Out indices, Up+Gate indices (concatenated), Down indices
    std::cout << "Packing weight indices into hardware format..." << std::endl;
    
    // Calculate total weight index vectors needed for hardware format
    int qkv_weight_vectors = (HIDDEN_DIM_DIV_2 / 16) * qkv_submatrices * 4;
    int attn_out_weight_vectors = (HIDDEN_DIM_DIV_2 / 16) * attn_out_submatrices * 4;
    int up_gate_weight_vectors = (HIDDEN_DIM_DIV_2 / 16) * up_submatrices * 2 * 4;  // *2 for up+gate, *4 for vec_idx
    int down_weight_vectors = (INTERM_DIM_DIV_2 / 16) * down_submatrices * 4;
    int total_weight_vectors = qkv_weight_vectors + attn_out_weight_vectors + up_gate_weight_vectors + down_weight_vectors;
    
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> weight_idx_hw(16);
    for (int buffer_idx = 0; buffer_idx < 16; buffer_idx++) {
        weight_idx_hw[buffer_idx].resize(total_weight_vectors);
    }
    
    std::cout << "  QKV weight vectors: " << qkv_weight_vectors << std::endl;
    std::cout << "  Attn Out weight vectors: " << attn_out_weight_vectors << std::endl;
    std::cout << "  Up+Gate weight vectors: " << up_gate_weight_vectors << std::endl;
    std::cout << "  Down weight vectors: " << down_weight_vectors << std::endl;
    std::cout << "  Total weight vectors: " << total_weight_vectors << std::endl;
    
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
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0;
                    }
                }
            }
        }
    }
    vector_offset += qkv_weight_vectors;
    
    // Pack attention output weight indices second
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < attn_out_submatrices; sub++) {
            for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                int hw_idx = vector_offset + local_pos * attn_out_submatrices * 4 + sub * 4 + vec_idx;
                
                for (int k = 0; k < 64; k++) {
                    int col = vec_idx * 128 + k * 2;
                    if (col < 512) {
                        ap_uint<8> tmp_idx;
                        tmp_idx(3, 0) = attn_out_weight_indices[pos][sub][col];
                        tmp_idx(7, 4) = attn_out_weight_indices[pos][sub][col + 1];
                        weight_idx_hw[buffer_idx][hw_idx][k] = tmp_idx;
                    } else {
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0;
                    }
                }
            }
        }
    }
    vector_offset += attn_out_weight_vectors;
    
    // Create concatenated up+gate weight indices (following FFN pattern)
    std::vector<std::vector<std::vector<int>>> up_gate_weight_indices(HIDDEN_DIM_DIV_2,
        std::vector<std::vector<int>>(up_submatrices * 2, std::vector<int>(512)));
    
    // Fill up weight indices first (index 0 to up_submatrices-1)
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < up_submatrices; sub++) {
            for (int i = 0; i < 512; i++) {
                up_gate_weight_indices[pos][sub][i] = up_weight_indices[pos][sub][i];
            }
        }
    }
    
    // Fill gate weight indices second (index up_submatrices to 2*up_submatrices-1)
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int sub = 0; sub < up_submatrices; sub++) {
            for (int i = 0; i < 512; i++) {
                up_gate_weight_indices[pos][sub + up_submatrices][i] = gate_weight_indices[pos][sub][i];
            }
        }
    }
    
    // Pack up+gate weight indices
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < up_submatrices * 2; sub++) {  // Iterate through all concatenated submatrices
            for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                int hw_idx = vector_offset + local_pos * up_submatrices * 2 * 4 + sub * 4 + vec_idx;
                
                for (int k = 0; k < 64; k++) {
                    int weight_idx = vec_idx * 128 + k * 2;
                    if (weight_idx < 512) {
                        ap_uint<8> tmp_idx;
                        tmp_idx(3, 0) = up_gate_weight_indices[pos][sub][weight_idx];
                        tmp_idx(7, 4) = up_gate_weight_indices[pos][sub][weight_idx + 1];
                        weight_idx_hw[buffer_idx][hw_idx][k] = tmp_idx;
                    } else {
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0;
                    }
                }
            }
        }
    }
    vector_offset += up_gate_weight_vectors;
    
    // Pack down weight indices last
    for (int pos = 0; pos < INTERM_DIM_DIV_2; pos++) {
        int buffer_idx = pos % 16;
        int local_pos = pos / 16;
        
        for (int sub = 0; sub < down_submatrices; sub++) {
            for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                int hw_idx = vector_offset + local_pos * down_submatrices * 4 + sub * 4 + vec_idx;
                
                for (int k = 0; k < 64; k++) {
                    int col = vec_idx * 128 + k * 2;
                    if (col < 512) {
                        ap_uint<8> tmp_idx;
                        tmp_idx(3, 0) = down_weight_indices[pos][sub][col];
                        tmp_idx(7, 4) = down_weight_indices[pos][sub][col + 1];
                        weight_idx_hw[buffer_idx][hw_idx][k] = tmp_idx;
                    } else {
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0;
                    }
                }
            }
        }
    }
    
    // Pack scale and zeropoint values for hardware (following memory_matcher_acc_overlay_half pattern)
    // Order: TOTAL_HEADS scales/zeropoints for QKV (round 0), 1 for attn out (round 1), 2 for up/gate (round 2), 1 for down (round 3)
    std::vector<ap_uint<64>> scale_zero_hw;
    
    // QKV projection - one scale/zeropoint per head (TOTAL_HEADS) 
    for (int h = 0; h < TOTAL_HEADS; h++) {
        float zero_hw = qkv_zeropoint * qkv_scale * HIDDEN_DIM_DIV_2;
        ap_uint<32> scale_bits = tapa::bit_cast<ap_uint<32>>(qkv_scale);
        ap_uint<32> zeropoint_bits = tapa::bit_cast<ap_uint<32>>(zero_hw);
        ap_uint<64> packed = ap_uint<64>((zeropoint_bits, scale_bits));
        scale_zero_hw.push_back(packed);
    }
    
    // Attention output projection
    {
        float zero_hw = attn_out_zeropoint * attn_out_scale * HIDDEN_DIM_DIV_2;
        ap_uint<32> scale_bits = tapa::bit_cast<ap_uint<32>>(attn_out_scale);
        ap_uint<32> zeropoint_bits = tapa::bit_cast<ap_uint<32>>(zero_hw);
        ap_uint<64> packed = ap_uint<64>((zeropoint_bits, scale_bits));
        scale_zero_hw.push_back(packed);
    }
    
    // Up and gate projections
    {
        float zero_hw = up_zeropoint * up_scale * HIDDEN_DIM_DIV_2;
        ap_uint<32> scale_bits = tapa::bit_cast<ap_uint<32>>(up_scale);
        ap_uint<32> zeropoint_bits = tapa::bit_cast<ap_uint<32>>(zero_hw);
        ap_uint<64> packed = ap_uint<64>((zeropoint_bits, scale_bits));
        scale_zero_hw.push_back(packed);
    }
    {
        float zero_hw = gate_zeropoint * gate_scale * HIDDEN_DIM_DIV_2;
        ap_uint<32> scale_bits = tapa::bit_cast<ap_uint<32>>(gate_scale);
        ap_uint<32> zeropoint_bits = tapa::bit_cast<ap_uint<32>>(zero_hw);
        ap_uint<64> packed = ap_uint<64>((zeropoint_bits, scale_bits));
        scale_zero_hw.push_back(packed);
    }
    
    // Down projection
    {
        float zero_hw = down_zeropoint * down_scale * INTERM_DIM_DIV_2;
        ap_uint<32> scale_bits = tapa::bit_cast<ap_uint<32>>(down_scale);
        ap_uint<32> zeropoint_bits = tapa::bit_cast<ap_uint<32>>(zero_hw);
        ap_uint<64> packed = ap_uint<64>((zeropoint_bits, scale_bits));
        scale_zero_hw.push_back(packed);
    }
    
    std::cout << "Hardware format preparation completed." << std::endl;
    std::cout << "  Total scale/zeropoint entries: " << scale_zero_hw.size() << std::endl;
    std::cout << "  Total LUT vectors: " << total_lut_vectors << std::endl;
    std::cout << "  Total weight index vectors: " << total_weight_vectors << std::endl;
    
    // Combine LUT and weight index buffers into the format expected by hardware
    std::cout << "Combining LUT and weight index buffers..." << std::endl;
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> lut_weight_idx_hw(16);
    for (int buffer_idx = 0; buffer_idx < 16; buffer_idx++) {
        lut_weight_idx_hw[buffer_idx].resize(lut_hw[0].size() + weight_idx_hw[0].size());
    }

    const int round_0_lut_bound = (num_act_centroids >> 2) * (QKV_DIM >> 9);
    const int round_1_lut_bound = (num_act_centroids >> 2) * (HIDDEN_DIM >> 9);
    const int round_0_weight_bound = (QKV_DIM >> 7);
    const int round_1_weight_bound = (HIDDEN_DIM >> 7);
    const int round_0_bound = (HIDDEN_DIM_DIV_2 >> 4);
    const int round_1_bound = (HIDDEN_DIM_DIV_2 >> 4);  // Only one layer for attention

    const int round_2_lut_bound = (num_act_centroids >> 2) * (INTERM_DIM_MUL_2 >> 9);
    const int round_3_lut_bound = (num_act_centroids >> 2) * (HIDDEN_DIM >> 9);
    const int round_2_weight_bound = (INTERM_DIM_MUL_2 >> 7);
    const int round_3_weight_bound = (HIDDEN_DIM >> 7);
    const int round_2_bound = (HIDDEN_DIM_DIV_2 >> 4);
    const int round_3_bound = (INTERM_DIM_DIV_2 >> 4);

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

        for(int r = 0; r < round_2_bound; r++){
            for(int i = 0; i < round_2_lut_bound; i++) {
                lut_weight_idx_hw[buffer_idx][vec_idx] = lut_hw[buffer_idx][i + r * round_2_lut_bound + round_1_bound * round_1_lut_bound + round_0_bound * round_0_lut_bound];
                vec_idx++;
            }
            for(int i = 0; i < round_2_weight_bound; i++) {
                lut_weight_idx_hw[buffer_idx][vec_idx] = weight_idx_hw[buffer_idx][i + r * round_2_weight_bound + round_1_bound * round_1_weight_bound + round_0_bound * round_0_weight_bound];
                vec_idx++;
            }
        }

        for(int r = 0; r < round_3_bound; r++){
            for(int i = 0; i < round_3_lut_bound; i++) {
                lut_weight_idx_hw[buffer_idx][vec_idx] = lut_hw[buffer_idx][i + r * round_3_lut_bound + round_2_bound * round_2_lut_bound + round_1_bound * round_1_lut_bound + round_0_bound * round_0_lut_bound];
                vec_idx++;
            }
            for(int i = 0; i < round_3_weight_bound; i++) {
                lut_weight_idx_hw[buffer_idx][vec_idx] = weight_idx_hw[buffer_idx][i + r * round_3_weight_bound + round_2_bound * round_2_weight_bound + round_1_bound * round_1_weight_bound + round_0_bound * round_0_weight_bound];
                vec_idx++;
            }
        }
    }
    
    // Compute reference implementation for comparison
    std::cout << "Computing reference implementation..." << std::endl;

    // Reference complete qwen block implementation
    std::vector<std::vector<float>> reference_output(L, std::vector<float>(HIDDEN_DIM));
    
    // Step 1: First RMS normalization
    std::vector<std::vector<float>> norm_input(L, std::vector<float>(HIDDEN_DIM));
    reference_rms_norm(input, rms_weight, norm_input, L);

    // std::cout << "\n=== DEBUG: Norm Output (first 4x8 elements) ===" << std::endl;
    // std::cout << std::fixed << std::setprecision(6);
    // for (int i = 0; i < std::min(4, L); i++) {
    //     for (int j = 0; j < std::min(8, HIDDEN_DIM); j++) {
    //         std::cout << "first norm [" << i << "][" << j << "]: " << norm_input[i][j] << std::endl;
    //     }
    // }
    // std::cout << "=== END DEBUG ===" << std::endl;
    
    // Step 2: QKV projection using quantized LUT
    std::vector<std::vector<float>> qkv_proj(L, std::vector<float>(QKV_DIM));
    
    // Reshape input for linear projection: L x HIDDEN_DIM → HIDDEN_DIM_DIV_2 x L x 2
    std::vector<std::vector<std::vector<float>>> attn_input_3d(HIDDEN_DIM_DIV_2, 
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < L; i++) {
            attn_input_3d[pos][i][0] = norm_input[i][pos * 2];
            attn_input_3d[pos][i][1] = norm_input[i][pos * 2 + 1];
        }
    }
    
    reference_linear_quantized_lut(
        attn_input_3d, qkv_act_centroids, qkv_weight_indices, qkv_lut_2d_quantized,
        qkv_scale, qkv_zeropoint, qkv_proj, L, HIDDEN_DIM_DIV_2, QKV_DIM
    );
    
    // Step 3: Extract QKV heads according to the hardware ordering
    // Hardware order: v[0], k[0], q[0:1], v[1], k[1], q[2:3], ..., v[7], k[7], q[14:15]
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
        
        // Q heads for group g (HEAD_PER_GROUP heads)
        for (int h = 0; h < HEAD_PER_GROUP; h++) {
            for (int i = 0; i < L; i++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    q_heads[i][g * HEAD_PER_GROUP + h][d] = qkv_proj[i][head_idx * HEAD_DIM + d];
                }
            }
            head_idx++;
        }
    }

    // std::cout << "\n=== DEBUG: V Output (first 4x8 elements) ===" << std::endl;
    // std::cout << std::fixed << std::setprecision(6);
    // for (int i = 0; i < std::min(4, L); i++) {
    //     for (int j = 0; j < std::min(8, HEAD_DIM); j++) {
    //         std::cout << "V Output [" << i << "][" << j << "]: " << v_heads[i][0][j] << std::endl;
    //     }
    // }
    // std::cout << "=== END DEBUG ===" << std::endl;

    // std::cout << "\n=== DEBUG: K Output (first 4x8 elements) ===" << std::endl;
    // std::cout << std::fixed << std::setprecision(6);
    // for (int j = 0; j < HEAD_DIM; j++) {
    //     std::cout << "K Output [" << 26 << "][" << j << "]: " << k_heads[26][0][j] << std::endl;
    // }
    // std::cout << "=== END DEBUG ===" << std::endl;

    // std::cout << "\n=== DEBUG: Q Output (first 4x8 elements) ===" << std::endl;
    // std::cout << std::fixed << std::setprecision(6);
    // for (int i = 0; i < std::min(4, L); i++) {
    //     for (int j = 0; j < std::min(8, HEAD_DIM); j++) {
    //         std::cout << "Q Output [" << i << "][" << j << "]: " << q_heads[i][0][j] << std::endl;
    //     }
    // }
    // std::cout << "=== END DEBUG ===" << std::endl;
    
    // Step 4: Apply RoPE to Q and K heads
    for (int g = 0; g < NUM_GROUPS; g++) {
        // Apply RoPE to K heads
        std::vector<std::vector<float>> k_head_2d(L, std::vector<float>(HEAD_DIM));
        std::vector<std::vector<float>> k_head_rope(L, std::vector<float>(HEAD_DIM));
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                k_head_2d[i][d] = k_heads[i][g][d];
            }
        }
        apply_rotary_pos_emb_ref(k_head_2d, k_head_rope, cos_table, sin_table, L, HEAD_DIM);
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                k_heads[i][g][d] = k_head_rope[i][d];
            }
        }

        // Debug: Print 4x16 region of k_heads for g=0
        // if (g == 0) {
        //     std::cout << "\n=== DEBUG: K heads after RoPE (g=0, 4x16 region) ===" << std::endl;
        //     for (int j = 0; j < std::min(16, HEAD_DIM); j++) {
        //         std::cout << std::fixed << std::setprecision(6) << std::setw(8) << k_head_rope[26][j];
        //         if (j < std::min(16, HEAD_DIM) - 1) std::cout << " ";
        //     }
        //     std::cout << std::endl;
        //     std::cout << "=== END DEBUG ===" << std::endl;
        // }
        
        // Apply RoPE to Q heads
        for (int h = 0; h < HEAD_PER_GROUP; h++) {
            std::vector<std::vector<float>> q_head_2d(L, std::vector<float>(HEAD_DIM));
            std::vector<std::vector<float>> q_head_rope(L, std::vector<float>(HEAD_DIM));
            for (int i = 0; i < L; i++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    q_head_2d[i][d] = q_heads[i][g * HEAD_PER_GROUP + h][d];
                }
            }
            apply_rotary_pos_emb_ref(q_head_2d, q_head_rope, cos_table, sin_table, L, HEAD_DIM);
            for (int i = 0; i < L; i++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    q_heads[i][g * HEAD_PER_GROUP + h][d] = q_head_rope[i][d];
                }
            }
        }
    }
    
    // Step 5: Compute grouped query attention
    std::vector<std::vector<float>> attn_output(L, std::vector<float>(HIDDEN_DIM));
    reference_grouped_query_attention(q_heads, k_heads, v_heads, attn_output, L);

    std::vector<std::vector<std::vector<float>>> attn_output_3d(HIDDEN_DIM_DIV_2, 
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < L; i++) {
            attn_output_3d[pos][i][0] = attn_output[i][pos * 2];
            attn_output_3d[pos][i][1] = attn_output[i][pos * 2 + 1];
        }
    }
    
    // Step 6: Output projection
    std::vector<std::vector<float>> attn_proj_output(L, std::vector<float>(HIDDEN_DIM));
    reference_linear_quantized_lut(
        attn_output_3d, out_act_centroids, attn_out_weight_indices, attn_out_lut_2d_quantized,
        attn_out_scale, attn_out_zeropoint, attn_proj_output, L, HIDDEN_DIM_DIV_2, HIDDEN_DIM
    );
    
    // Debug logging: Attention output (first 4x8 elements)
    // std::cout << "\n=== DEBUG: Attention Output (first 4x8 elements) ===" << std::endl;
    // std::cout << std::fixed << std::setprecision(6);
    // for (int j = 0; j < 16; j++) {
    //     std::cout << "Attention Output [" << 26 << "][" << j << "]: " << attn_proj_output[26][j] << std::endl;
    // }
    // std::cout << "=== END DEBUG ===" << std::endl;
    
    // Step 7: First residual connection
    std::vector<std::vector<float>> residual_after_attn(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            residual_after_attn[i][j] = input[i][j] + attn_proj_output[i][j];
        }
    }
    
    // Step 8: Second RMS normalization
    std::vector<std::vector<float>> norm_ffn_input(L, std::vector<float>(HIDDEN_DIM));
    reference_rms_norm(residual_after_attn, rms_weight, norm_ffn_input, L);

    // std::cout << "\n=== DEBUG: Norm Output (first 4x8 elements) ===" << std::endl;
    // std::cout << std::fixed << std::setprecision(6);
    // for (int i = 0; i < std::min(4, L); i++) {
    //     for (int j = 0; j < std::min(8, HIDDEN_DIM); j++) {
    //         std::cout << "first norm [" << i << "][" << j << "]: " << norm_ffn_input[i][j] << std::endl;
    //     }
    // }
    // std::cout << "=== END DEBUG ===" << std::endl;
    
    // Step 9: FFN Up projection
    std::vector<std::vector<float>> up_output(L, std::vector<float>(INTERM_DIM));
    std::vector<std::vector<std::vector<float>>> ffn_input_3d(HIDDEN_DIM_DIV_2, 
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < L; i++) {
            ffn_input_3d[pos][i][0] = norm_ffn_input[i][pos * 2];
            ffn_input_3d[pos][i][1] = norm_ffn_input[i][pos * 2 + 1];
        }
    }
    
    reference_linear_quantized_lut(
        ffn_input_3d, up_act_centroids, up_weight_indices, up_lut_2d_quantized,
        up_scale, up_zeropoint, up_output, L, HIDDEN_DIM_DIV_2, INTERM_DIM
    );
    
    // Step 10: FFN Gate projection
    std::vector<std::vector<float>> gate_output(L, std::vector<float>(INTERM_DIM));
    reference_linear_quantized_lut(
        ffn_input_3d, up_act_centroids, gate_weight_indices, gate_lut_2d_quantized,
        gate_scale, gate_zeropoint, gate_output, L, HIDDEN_DIM_DIV_2, INTERM_DIM
    );
    
    // Step 11: Apply SiLU to gate and element-wise multiply
    std::vector<std::vector<float>> intermediate(L, std::vector<float>(INTERM_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < INTERM_DIM; j++) {
            float silu_gate = silu_piecewise(gate_output[i][j]);
            intermediate[i][j] = up_output[i][j] * silu_gate;
        }
    }
    
    // Step 12: FFN Down projection
    std::vector<std::vector<float>> ffn_output(L, std::vector<float>(HIDDEN_DIM));
    std::vector<std::vector<std::vector<float>>> down_input_3d(INTERM_DIM_DIV_2, 
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    for (int pos = 0; pos < INTERM_DIM_DIV_2; pos++) {
        for (int i = 0; i < L; i++) {
            down_input_3d[pos][i][0] = intermediate[i][pos * 2];
            down_input_3d[pos][i][1] = intermediate[i][pos * 2 + 1];
        }
    }
    
    reference_linear_quantized_lut(
        down_input_3d, down_act_centroids, down_weight_indices, down_lut_2d_quantized,
        down_scale, down_zeropoint, ffn_output, L, INTERM_DIM_DIV_2, HIDDEN_DIM
    );
    
    // Debug logging: FFN output (first 4x8 elements)
    // std::cout << "\n=== DEBUG: FFN Output (first 4x8 elements) ===" << std::endl;
    // std::cout << std::fixed << std::setprecision(6);
    // for (int j = 0; j < 16; j++) {
    //     std::cout << "FFN Output [" << 26 << "][" << j << "]: " << ffn_output[26][j] << std::endl;
    // }
    // std::cout << "=== END DEBUG ===" << std::endl;
    
    // Step 13: Second residual connection
    std::vector<std::vector<float>> final_residual(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            final_residual[i][j] = residual_after_attn[i][j] + ffn_output[i][j];
        }
    }
    
    // Step 14: Final RMS normalization
    reference_rms_norm(final_residual, rms_weight, reference_output, L);
    
    std::cout << "Reference computation completed!" << std::endl;
    std::cout << "Sample reference output (first sequence, first 8 elements): ";
    for (int j = 0; j < std::min(8, HIDDEN_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << reference_output[0][j] << " ";
    }
    std::cout << std::endl;
    
    // Prepare output buffer for hardware
    std::vector<std::vector<tapa::vec_t<float, 16>>> output_hw(2, std::vector<tapa::vec_t<float, 16>>(L * HIDDEN_DIM / 32));
    std::vector<int> cycle_count_hw(1);
    
    // Initialize output buffer
    // for (int i = 0; i < output_hw[0].size(); i++) {
    //     for (int j = 0; j < 32; j++) {
    //         output_hw[0][i][j] = 0.0f;
    //         output_hw[1][i][j] = 0.0f;
    //     }
    // }
    cycle_count_hw[0] = 0;
    
    std::cout << "Invoking hardware..." << std::endl;

    std::vector<tapa::vec_t<float, 16>> k_cache_hw(L * HIDDEN_DIM / 32);
    std::vector<tapa::vec_t<float, 16>> v_cache_hw(L * HIDDEN_DIM / 32);

    // Hardware invocation
    tapa::invoke(
        qwen_block, FLAGS_bitstream,
        L_hw,
        tapa::read_write_mmap<tapa::vec_t<float, 16>>(k_cache_hw),
        tapa::read_write_mmap<tapa::vec_t<float, 16>>(v_cache_hw),
        tapa::read_only_mmaps<tapa::vec_t<float, 16>, 2>(input_hw),
        tapa::read_only_mmaps<tapa::vec_t<float, 16>, 2>(centroid_hw),
        tapa::read_only_mmaps<tapa::vec_t<ap_uint<8>, 64>, 16>(lut_weight_idx_hw),
        tapa::read_only_mmap<ap_uint<64>>(scale_zero_hw),
        tapa::read_only_mmap<tapa::vec_t<float, 16>>(sin_hw),
        tapa::read_only_mmap<tapa::vec_t<float, 16>>(cos_hw),
        tapa::read_only_mmap<tapa::vec_t<float, 16>>(rms_weight_hw),
        tapa::write_only_mmaps<tapa::vec_t<float, 16>, 2>(output_hw)
    );
    
    std::cout << "Hardware execution completed!" << std::endl;
    std::cout << "Hardware cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Extract hardware output to 2D format for comparison
    std::vector<std::vector<float>> hardware_output(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            int vec_idx = (i * HIDDEN_DIM + j) / 32;
            int elem_idx = (i * HIDDEN_DIM + j) % 32;
            hardware_output[i][j] = output_hw[elem_idx/16][vec_idx][elem_idx%16];
        }
    }
    
    // Compare hardware and reference outputs
    std::cout << "Comparing hardware and reference outputs..." << std::endl;
    
    int errors = 0;
    float max_error = 0.0f;
    const float tolerance = 2e-1f;  // Tolerance for quantization effects
    const float rel_tol = 1e-1f;
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float diff = std::abs(hardware_output[i][j] - reference_output[i][j]);
            if (diff > max_error) {
                max_error = diff;
            }

            float rel_error = 0.0f;
            if (std::abs(reference_output[i][j]) > 1e-8f) {
                rel_error = diff / std::abs(reference_output[i][j]);
            }
            
            // Consider it correct if either absolute or relative error is within tolerance
            bool is_correct = (diff <= tolerance) || (rel_error <= rel_tol);

            if (!is_correct) {
                errors++;
                if (errors <= 10) {  // Print first 10 errors for debugging
                    std::cout << "Error at [" << i << "][" << j << "]: HW=" 
                             << std::fixed << std::setprecision(6) << hardware_output[i][j] 
                             << ", REF=" << reference_output[i][j] 
                             << ", diff=" << diff << std::endl;
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
        std::cout << "Output [0][" << j << "]: HW=" << hardware_output[0][j] 
                 << ", REF=" << reference_output[0][j] 
                 << ", diff=" << std::abs(hardware_output[0][j] - reference_output[0][j]) << std::endl;
    }
    
    // Print transformer block analysis
    std::cout << "\n=== Transformer Block Analysis ===" << std::endl;
    std::cout << "Processing flow: Input -> RMS Norm -> QKV -> Attention -> Output Proj -> Residual -> RMS Norm -> FFN -> Residual -> Final RMS Norm" << std::endl;
    std::cout << "Attention mechanism: " << NUM_GROUPS << " groups, " << NUM_HEADS << " total heads, " << HEAD_PER_GROUP << " heads per group" << std::endl;
    std::cout << "FFN mechanism: " << HIDDEN_DIM << " -> " << INTERM_DIM << " -> " << HIDDEN_DIM << " with SiLU activation" << std::endl;
    std::cout << "All projections use quantized LUT-based matrix multiplication" << std::endl;
    
    // Print statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Input sequence length: " << L << std::endl;
    std::cout << "  Hidden dimension: " << HIDDEN_DIM << std::endl;
    std::cout << "  Intermediate dimension: " << INTERM_DIM << std::endl;
    std::cout << "  Total parameters processed:" << std::endl;
    std::cout << "    QKV projection: " << (HIDDEN_DIM * QKV_DIM) << std::endl;
    std::cout << "    Attention output: " << (HIDDEN_DIM * HIDDEN_DIM) << std::endl;
    std::cout << "    FFN up/gate: " << (2 * HIDDEN_DIM * INTERM_DIM) << std::endl;
    std::cout << "    FFN down: " << (INTERM_DIM * HIDDEN_DIM) << std::endl;
    std::cout << "  Total activation centroids: " << (ATTN_CENTROID_SIZE + FFN_CENTROID_SIZE) * num_act_centroids << std::endl;
    std::cout << "  Total weight centroids per layer:" << std::endl;
    std::cout << "    QKV: " << (HIDDEN_DIM_DIV_2 * qkv_submatrices * num_weight_centroids) << std::endl;
    std::cout << "    Attention Output: " << (HIDDEN_DIM_DIV_2 * attn_out_submatrices * num_weight_centroids) << std::endl;
    std::cout << "    FFN Up: " << (HIDDEN_DIM_DIV_2 * up_submatrices * num_weight_centroids) << std::endl;
    std::cout << "    FFN Gate: " << (HIDDEN_DIM_DIV_2 * up_submatrices * num_weight_centroids) << std::endl;
    std::cout << "    FFN Down: " << (INTERM_DIM_DIV_2 * down_submatrices * num_weight_centroids) << std::endl;
    std::cout << "  Memory bandwidth:" << std::endl;
    std::cout << "    Input: " << (L * HIDDEN_DIM * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    LUT tables: " << (16 * total_lut_vectors * 64) << " bytes" << std::endl;
    std::cout << "    Weight indices: " << (16 * total_weight_vectors * 64) << " bytes" << std::endl;
    std::cout << "    RoPE tables: " << (2 * L * HEAD_DIM / 2 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Output: " << (L * HIDDEN_DIM * sizeof(float)) << " bytes" << std::endl;
    
    return errors == 0 ? 0 : 1;
}
