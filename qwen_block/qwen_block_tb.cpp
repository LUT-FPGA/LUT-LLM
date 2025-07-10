#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "qwen_block.h"

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

// Reference softmax implementation (matching hardware behavior)
void softmax_ref(std::vector<std::vector<float>>& attention_scores, int L) {
    for (int i = 0; i < L; i++) {
        // Compute exp and sum (hardware uses scaling factor 0.125)
        float sum = 0.0f;
        for (int j = 0; j < L; j++) {
            if (i >= j) {  // causal mask: row i can attend to column j if i >= j
                float scaled_score = attention_scores[i][j] * 0.125f;  // Apply same scaling as hardware
                float exp_val = std::exp(scaled_score);
                attention_scores[i][j] = exp_val;
                sum += exp_val;
            } else {
                attention_scores[i][j] = 0.0f;  // masked positions
            }
        }
        
        // Normalize
        for (int j = 0; j < L; j++) {
            if (i >= j) {
                attention_scores[i][j] /= sum;
            }
        }
    }
}

// Reference linear projection using LUT-based approach
void reference_linear_projection(
    const std::vector<std::vector<float>>& input,  // L x in_dim
    const std::vector<std::vector<std::vector<float>>>& centroids,  // (in_dim/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& lut,        // (in_dim/2) x num_centroids x out_dim
    std::vector<std::vector<float>>& output,       // L x out_dim
    int L, int in_dim, int out_dim
) {
    int vector_dim = 2;
    int in_size = in_dim / vector_dim;
    
    // Initialize output
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_dim; j++) {
            output[i][j] = 0.0f;
        }
    }
    
    // For each sequence and each position
    for (int i = 0; i < L; i++) {
        for (int pos = 0; pos < in_size; pos++) {
            std::vector<float> point(vector_dim);
            point[0] = input[i][pos * vector_dim];
            point[1] = input[i][pos * vector_dim + 1];
            
            int closest_idx = find_closest_centroid(point, centroids[pos]);
            
            for (int j = 0; j < out_dim; j++) {
                output[i][j] += lut[pos][closest_idx][j];
            }
        }
    }
}

// Reference attention block implementation
void reference_attention_block(
    const std::vector<std::vector<float>>& input,  // L x HIDDEN_DIM
    const std::vector<std::vector<std::vector<float>>>& qk_centroids,    // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& qk_lut,          // (HIDDEN_DIM/2) x num_centroids x QK_DIM
    const std::vector<std::vector<std::vector<float>>>& v_centroids,     // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& v_lut,           // (HIDDEN_DIM/2) x num_centroids x V_DIM
    const std::vector<std::vector<std::vector<float>>>& out_centroids,   // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& out_lut,         // (HIDDEN_DIM/2) x num_centroids x HIDDEN_DIM
    const std::vector<std::vector<float>>& sin_table,  // L x HEAD_DIM
    const std::vector<std::vector<float>>& cos_table,  // L x HEAD_DIM
    std::vector<std::vector<float>>& output,       // L x HIDDEN_DIM
    int L
) {
    const int num_heads = 14;
    const int num_kv_heads = 2;
    const int head_dim = HEAD_DIM;
    
    // Step 1: Compute QK and V projections
    std::vector<std::vector<float>> qk_proj(L, std::vector<float>(QK_DIM));
    std::vector<std::vector<float>> v_proj(L, std::vector<float>(V_DIM));
    
    reference_linear_projection(input, qk_centroids, qk_lut, qk_proj, L, HIDDEN_DIM, QK_DIM);
    reference_linear_projection(input, v_centroids, v_lut, v_proj, L, HIDDEN_DIM, V_DIM);
    
    // Step 2: Extract Q and K from the combined QK projection
    std::vector<std::vector<std::vector<float>>> q_heads(L, 
        std::vector<std::vector<float>>(num_heads, std::vector<float>(head_dim)));
    std::vector<std::vector<std::vector<float>>> k_heads(L, 
        std::vector<std::vector<float>>(num_kv_heads, std::vector<float>(head_dim)));
    std::vector<std::vector<std::vector<float>>> v_heads(L, 
        std::vector<std::vector<float>>(num_kv_heads, std::vector<float>(head_dim)));
    
    // Extract from QK projection: k[0], q[0:6], k[1], q[7:13]
    for (int i = 0; i < L; i++) {
        // k[0]
        for (int d = 0; d < head_dim; d++) {
            k_heads[i][0][d] = qk_proj[i][d];
        }
        // q[0:6]
        for (int h = 0; h < 7; h++) {
            for (int d = 0; d < head_dim; d++) {
                q_heads[i][h][d] = qk_proj[i][head_dim + h * head_dim + d];
            }
        }
        // k[1]
        for (int d = 0; d < head_dim; d++) {
            k_heads[i][1][d] = qk_proj[i][head_dim + 7 * head_dim + d];
        }
        // q[7:13]
        for (int h = 7; h < 14; h++) {
            for (int d = 0; d < head_dim; d++) {
                q_heads[i][h][d] = qk_proj[i][head_dim + 7 * head_dim + head_dim + (h - 7) * head_dim + d];
            }
        }
    }
    
    // Extract V heads from V projection
    for (int i = 0; i < L; i++) {
        for (int h = 0; h < num_kv_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                v_heads[i][h][d] = v_proj[i][h * head_dim + d];
            }
        }
    }
    
    // Step 3: Apply RoPE to Q and K heads
    for (int h = 0; h < num_heads; h++) {
        std::vector<std::vector<float>> q_tensor(L, std::vector<float>(head_dim));
        std::vector<std::vector<float>> q_rope_out(L, std::vector<float>(head_dim));
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < head_dim; d++) {
                q_tensor[i][d] = q_heads[i][h][d];
            }
        }
        apply_rotary_pos_emb_ref(q_tensor, q_rope_out, cos_table, sin_table, L, head_dim);
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < head_dim; d++) {
                q_heads[i][h][d] = q_rope_out[i][d];
            }
        }
    }
    
    for (int h = 0; h < num_kv_heads; h++) {
        std::vector<std::vector<float>> k_tensor(L, std::vector<float>(head_dim));
        std::vector<std::vector<float>> k_rope_out(L, std::vector<float>(head_dim));
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < head_dim; d++) {
                k_tensor[i][d] = k_heads[i][h][d];
            }
        }
        apply_rotary_pos_emb_ref(k_tensor, k_rope_out, cos_table, sin_table, L, head_dim);
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < head_dim; d++) {
                k_heads[i][h][d] = k_rope_out[i][d];
            }
        }
    }
    
    // Step 4: Compute attention for each head group
    std::vector<std::vector<std::vector<float>>> attn_output(L, 
        std::vector<std::vector<float>>(num_heads, std::vector<float>(head_dim)));
    
    int heads_per_kv_head = num_heads / num_kv_heads;
    
    for (int kv_h = 0; kv_h < num_kv_heads; kv_h++) {
        for (int rel_h = 0; rel_h < heads_per_kv_head; rel_h++) {
            int abs_h = kv_h * heads_per_kv_head + rel_h;
            
            // Compute attention scores
            std::vector<std::vector<float>> scores(L, std::vector<float>(L));
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    scores[i][j] = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        scores[i][j] += q_heads[i][abs_h][d] * k_heads[j][kv_h][d];
                    }
                }
            }
            
            softmax_ref(scores, L);
            
            // Apply attention to values
            for (int i = 0; i < L; i++) {
                for (int d = 0; d < head_dim; d++) {
                    attn_output[i][abs_h][d] = 0.0f;
                    for (int j = 0; j < L; j++) {
                        attn_output[i][abs_h][d] += scores[i][j] * v_heads[j][kv_h][d];
                    }
                }
            }
        }
    }
    
    // Step 5: Concatenate heads and apply output projection
    std::vector<std::vector<float>> concat_output(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                concat_output[i][h * head_dim + d] = attn_output[i][h][d];
            }
        }
    }
    
    // Apply output projection
    reference_linear_projection(concat_output, out_centroids, out_lut, output, L, HIDDEN_DIM, HIDDEN_DIM);
}

// Reference FFN implementation (SwiGLU style)
void reference_ffn(
    const std::vector<std::vector<float>>& input,  // L x HIDDEN_DIM
    const std::vector<std::vector<std::vector<float>>>& up_centroids,    // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& up_lut,          // (HIDDEN_DIM/2) x num_centroids x INTERM_DIM
    const std::vector<std::vector<std::vector<float>>>& gate_centroids,  // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& gate_lut,        // (HIDDEN_DIM/2) x num_centroids x INTERM_DIM
    const std::vector<std::vector<std::vector<float>>>& down_centroids,  // (INTERM_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& down_lut,        // (INTERM_DIM/2) x num_centroids x HIDDEN_DIM
    std::vector<std::vector<float>>& output,       // L x HIDDEN_DIM
    int L, bool use_direct_silu = false
) {
    // Up projection
    std::vector<std::vector<float>> up_output(L, std::vector<float>(INTERM_DIM));
    reference_linear_projection(input, up_centroids, up_lut, up_output, L, HIDDEN_DIM, INTERM_DIM);
    
    // Gate projection
    std::vector<std::vector<float>> gate_output(L, std::vector<float>(INTERM_DIM));
    reference_linear_projection(input, gate_centroids, gate_lut, gate_output, L, HIDDEN_DIM, INTERM_DIM);
    
    // Apply SiLU to gate projection
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < INTERM_DIM; j++) {
            if (use_direct_silu) {
                gate_output[i][j] = silu_direct(gate_output[i][j]);
            } else {
                gate_output[i][j] = silu_piecewise(gate_output[i][j]);
            }
        }
    }
    
    // Element-wise multiplication
    std::vector<std::vector<float>> intermediate(L, std::vector<float>(INTERM_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < INTERM_DIM; j++) {
            intermediate[i][j] = up_output[i][j] * gate_output[i][j];
        }
    }
    
    // Down projection
    reference_linear_projection(intermediate, down_centroids, down_lut, output, L, INTERM_DIM, HIDDEN_DIM);
}

// Reference Qwen block implementation
void reference_qwen_block(
    const std::vector<std::vector<float>>& input,  // L x HIDDEN_DIM
    const std::vector<float>& rms_weight,         // HIDDEN_DIM
    const std::vector<std::vector<std::vector<float>>>& qk_centroids,    // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& qk_lut,          // (HIDDEN_DIM/2) x num_centroids x QK_DIM
    const std::vector<std::vector<std::vector<float>>>& v_centroids,     // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& v_lut,           // (HIDDEN_DIM/2) x num_centroids x V_DIM
    const std::vector<std::vector<std::vector<float>>>& out_proj_centroids,   // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& out_proj_lut,         // (HIDDEN_DIM/2) x num_centroids x HIDDEN_DIM
    const std::vector<std::vector<float>>& sin_table,  // L x HEAD_DIM
    const std::vector<std::vector<float>>& cos_table,  // L x HEAD_DIM
    const std::vector<std::vector<std::vector<float>>>& up_centroids,    // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& up_lut,          // (HIDDEN_DIM/2) x num_centroids x INTERM_DIM
    const std::vector<std::vector<std::vector<float>>>& gate_centroids,  // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& gate_lut,        // (HIDDEN_DIM/2) x num_centroids x INTERM_DIM
    const std::vector<std::vector<std::vector<float>>>& down_centroids,  // (INTERM_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<float>>>& down_lut,        // (INTERM_DIM/2) x num_centroids x HIDDEN_DIM
    std::vector<std::vector<float>>& output,       // L x HIDDEN_DIM
    int L
) {
    // Step 1: First residual connection and RMS norm (input -> attention)
    std::vector<std::vector<float>> norm_attn_input(L, std::vector<float>(HIDDEN_DIM));
    reference_rms_norm(input, rms_weight, norm_attn_input, L);
    
    // Step 2: Attention block
    std::vector<std::vector<float>> attn_output(L, std::vector<float>(HIDDEN_DIM));
    reference_attention_block(norm_attn_input, qk_centroids, qk_lut, v_centroids, v_lut,
                             out_proj_centroids, out_proj_lut, sin_table, cos_table, attn_output, L);
    
    // Step 3: Add residual connection after attention
    std::vector<std::vector<float>> residual_after_attn(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            residual_after_attn[i][j] = input[i][j] + attn_output[i][j];
        }
    }
    
    // Step 4: Second RMS norm (residual -> FFN)
    std::vector<std::vector<float>> norm_ffn_input(L, std::vector<float>(HIDDEN_DIM));
    reference_rms_norm(residual_after_attn, rms_weight, norm_ffn_input, L);
    
    // Step 5: FFN block
    std::vector<std::vector<float>> ffn_output(L, std::vector<float>(HIDDEN_DIM));
    reference_ffn(norm_ffn_input, up_centroids, up_lut, gate_centroids, gate_lut,
                  down_centroids, down_lut, ffn_output, L, false);
    
    // Step 6: Final residual connection
    std::vector<std::vector<float>> final_residual(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            final_residual[i][j] = residual_after_attn[i][j] + ffn_output[i][j];
        }
    }
    
    // Step 7: Final RMS normalization
    reference_rms_norm(final_residual, rms_weight, output, L);
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters
    const int L = 32;              // Sequence length
    const int num_centroids = 64;   // Number of centroids per position
    const int vector_dim = 2;       // Dimension of each centroid
    
    std::cout << "Testing Qwen Block kernel with:" << std::endl;
    std::cout << "  L (sequence length): " << L << std::endl;
    std::cout << "  Hidden dimension: " << HIDDEN_DIM << std::endl;
    std::cout << "  Intermediate dimension: " << INTERM_DIM << std::endl;
    std::cout << "  QK dimension: " << QK_DIM << " (k[0]:64 + q[0:6]:448 + k[1]:64 + q[7:13]:448)" << std::endl;
    std::cout << "  V dimension: " << V_DIM << " (v[0]:64 + v[1]:64)" << std::endl;
    std::cout << "  Head dimension: " << HEAD_DIM << std::endl;
    std::cout << "  Number of centroids per position: " << num_centroids << std::endl;
    std::cout << "  Vector dimension: " << vector_dim << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> centroid_dis(-2.0f, 2.0f);
    std::uniform_real_distribution<float> lut_dis(-0.2f, 0.2f);
    std::uniform_real_distribution<float> input_dis(-2.0f, 2.0f);
    std::uniform_real_distribution<float> weight_dis(0.5f, 1.5f);
    
    // Generate random input
    std::vector<std::vector<float>> input(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            input[i][j] = input_dis(gen);
        }
    }
    
    // Pack input into hardware format - input now reads L*(HIDDEN_DIM/16) vec_t<float,16> elements
    // Following attention_block_tb and ffn_tb output format: for each 16-sequence chunk, then for each dimension
    std::vector<tapa::vec_t<float, 16>> input_hw((L * HIDDEN_DIM) / 16);
    int vec_idx = 0;
    for (int i = 0; i < (L / 16); i++) {                    // For each 16-sequence chunk
        for (int j = 0; j < HIDDEN_DIM; j++) {              // For each dimension
            tapa::vec_t<float, 16> vec;
            for (int k = 0; k < 16; k++) {                  // 16 sequences in this chunk
                vec[k] = input[i * 16 + k][j];
            }
            input_hw[vec_idx++] = vec;
        }
    }
    
    // Generate RMS normalization weights
    std::cout << "Generating RMS normalization weights..." << std::endl;
    std::vector<float> rms_weight(HIDDEN_DIM);
    for (int i = 0; i < HIDDEN_DIM; i++) {
        rms_weight[i] = weight_dis(gen);
    }
    
    // Pack RMS weights into hardware format
    std::vector<tapa::vec_t<float, 16>> rms_weight_hw(HIDDEN_DIM / 16);
    for (int i = 0; i < HIDDEN_DIM / 16; i++) {
        for (int j = 0; j < 16; j++) {
            rms_weight_hw[i][j] = rms_weight[i * 16 + j];
        }
    }
    
    // Generate centroids and LUTs for QK projection
    std::cout << "Generating QK projection centroids and LUTs..." << std::endl;
    int qk_in_size = HIDDEN_DIM_DIV_2;
    std::vector<std::vector<std::vector<float>>> qk_centroids(qk_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(vector_dim)));
    std::vector<std::vector<std::vector<float>>> qk_lut(qk_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(QK_DIM)));
    
    for (int pos = 0; pos < qk_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                qk_centroids[pos][i][j] = centroid_dis(gen);
            }
            for (int j = 0; j < QK_DIM; j++) {
                qk_lut[pos][i][j] = lut_dis(gen);
            }
        }
    }
    
    // Generate centroids and LUTs for V projection
    std::cout << "Generating V projection centroids and LUTs..." << std::endl;
    std::vector<std::vector<std::vector<float>>> v_centroids(qk_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(vector_dim)));
    std::vector<std::vector<std::vector<float>>> v_lut(qk_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(V_DIM)));
    
    for (int pos = 0; pos < qk_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                v_centroids[pos][i][j] = centroid_dis(gen);
            }
            for (int j = 0; j < V_DIM; j++) {
                v_lut[pos][i][j] = lut_dis(gen);
            }
        }
    }
    
    // Generate centroids and LUTs for output projection
    std::cout << "Generating output projection centroids and LUTs..." << std::endl;
    std::vector<std::vector<std::vector<float>>> out_proj_centroids(qk_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(vector_dim)));
    std::vector<std::vector<std::vector<float>>> out_proj_lut(qk_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(HIDDEN_DIM)));
    
    for (int pos = 0; pos < qk_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                out_proj_centroids[pos][i][j] = centroid_dis(gen);
            }
            for (int j = 0; j < HIDDEN_DIM; j++) {
                out_proj_lut[pos][i][j] = lut_dis(gen);
            }
        }
    }
    
    // Generate centroids and LUTs for up projection
    std::cout << "Generating up projection centroids and LUTs..." << std::endl;
    int up_in_size = HIDDEN_DIM_DIV_2;
    std::vector<std::vector<std::vector<float>>> up_centroids(up_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(vector_dim)));
    std::vector<std::vector<std::vector<float>>> up_lut(up_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(INTERM_DIM)));
    
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                up_centroids[pos][i][j] = centroid_dis(gen);
            }
            for (int j = 0; j < INTERM_DIM; j++) {
                up_lut[pos][i][j] = lut_dis(gen);
            }
        }
    }
    
    // Generate centroids and LUTs for gate projection
    std::cout << "Generating gate projection centroids and LUTs..." << std::endl;
    std::vector<std::vector<std::vector<float>>> gate_centroids(up_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(vector_dim)));
    std::vector<std::vector<std::vector<float>>> gate_lut(up_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(INTERM_DIM)));
    
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                gate_centroids[pos][i][j] = centroid_dis(gen);
            }
            for (int j = 0; j < INTERM_DIM; j++) {
                gate_lut[pos][i][j] = lut_dis(gen);
            }
        }
    }
    
    // Generate centroids and LUTs for down projection
    std::cout << "Generating down projection centroids and LUTs..." << std::endl;
    int down_in_size = INTERM_DIM_DIV_2;
    std::vector<std::vector<std::vector<float>>> down_centroids(down_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(vector_dim)));
    std::vector<std::vector<std::vector<float>>> down_lut(down_in_size,
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(HIDDEN_DIM)));
    
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                down_centroids[pos][i][j] = centroid_dis(gen);
            }
            for (int j = 0; j < HIDDEN_DIM; j++) {
                down_lut[pos][i][j] = lut_dis(gen);
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
            // Each frequency applies to two consecutive dimensions
            sin_table[pos][i] = std::sin(angle);
            cos_table[pos][i] = std::cos(angle);
            sin_table[pos][i + HEAD_DIM / 2] = std::sin(angle);
            cos_table[pos][i + HEAD_DIM / 2] = std::cos(angle);
        }
    }
    
    // Pack centroids into hardware format
    std::cout << "Packing centroids into hardware format..." << std::endl;
    
    // QK centroids
    std::vector<tapa::vec_t<float, 16>> qk_centroid_hw(qk_in_size * 8);
    for (int pos = 0; pos < qk_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            int vec_index = pos * 8 + i / 8;
            int element_index = (i % 8) * 2;
            qk_centroid_hw[vec_index][element_index] = qk_centroids[pos][i][0];
            qk_centroid_hw[vec_index][element_index + 1] = qk_centroids[pos][i][1];
        }
    }
    
    // V centroids
    std::vector<tapa::vec_t<float, 16>> v_centroid_hw(qk_in_size * 8);
    for (int pos = 0; pos < qk_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            int vec_index = pos * 8 + i / 8;
            int element_index = (i % 8) * 2;
            v_centroid_hw[vec_index][element_index] = v_centroids[pos][i][0];
            v_centroid_hw[vec_index][element_index + 1] = v_centroids[pos][i][1];
        }
    }
    
    // Output projection centroids
    std::vector<tapa::vec_t<float, 16>> out_proj_centroid_hw(qk_in_size * 8);
    for (int pos = 0; pos < qk_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            int vec_index = pos * 8 + i / 8;
            int element_index = (i % 8) * 2;
            out_proj_centroid_hw[vec_index][element_index] = out_proj_centroids[pos][i][0];
            out_proj_centroid_hw[vec_index][element_index + 1] = out_proj_centroids[pos][i][1];
        }
    }
    
    // Up centroids
    std::vector<tapa::vec_t<float, 16>> up_centroid_hw(up_in_size * 8);
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            int vec_index = pos * 8 + i / 8;
            int element_index = (i % 8) * 2;
            up_centroid_hw[vec_index][element_index] = up_centroids[pos][i][0];
            up_centroid_hw[vec_index][element_index + 1] = up_centroids[pos][i][1];
        }
    }
    
    // Gate centroids
    std::vector<tapa::vec_t<float, 16>> gate_centroid_hw(up_in_size * 8);
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            int vec_index = pos * 8 + i / 8;
            int element_index = (i % 8) * 2;
            gate_centroid_hw[vec_index][element_index] = gate_centroids[pos][i][0];
            gate_centroid_hw[vec_index][element_index + 1] = gate_centroids[pos][i][1];
        }
    }
    
    // Down centroids
    std::vector<tapa::vec_t<float, 16>> down_centroid_hw(down_in_size * 8);
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            int vec_index = pos * 8 + i / 8;
            int element_index = (i % 8) * 2;
            down_centroid_hw[vec_index][element_index] = down_centroids[pos][i][0];
            down_centroid_hw[vec_index][element_index + 1] = down_centroids[pos][i][1];
        }
    }
    
    // Pack LUTs into hardware format
    std::cout << "Packing LUTs into hardware format..." << std::endl;
    
    // QK LUT
    int qk_lut_elements = qk_in_size * num_centroids * QK_DIM;
    int qk_lut_vectors = qk_lut_elements / 16;
    std::vector<tapa::vec_t<float, 16>> qk_lut_hw(qk_lut_vectors);
    for (int pos = 0; pos < qk_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < QK_DIM; j++) {
                int flat_index = pos * num_centroids * QK_DIM + i * QK_DIM + j;
                int vec_index = flat_index / 16;
                int element_index = flat_index % 16;
                qk_lut_hw[vec_index][element_index] = qk_lut[pos][i][j];
            }
        }
    }
    
    // V LUT
    int v_lut_elements = qk_in_size * num_centroids * V_DIM;
    int v_lut_vectors = v_lut_elements / 16;
    std::vector<tapa::vec_t<float, 16>> v_lut_hw(v_lut_vectors);
    for (int pos = 0; pos < qk_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < V_DIM; j++) {
                int flat_index = pos * num_centroids * V_DIM + i * V_DIM + j;
                int vec_index = flat_index / 16;
                int element_index = flat_index % 16;
                v_lut_hw[vec_index][element_index] = v_lut[pos][i][j];
            }
        }
    }
    
    // Output projection LUT
    int out_proj_lut_elements = qk_in_size * num_centroids * HIDDEN_DIM;
    int out_proj_lut_vectors = out_proj_lut_elements / 16;
    std::vector<tapa::vec_t<float, 16>> out_proj_lut_hw(out_proj_lut_vectors);
    for (int pos = 0; pos < qk_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < HIDDEN_DIM; j++) {
                int flat_index = pos * num_centroids * HIDDEN_DIM + i * HIDDEN_DIM + j;
                int vec_index = flat_index / 16;
                int element_index = flat_index % 16;
                out_proj_lut_hw[vec_index][element_index] = out_proj_lut[pos][i][j];
            }
        }
    }
    
    // Up LUT
    int up_lut_elements = up_in_size * num_centroids * INTERM_DIM;
    int up_lut_vectors = up_lut_elements / 16;
    std::vector<tapa::vec_t<float, 16>> up_lut_hw(up_lut_vectors);
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < INTERM_DIM; j++) {
                int flat_index = pos * num_centroids * INTERM_DIM + i * INTERM_DIM + j;
                int vec_index = flat_index / 16;
                int element_index = flat_index % 16;
                up_lut_hw[vec_index][element_index] = up_lut[pos][i][j];
            }
        }
    }
    
    // Gate LUT
    std::vector<tapa::vec_t<float, 16>> gate_lut_hw(up_lut_vectors);
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < INTERM_DIM; j++) {
                int flat_index = pos * num_centroids * INTERM_DIM + i * INTERM_DIM + j;
                int vec_index = flat_index / 16;
                int element_index = flat_index % 16;
                gate_lut_hw[vec_index][element_index] = gate_lut[pos][i][j];
            }
        }
    }
    
    // Down LUT
    int down_lut_elements = down_in_size * num_centroids * HIDDEN_DIM;
    int down_lut_vectors = down_lut_elements / 16;
    std::vector<tapa::vec_t<float, 16>> down_lut_hw(down_lut_vectors);
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < HIDDEN_DIM; j++) {
                int flat_index = pos * num_centroids * HIDDEN_DIM + i * HIDDEN_DIM + j;
                int vec_index = flat_index / 16;
                int element_index = flat_index % 16;
                down_lut_hw[vec_index][element_index] = down_lut[pos][i][j];
            }
        }
    }
    
    // Pack RoPE tables into hardware format
    std::cout << "Packing RoPE tables into hardware format..." << std::endl;
    int rope_vectors = (L * HEAD_DIM) / 16;
    std::vector<tapa::vec_t<float, 16>> sin_hw(rope_vectors);
    std::vector<tapa::vec_t<float, 16>> cos_hw(rope_vectors);
    
    // Pack row by row, 16 elements per vector
    vec_idx = 0;
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
    std::vector<tapa::vec_t<float, 16>> layer_out_hw(output_vectors);
    std::vector<int> cycle_count_hw(1);
    
    // Compute reference results
    std::cout << "Computing reference results..." << std::endl;
    std::vector<std::vector<float>> output_ref(L, std::vector<float>(HIDDEN_DIM));
    reference_qwen_block(input, rms_weight, qk_centroids, qk_lut, v_centroids, v_lut,
                        out_proj_centroids, out_proj_lut, sin_table, cos_table,
                        up_centroids, up_lut, gate_centroids, gate_lut,
                        down_centroids, down_lut, output_ref, L);
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(qwen_block, FLAGS_bitstream,
                L,
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(input_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(rms_weight_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(qk_centroid_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(v_centroid_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(out_proj_centroid_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(qk_lut_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(v_lut_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(out_proj_lut_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(sin_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(cos_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(up_centroid_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(gate_centroid_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(down_centroid_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(up_lut_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(gate_lut_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(down_lut_hw),
                tapa::write_only_mmap<tapa::vec_t<float, 16>>(layer_out_hw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Convert hardware output from tapa::vec_t<float, 16> vectors to 2D array
    // Hardware linear_out_writer writes in same format as input: for each 16-sequence chunk, then for each dimension
    std::cout << "Converting hardware output..." << std::endl;
    std::vector<std::vector<float>> output_hw(L, std::vector<float>(HIDDEN_DIM));
    
    vec_idx = 0;
    for (int i = 0; i < (L / 16); i++) {                    // For each 16-sequence chunk
        for (int j = 0; j < HIDDEN_DIM; j++) {              // For each dimension
            for (int k = 0; k < 16; k++) {                  // 16 sequences in this chunk
                output_hw[i * 16 + k][j] = layer_out_hw[vec_idx][k];
            }
            vec_idx++;
        }
    }
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    float max_error = 0.0f;
    float tolerance = 1e-5f;  // Relaxed tolerance for full transformer block
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float error = std::abs(output_hw[i][j] - output_ref[i][j]);
            if (error > max_error) {
                max_error = error;
            }
            if (!isClose(output_hw[i][j], output_ref[i][j], tolerance)) {
                if (errors < 10) {  // Only print first 10 errors
                    std::cout << "Mismatch at [" << i << "][" << j << "]: "
                             << "hw=" << output_hw[i][j] << " vs ref=" << output_ref[i][j]
                             << " (error=" << error << ")" << std::endl;
                }
                errors++;
            }
        }
    }
    
    std::cout << "Maximum error: " << max_error << std::endl;
    
    if (errors == 0) {
        std::cout << "SUCCESS: All " << (L * HIDDEN_DIM) 
                 << " results match reference within tolerance!" << std::endl;
    } else {
        std::cout << "FAILURE: " << errors << " out of " << (L * HIDDEN_DIM) 
                 << " results do not match reference!" << std::endl;
    }
    
    // Print some sample results
    std::cout << "\nSample results (first sequence, first 10 outputs):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < std::min(10, HIDDEN_DIM); j++) {
        std::cout << "  [0][" << j << "]: hw=" << output_hw[0][j] 
                 << " vs ref=" << output_ref[0][j] 
                 << " (error=" << std::abs(output_hw[0][j] - output_ref[0][j]) << ")" << std::endl;
    }
    
    // Print statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total input elements: " << (L * HIDDEN_DIM) << std::endl;
    std::cout << "  Total centroids: " << ((qk_in_size + up_in_size + down_in_size) * num_centroids * 3) << std::endl;
    std::cout << "  Total LUT entries: " << (qk_lut_elements + v_lut_elements + out_proj_lut_elements + up_lut_elements + up_lut_elements + down_lut_elements) << std::endl;
    std::cout << "  Total output elements: " << (L * HIDDEN_DIM) << std::endl;
    std::cout << "  Memory usage:" << std::endl;
    std::cout << "    Input: " << (input_hw.size() * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    RMS weights: " << (rms_weight_hw.size() * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Centroids: " << ((qk_centroid_hw.size() + v_centroid_hw.size() + out_proj_centroid_hw.size() + up_centroid_hw.size() + gate_centroid_hw.size() + down_centroid_hw.size()) * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    LUTs: " << ((qk_lut_hw.size() + v_lut_hw.size() + out_proj_lut_hw.size() + up_lut_hw.size() + gate_lut_hw.size() + down_lut_hw.size()) * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    RoPE tables: " << ((sin_hw.size() + cos_hw.size()) * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Output: " << (layer_out_hw.size() * 16 * sizeof(float)) << " bytes" << std::endl;
    
    return errors == 0 ? 0 : 1;
}
