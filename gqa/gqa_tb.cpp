#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "gqa.h"

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// Helper function to check if two floating point numbers are close
bool isClose(float a, float b, float tolerance = 1e-4) {
    return std::abs(a - b) < tolerance;
}

// Reference implementation for softmax with causal mask (matching hardware behavior)
void reference_softmax(const std::vector<std::vector<float>>& qk_scores,
                      std::vector<std::vector<float>>& attention_weights,
                      int seq_len) {
    for (int i = 0; i < seq_len; i++) {
        float sum = 0.0f;
        
        // Compute exp and sum (hardware doesn't do max subtraction for stability)
        for (int j = 0; j < seq_len; j++) {
            if (i >= j) {  // causal mask: row i can attend to column j if i >= j
                float scaled_score = qk_scores[i][j] * 0.125f;  // Apply same scaling as hardware
                float exp_val = std::exp(scaled_score);
                attention_weights[i][j] = exp_val;
                sum += exp_val;
            } else {
                attention_weights[i][j] = 0.0f;  // masked positions
            }
        }
        
        // Normalize
        for (int j = 0; j < seq_len; j++) {
            if (i >= j) {
                attention_weights[i][j] /= sum;
            }
        }
    }
}

// Reference implementation for Grouped Query Attention
void reference_gqa(
    const std::vector<std::vector<std::vector<float>>>& k_matrices,  // [2][seq_len][head_dim]
    const std::vector<std::vector<std::vector<float>>>& v_matrices,  // [2][seq_len][head_dim]
    const std::vector<std::vector<std::vector<float>>>& q_matrices,  // [14][seq_len][head_dim]
    std::vector<std::vector<std::vector<float>>>& output,           // [14][seq_len][head_dim]
    int seq_len
) {
    const int num_groups = 2;
    const int heads_per_group = 7;
    const int head_dim = HEAD_DIM;
    
    // Process each group
    for (int group = 0; group < num_groups; group++) {
        const auto& k = k_matrices[group];  // [seq_len][head_dim]
        const auto& v = v_matrices[group];  // [seq_len][head_dim]
        
        // Process each head in the group
        for (int head_in_group = 0; head_in_group < heads_per_group; head_in_group++) {
            int head_idx = group * heads_per_group + head_in_group;
            
            const auto& q = q_matrices[head_idx];  // [seq_len][head_dim]
            
            // Compute Q @ K^T
            std::vector<std::vector<float>> qk_scores(seq_len, std::vector<float>(seq_len, 0.0f));
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    for (int d = 0; d < head_dim; d++) {
                        qk_scores[i][j] += q[i][d] * k[j][d];
                    }
                }
            }
            
            // Apply softmax with causal mask
            std::vector<std::vector<float>> attention_weights(seq_len, std::vector<float>(seq_len, 0.0f));
            reference_softmax(qk_scores, attention_weights, seq_len);

            
            // Compute attention_weights @ V with detailed logging
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float accumulated_value = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        accumulated_value += attention_weights[i][j] * v[j][d];
                    }
                    output[head_idx][i][d] = accumulated_value;
                    
                }
            }
            
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters - start with smaller size for debugging
    const int L = 128;  // Sequence length (must be multiple of 16)
    const int num_groups = 2;
    const int heads_per_group = 7;
    const int total_heads = 14;
    const int head_dim = HEAD_DIM;  // 64
    const int hidden_dim = HIDDEN_DIM;  // 896
    const int qkv_dim = QKV_DIM;  // 896 + 64 * 4 = 1152
    
    std::cout << "Testing GQA kernel with:" << std::endl;
    std::cout << "  Sequence length (L): " << L << std::endl;
    std::cout << "  Number of groups: " << num_groups << std::endl;
    std::cout << "  Heads per group: " << heads_per_group << std::endl;
    std::cout << "  Total heads: " << total_heads << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Hidden dimension: " << hidden_dim << std::endl;
    std::cout << "  QKV dimension: " << qkv_dim << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    
    // Generate random K, V, Q matrices
    // K matrices: [num_groups][seq_len][head_dim]
    std::vector<std::vector<std::vector<float>>> k_matrices(num_groups, 
                                                            std::vector<std::vector<float>>(L, std::vector<float>(head_dim)));
    
    // V matrices: [num_groups][seq_len][head_dim]  
    std::vector<std::vector<std::vector<float>>> v_matrices(num_groups,
                                                            std::vector<std::vector<float>>(L, std::vector<float>(head_dim)));
    
    // Q matrices: [total_heads][seq_len][head_dim]
    std::vector<std::vector<std::vector<float>>> q_matrices(total_heads,
                                                            std::vector<std::vector<float>>(L, std::vector<float>(head_dim)));
    
    // Fill with random values
    for (int g = 0; g < num_groups; g++) {
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < head_dim; d++) {
                k_matrices[g][i][d] = dis(gen);
                v_matrices[g][i][d] = dis(gen);
            }
        }
    }
    
    for (int h = 0; h < total_heads; h++) {
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < head_dim; d++) {
                q_matrices[h][i][d] = dis(gen);
            }
        }
    }
    
    std::cout << "Generated random K, V, Q matrices" << std::endl;
    
    // Pack input data according to the format expected by the hardware
    // Looking at input_reader, it reads (L * QKV_DIM) >> 4 vec_t<float, 16> elements
    // This equals (L * 1152) / 16 = 72L vectors for L sequence length
    
    // Based on the gemm_gqa function, the data is read in the following order:
    // For each group: K matrix (by columns), V matrix (by rows), then Q matrices for that group
    
    int total_input_vectors = (L * qkv_dim) / 16;  // Corrected: >> 4 means divide by 16
    std::vector<tapa::vec_t<float, 16>> input_hw(total_input_vectors);
    
    std::cout << "Packing input data (total vectors: " << total_input_vectors << ")..." << std::endl;
    
    int vec_idx = 0;
    
    // Pack data for both groups
    for (int g = 0; g < num_groups; g++) {
        // Pack K matrix - the hardware reads it column by column
        // load_k: for (int i = 0; i < HEAD_DIM; i++) for (int j = 0; j < (L >> 4); j++)
        for (int col = 0; col < head_dim; col++) {
            for (int row_chunk = 0; row_chunk < (L / 16); row_chunk++) {
                for (int elem = 0; elem < 16; elem++) {
                    int row = row_chunk * 16 + elem;
                    input_hw[vec_idx][elem] = k_matrices[g][row][col];
                }
                vec_idx++;
            }
        }
        
        // Pack V matrix - the hardware reads it row by row
        // load_v: for (int i = 0; i < L; i++) for (int j = 0; j < (HEAD_DIM >> 4); j++)
        for (int row = 0; row < L; row++) {
            for (int col_chunk = 0; col_chunk < (head_dim / 16); col_chunk++) {
                for (int elem = 0; elem < 16; elem++) {
                    int col = col_chunk * 16 + elem;
                    input_hw[vec_idx][elem] = v_matrices[g][row][col];
                }
                vec_idx++;
            }
        }
        
        // Pack Q matrices for this group (7 matrices)
        // The Q matrices are read during the QK computation
        for (int h = 0; h < heads_per_group; h++) {
            int head_idx = g * heads_per_group + h;
            for (int col = 0; col < head_dim; col++) {
                for (int row_chunk = 0; row_chunk < (L / 16); row_chunk++) {
                    for (int elem = 0; elem < 16; elem++) {
                        int row = row_chunk * 16 + elem;
                        input_hw[vec_idx][elem] = q_matrices[head_idx][row][col];
                    }
                    vec_idx++;
                }
            }
        }
    }
    
    std::cout << "Packed " << vec_idx << " vectors (expected: " << total_input_vectors << ")" << std::endl;
    
    // Fill remaining vectors with zeros if needed
    while (vec_idx < total_input_vectors) {
        for (int elem = 0; elem < 16; elem++) {
            input_hw[vec_idx][elem] = 0.0f;
        }
        vec_idx++;
    }
    
    // Allocate output arrays
    int total_output_floats = L * hidden_dim;
    int num_output_vectors = total_output_floats / 16;
    std::vector<tapa::vec_t<float, 16>> output_hw_raw(num_output_vectors);
    std::vector<int> cycle_count_hw(1);
    
    std::cout << "Expected output: " << total_output_floats << " floats in " 
             << num_output_vectors << " vectors" << std::endl;
    
    // Compute reference results
    std::cout << "Computing reference results..." << std::endl;
    std::vector<std::vector<std::vector<float>>> output_ref(total_heads,
                                                            std::vector<std::vector<float>>(L, std::vector<float>(head_dim)));
    
    reference_gqa(k_matrices, v_matrices, q_matrices, output_ref, L);
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(gqa, FLAGS_bitstream,
                L,
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(input_hw),
                tapa::write_only_mmap<tapa::vec_t<float, 16>>(output_hw_raw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Convert hardware output back to structured format
    std::cout << "Converting hardware output..." << std::endl;
    std::vector<std::vector<std::vector<float>>> output_hw(total_heads,
                                                           std::vector<std::vector<float>>(L, std::vector<float>(head_dim)));
    
    // The hardware outputs results grouped by attention heads
    // Based on the write pattern in gemm_gqa, output is organized as:
    // For each group, for each head in group, for each 16-row batch, for each dimension chunk
    int hw_vec_idx = 0;
    
    for (int g = 0; g < num_groups; g++) {
        for (int h = 0; h < heads_per_group; h++) {
            int head_idx = g * heads_per_group + h;
            
            // Process in 16-row batches
            for (int row_batch = 0; row_batch < (L / 16); row_batch++) {
                // For each row in the batch
                for (int row_in_batch = 0; row_in_batch < 16; row_in_batch++) {
                    int seq_pos = row_batch * 16 + row_in_batch;
                    
                    // Process dimension in chunks of 16
                    for (int dim_chunk = 0; dim_chunk < (head_dim / 16); dim_chunk++) {
                        for (int elem = 0; elem < 16; elem++) {
                            int dim = dim_chunk * 16 + elem;
                            output_hw[head_idx][seq_pos][dim] = output_hw_raw[hw_vec_idx][elem];
                        }
                        hw_vec_idx++;
                    }
                }
            }
        }
    }
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    float max_error = 0.0f;
    float tolerance = 1e-3f;  // Relaxed tolerance due to accumulated floating point errors
    
    for (int h = 0; h < total_heads; h++) {
        for (int i = 0; i < L; i++) {
            for (int d = 0; d < head_dim; d++) {
                float diff = std::abs(output_hw[h][i][d] - output_ref[h][i][d]);
                if (diff > max_error) {
                    max_error = diff;
                }
                
                if (!isClose(output_hw[h][i][d], output_ref[h][i][d], tolerance)) {
                    errors++;
                    if (errors <= 20) {  // Print first 20 errors
                        std::cout << "Error at head[" << h << "][" << i << "][" << d << "]: HW=" 
                                 << output_hw[h][i][d] << ", REF=" << output_ref[h][i][d] 
                                 << ", diff=" << diff << std::endl;
                    }
                }
            }
        }
    }
    
    std::cout << "Maximum error: " << max_error << std::endl;
    
    if (errors == 0) {
        std::cout << "SUCCESS: All " << (total_heads * L * head_dim) 
                 << " results match within tolerance!" << std::endl;
    } else {
        std::cout << "FAILURE: " << errors << " out of " << (total_heads * L * head_dim) 
                 << " results don't match!" << std::endl;
    }
    
    // Print some sample results for debugging
    std::cout << "\nSample results (head 0, position 0, first 8 dimensions):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int d = 0; d < std::min(8, head_dim); d++) {
        std::cout << "Output [0][0][" << d << "]: HW=" << output_hw[0][0][d] 
                 << ", REF=" << output_ref[0][0][d] 
                 << ", diff=" << std::abs(output_hw[0][0][d] - output_ref[0][0][d]) << std::endl;
    }
    
    // Test attention pattern (should be causal)
    std::cout << "\nTesting causal attention pattern for first head, first few positions:" << std::endl;
    
    // Compute a sample attention pattern to verify causality
    const auto& k = k_matrices[0];  // Use first group's K
    const auto& q = q_matrices[0];  // Use first head's Q
    
    std::cout << "Sample QK scores (before softmax) for head 0:" << std::endl;
    for (int i = 0; i < std::min(4, L); i++) {
        std::cout << "Query " << i << ": ";
        for (int j = 0; j < std::min(8, L); j++) {
            float qk_score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                qk_score += q[i][d] * k[j][d];
            }
            if (j <= i) {
                std::cout << std::setw(8) << qk_score << " ";
            } else {
                std::cout << "  masked  ";
            }
        }
        std::cout << std::endl;
    }
    
    // Show sample attention weights after softmax
    std::vector<std::vector<float>> sample_qk_scores(4, std::vector<float>(8, 0.0f));
    std::vector<std::vector<float>> sample_attn_weights(4, std::vector<float>(8, 0.0f));
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            for (int d = 0; d < head_dim; d++) {
                sample_qk_scores[i][j] += q[i][d] * k[j][d];
            }
        }
    }
    
    reference_softmax(sample_qk_scores, sample_attn_weights, 4);
    
    std::cout << "\nSample attention weights (after softmax) for head 0:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "Query " << i << ": ";
        float sum = 0.0f;
        for (int j = 0; j < std::min(8, L); j++) {
            if (i >= j) {
                std::cout << std::setw(8) << std::setprecision(4) << sample_attn_weights[i][j] << " ";
                sum += sample_attn_weights[i][j];
            } else {
                std::cout << "  0.0000 ";
            }
        }
        std::cout << " (sum=" << std::setprecision(4) << sum << ")" << std::endl;
    }

    // Debug: Print data packing information
    std::cout << "\nData packing debug info:" << std::endl;
    std::cout << "  Input vectors created: " << total_input_vectors << std::endl;
    std::cout << "  Expected input size: " << (L * qkv_dim) / 16 << " vectors" << std::endl;
    std::cout << "  Output vectors: " << num_output_vectors << std::endl;
    std::cout << "  Expected output size: " << (L * hidden_dim) / 16 << " vectors" << std::endl;
    
    // Print statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total parameters: " << std::endl;
    std::cout << "    K matrices: " << (num_groups * L * head_dim) << " floats" << std::endl;
    std::cout << "    V matrices: " << (num_groups * L * head_dim) << " floats" << std::endl;
    std::cout << "    Q matrices: " << (total_heads * L * head_dim) << " floats" << std::endl;
    std::cout << "  Total input size: " << (total_input_vectors * 16) << " floats" << std::endl;
    std::cout << "  Total output size: " << total_output_floats << " floats" << std::endl;
    std::cout << "  Memory bandwidth utilization:" << std::endl;
    std::cout << "    Input: " << (total_input_vectors * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Output: " << (total_output_floats * sizeof(float)) << " bytes" << std::endl;
    
    return errors == 0 ? 0 : 1;
}
