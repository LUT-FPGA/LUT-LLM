#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "rope.h"

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// Helper function to check if two floating point numbers are close
bool isClose(float a, float b, float tolerance = 1e-4) {
    return std::abs(a - b) < tolerance;
}

// Generate RoPE sin and cos embeddings with theta = 1e6
void generate_rope_embeddings(
    int seq_len,
    int head_dim,
    float theta,
    std::vector<std::vector<float>>& sin_embeddings,
    std::vector<std::vector<float>>& cos_embeddings
) {
    sin_embeddings.resize(seq_len, std::vector<float>(head_dim));
    cos_embeddings.resize(seq_len, std::vector<float>(head_dim));
    
    // Generate frequency for each dimension pair
    std::vector<float> inv_freq(head_dim / 2);
    for (int i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0f / std::pow(theta, 2.0f * i / head_dim);
    }
    
    // Generate sin and cos for each position
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float angle = pos * inv_freq[i];
            // Each frequency applies to two consecutive dimensions
            sin_embeddings[pos][i] = std::sin(angle);
            cos_embeddings[pos][i] = std::cos(angle);
            sin_embeddings[pos][i + head_dim / 2] = std::sin(angle);
            cos_embeddings[pos][i + head_dim / 2] = std::cos(angle);
        }
    }
}

// Reference implementation for RoPE embedding
void reference_rope(
    const std::vector<std::vector<float>>& input,        // [seq_len][head_dim]
    const std::vector<std::vector<float>>& sin_embeddings, // [seq_len][head_dim]
    const std::vector<std::vector<float>>& cos_embeddings, // [seq_len][head_dim]
    std::vector<std::vector<float>>& output              // [seq_len][head_dim]
) {
    int seq_len = input.size();
    int head_dim = input[0].size();
    
    output.resize(seq_len, std::vector<float>(head_dim));
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < head_dim; j++) {
            float x = input[i][j];
            float sin_val = sin_embeddings[i][j];
            float cos_val = cos_embeddings[i][j];
            
            // Apply RoPE transformation: rotate_half operation
            float x_rotated;
            if (j < head_dim / 2) {
                // First half: -x[j + head_dim/2]
                x_rotated = -input[i][j + head_dim / 2];
            } else {
                // Second half: x[j - head_dim/2]
                x_rotated = input[i][j - head_dim / 2];
            }
            
            // RoPE formula: x * cos + rotate_half(x) * sin
            output[i][j] = x * cos_val + x_rotated * sin_val;
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters
    const int L = 128;                    // Sequence length (must be multiple of 16)
    const int head_dim = HEAD_DIM;        // Head dimension (64)
    const float theta = 1e6f;             // RoPE theta parameter
    
    std::cout << "Testing RoPE kernel with:" << std::endl;
    std::cout << "  Sequence length (L): " << L << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  RoPE theta: " << theta << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    
    // Generate random input data
    std::vector<std::vector<float>> input_data(L, std::vector<float>(head_dim));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < head_dim; j++) {
            input_data[i][j] = dis(gen);
        }
    }
    
    std::cout << "Generated random input data" << std::endl;
    
    // Generate RoPE sin and cos embeddings
    std::vector<std::vector<float>> sin_embeddings, cos_embeddings;
    generate_rope_embeddings(L, head_dim, theta, sin_embeddings, cos_embeddings);
    
    std::cout << "Generated RoPE sin/cos embeddings" << std::endl;
    
    // Print sample embeddings for verification
    std::cout << "Sample embeddings for position 0:" << std::endl;
    std::cout << "  cos[0][0:4]: ";
    for (int i = 0; i < 4; i++) {
        std::cout << std::fixed << std::setprecision(6) << cos_embeddings[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  sin[0][0:4]: ";
    for (int i = 0; i < 4; i++) {
        std::cout << std::fixed << std::setprecision(6) << sin_embeddings[0][i] << " ";
    }
    std::cout << std::endl;
    
    // Print some positions to show the pattern
    std::cout << "Sample cos values for dim 0 across positions:" << std::endl;
    for (int pos = 0; pos < std::min(8, L); pos++) {
        std::cout << "  pos " << pos << ": " << std::fixed << std::setprecision(6) 
                 << cos_embeddings[pos][0] << std::endl;
    }
    
    // Pack input data for hardware
    int total_input_vectors = (L * head_dim) / 16;
    std::vector<tapa::vec_t<float, 16>> input_hw(total_input_vectors);
    std::vector<tapa::vec_t<float, 16>> sin_hw(total_input_vectors);
    std::vector<tapa::vec_t<float, 16>> cos_hw(total_input_vectors);
    
    std::cout << "Packing input data (" << total_input_vectors << " vectors)..." << std::endl;
    
    int vec_idx = 0;
    // Pack input data row by row, 16 elements per vector
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < head_dim; j += 16) {
            for (int k = 0; k < 16; k++) {
                input_hw[vec_idx][k] = input_data[i][j + k];
                sin_hw[vec_idx][k] = sin_embeddings[i][j + k];
                cos_hw[vec_idx][k] = cos_embeddings[i][j + k];
            }
            vec_idx++;
        }
    }
    
    std::cout << "Packed " << vec_idx << " vectors (expected: " << total_input_vectors << ")" << std::endl;
    
    // Allocate output arrays
    std::vector<tapa::vec_t<float, 16>> output_hw_raw(total_input_vectors);
    std::vector<int> cycle_count_hw(1);
    
    // Compute reference results
    std::cout << "Computing reference results..." << std::endl;
    std::vector<std::vector<float>> output_ref;
    reference_rope(input_data, sin_embeddings, cos_embeddings, output_ref);
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(rope, FLAGS_bitstream,
                L,
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(input_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(sin_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(cos_hw),
                tapa::write_only_mmap<tapa::vec_t<float, 16>>(output_hw_raw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Convert hardware output back to structured format
    std::cout << "Converting hardware output..." << std::endl;
    std::vector<std::vector<float>> output_hw(L, std::vector<float>(head_dim));
    
    vec_idx = 0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < head_dim; j += 16) {
            for (int k = 0; k < 16; k++) {
                output_hw[i][j + k] = output_hw_raw[vec_idx][k];
            }
            vec_idx++;
        }
    }
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    float max_error = 0.0f;
    float tolerance = 1e-3f;  // Relaxed tolerance due to floating point precision
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < head_dim; j++) {
            float diff = std::abs(output_hw[i][j] - output_ref[i][j]);
            if (diff > max_error) {
                max_error = diff;
            }
            
            if (!isClose(output_hw[i][j], output_ref[i][j], tolerance)) {
                errors++;
                if (errors <= 20) {  // Print first 20 errors
                    std::cout << "Error at [" << i << "][" << j << "]: HW=" 
                             << output_hw[i][j] << ", REF=" << output_ref[i][j] 
                             << ", diff=" << diff << std::endl;
                }
            }
        }
    }
    
    std::cout << "Maximum error: " << max_error << std::endl;
    
    if (errors == 0) {
        std::cout << "SUCCESS: All " << (L * head_dim) 
                 << " results match within tolerance!" << std::endl;
    } else {
        std::cout << "FAILURE: " << errors << " out of " << (L * head_dim) 
                 << " results don't match!" << std::endl;
    }
    
    // Print some sample results for debugging
    std::cout << "\nSample results (position 0, first 8 dimensions):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < std::min(8, head_dim); j++) {
        std::cout << "Output [0][" << j << "]: HW=" << output_hw[0][j] 
                 << ", REF=" << output_ref[0][j] 
                 << ", diff=" << std::abs(output_hw[0][j] - output_ref[0][j]) << std::endl;
    }
    
    // Test rotation property: verify that the RoPE embedding preserves relative positions
    std::cout << "\nTesting RoPE rotation property..." << std::endl;
    
    // Compute dot product between original and rotated vectors for different positions
    for (int pos1 = 0; pos1 < std::min(4, L); pos1++) {
        for (int pos2 = pos1; pos2 < std::min(pos1 + 4, L); pos2++) {
            float dot_original = 0.0f;
            float dot_rotated = 0.0f;
            
            for (int d = 0; d < head_dim; d++) {
                dot_original += input_data[pos1][d] * input_data[pos2][d];
                dot_rotated += output_ref[pos1][d] * output_ref[pos2][d];
            }
            
            std::cout << "Dot product pos(" << pos1 << "," << pos2 << "): "
                     << "original=" << std::setprecision(4) << dot_original 
                     << ", rotated=" << std::setprecision(4) << dot_rotated 
                     << ", ratio=" << std::setprecision(4) << (dot_original != 0 ? dot_rotated/dot_original : 0) 
                     << std::endl;
        }
    }
    
    // Show frequency pattern
    std::cout << "\nRoPE frequency analysis (first few dimensions):" << std::endl;
    for (int d = 0; d < std::min(8, head_dim/2); d++) {
        float freq = 1.0f / std::pow(theta, 2.0f * d / head_dim);
        std::cout << "Dimension pair " << d << ": frequency=" << std::scientific << freq 
                 << ", period=" << std::fixed << std::setprecision(1) << (2 * M_PI / freq) << std::endl;
    }
    
    // Verify orthogonality property for some positions
    std::cout << "\nTesting orthogonality after rotation:" << std::endl;
    for (int pos = 0; pos < std::min(4, L); pos++) {
        float norm_original = 0.0f;
        float norm_rotated = 0.0f;
        
        for (int d = 0; d < head_dim; d++) {
            norm_original += input_data[pos][d] * input_data[pos][d];
            norm_rotated += output_ref[pos][d] * output_ref[pos][d];
        }
        
        std::cout << "Position " << pos << " norm: original=" << std::setprecision(4) << std::sqrt(norm_original)
                 << ", rotated=" << std::setprecision(4) << std::sqrt(norm_rotated)
                 << ", preserved=" << (isClose(norm_original, norm_rotated, 1e-3) ? "YES" : "NO")
                 << std::endl;
    }
    
    // Debug: Print data packing information
    std::cout << "\nData packing debug info:" << std::endl;
    std::cout << "  Input vectors: " << total_input_vectors << std::endl;
    std::cout << "  Expected input size: " << (L * head_dim) / 16 << " vectors" << std::endl;
    std::cout << "  Each position needs: " << head_dim / 16 << " vectors" << std::endl;
    std::cout << "  Total positions: " << L << std::endl;
    
    // Print statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Input size: " << (L * head_dim) << " floats" << std::endl;
    std::cout << "  Sin/Cos embeddings: " << (2 * L * head_dim) << " floats" << std::endl;
    std::cout << "  Output size: " << (L * head_dim) << " floats" << std::endl;
    std::cout << "  Memory bandwidth:" << std::endl;
    std::cout << "    Input: " << (L * head_dim * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Sin/Cos: " << (2 * L * head_dim * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Output: " << (L * head_dim * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Total: " << (4 * L * head_dim * sizeof(float)) << " bytes" << std::endl;
    
    // Performance analysis
    if (cycle_count_hw[0] > 0) {
        float throughput = (float)(L * head_dim) / cycle_count_hw[0];
        std::cout << "  Throughput: " << std::setprecision(2) << throughput << " elements/cycle" << std::endl;
        std::cout << "  Theoretical peak (16 elements/cycle): " << (throughput / 16.0f * 100) << "% efficiency" << std::endl;
    }
    
    return errors == 0 ? 0 : 1;
}
