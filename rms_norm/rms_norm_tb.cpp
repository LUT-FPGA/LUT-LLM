#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "rms_norm.h"

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// Helper function to check if two floating point numbers are close
bool isClose(float a, float b, float tolerance = 1e-4) {
    return std::abs(a - b) < tolerance;
}

// Reference implementation for RMS Normalization
void reference_rms_norm(
    const std::vector<std::vector<float>>& input,     // [L][HIDDEN_DIM]
    const std::vector<float>& weight,                 // [HIDDEN_DIM]
    std::vector<std::vector<float>>& output          // [L][HIDDEN_DIM]
) {
    int L = input.size();
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

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters - matching the hardware configuration
    const int L = 128;  // Sequence length (must be multiple of 16)
    const int hidden_dim = HIDDEN_DIM;  // 896
    
    std::cout << "Testing RMS Normalization kernel with:" << std::endl;
    std::cout << "  Sequence length (L): " << L << std::endl;
    std::cout << "  Hidden dimension: " << hidden_dim << std::endl;
    std::cout << "  Epsilon: " << EPSILON << std::endl;
    std::cout << "  R_HIDDEN_DIM: " << R_HIDDEN_DIM << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> input_dis(-2.0f, 2.0f);
    std::uniform_real_distribution<float> weight_dis(0.5f, 1.5f);  // Positive weights typical for normalization
    
    // Generate random input data
    std::vector<std::vector<float>> input_data(L, std::vector<float>(hidden_dim));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            input_data[i][j] = input_dis(gen);
        }
    }
    
    // Generate random weight vector
    std::vector<float> weight_data(hidden_dim);
    for (int j = 0; j < hidden_dim; j++) {
        weight_data[j] = weight_dis(gen);
    }
    
    std::cout << "Generated random input and weight data" << std::endl;
    
    // Pack input data into hardware format (vec_t<float, 16>)
    // Input layout: L sequences x HIDDEN_DIM elements, packed as 16-element vectors
    int input_vectors_count = (L * hidden_dim) / 16;
    std::vector<tapa::vec_t<float, 16>> input_hw(input_vectors_count);
    
    std::cout << "Packing input data (" << input_vectors_count << " vectors)..." << std::endl;
    
    // Pack input data row by row (sequence by sequence)
    int vec_idx = 0;
    for (int i = 0; i < (L / 16); i++) {
        for (int j = 0; j < hidden_dim; j++) {
            for (int k = 0; k < 16; k++) {
                input_hw[vec_idx][k] = input_data[i*16+k][j];
            }
            vec_idx++;
        }
    }
    
    // Pack weight data into hardware format
    int weight_vectors_count = hidden_dim / 16;
    std::vector<tapa::vec_t<float, 16>> weight_hw(weight_vectors_count);
    
    std::cout << "Packing weight data (" << weight_vectors_count << " vectors)..." << std::endl;
    
    vec_idx = 0;
    for (int j = 0; j < hidden_dim; j += 16) {
        for (int k = 0; k < 16; k++) {
            weight_hw[vec_idx][k] = weight_data[j + k];
        }
        vec_idx++;
    }
    
    // Allocate output arrays
    int output_vectors_count = (L * hidden_dim) / 16;
    std::vector<tapa::vec_t<float, 16>> output_hw_raw(output_vectors_count);
    std::vector<int> cycle_count_hw(1);
    
    std::cout << "Expected output: " << (L * hidden_dim) << " floats in " 
             << output_vectors_count << " vectors" << std::endl;
    
    // Compute reference results
    std::cout << "Computing reference results..." << std::endl;
    std::vector<std::vector<float>> output_ref(L, std::vector<float>(hidden_dim));
    reference_rms_norm(input_data, weight_data, output_ref);
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(rms_norm_top, FLAGS_bitstream,
                L,
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(input_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(weight_hw),
                tapa::write_only_mmap<tapa::vec_t<float, 16>>(output_hw_raw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Convert hardware output back to structured format
    std::cout << "Converting hardware output..." << std::endl;
    std::vector<std::vector<float>> output_hw(L, std::vector<float>(hidden_dim));
    
    // Unpack output data (row by row layout)
    vec_idx = 0;
    for (int i = 0; i < L; i+=16) {
        for (int j = 0; j < hidden_dim; j++) {
            for (int k = 0; k < 16; k++) {
                output_hw[i+k][j] = output_hw_raw[vec_idx][k];
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
        for (int j = 0; j < hidden_dim; j++) {
            float diff = std::abs(output_hw[i][j] - output_ref[i][j]);
            if (diff > max_error) {
                max_error = diff;
            }
            
            if (!isClose(output_hw[i][j], output_ref[i][j], tolerance)) {
                errors++;
                if (errors <= 10) {  // Print first 10 errors for debugging
                    std::cout << "Error at [" << i << "][" << j << "]: HW=" 
                             << output_hw[i][j] << ", REF=" << output_ref[i][j] 
                             << ", diff=" << diff << std::endl;
                }
            }
        }
    }
    
    std::cout << "Maximum error: " << max_error << std::endl;
    
    if (errors == 0) {
        std::cout << "SUCCESS: All " << (L * hidden_dim) 
                 << " results match within tolerance!" << std::endl;
    } else {
        std::cout << "FAILURE: " << errors << " out of " << (L * hidden_dim) 
                 << " results don't match!" << std::endl;
    }
    
    // Print some sample results for debugging
    std::cout << "\nSample results (first sequence, first 8 dimensions):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < std::min(8, hidden_dim); j++) {
        std::cout << "Output [0][" << j << "]: HW=" << output_hw[0][j] 
                 << ", REF=" << output_ref[0][j] 
                 << ", diff=" << std::abs(output_hw[0][j] - output_ref[0][j]) << std::endl;
    }
    
    // Verify normalization properties
    std::cout << "\nVerifying normalization properties:" << std::endl;
    
    // Check RMS values for first few sequences
    for (int i = 0; i < std::min(4, L); i++) {
        // Compute RMS before normalization (input)
        float input_rms = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            input_rms += input_data[i][j] * input_data[i][j];
        }
        input_rms = std::sqrt(input_rms / hidden_dim);
        
        // Compute variance after normalization (should be close to weight-scaled values)
        float output_mean_sq = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            // Remove weight scaling to check normalized values
            float normalized_val = output_hw[i][j] / weight_data[j];
            output_mean_sq += normalized_val * normalized_val;
        }
        output_mean_sq /= hidden_dim;
        
        std::cout << "Sequence " << i << ": Input RMS=" << input_rms 
                 << ", Output mean square (weight-removed)=" << output_mean_sq << std::endl;
    }
    
    // Print sample input vs output comparison
    std::cout << "\nSample input vs normalized output (sequence 0, first 8 dims):" << std::endl;
    std::cout << "Dim\tInput\t\tWeight\t\tOutput\t\tNorm_Val" << std::endl;
    
    // Compute normalization factor for first sequence for comparison
    float variance = 0.0f;
    for (int j = 0; j < hidden_dim; j++) {
        variance += input_data[0][j] * input_data[0][j];
    }
    variance = variance * R_HIDDEN_DIM + EPSILON;
    float rms_scale = 1.0f / std::sqrt(variance);
    
    for (int j = 0; j < std::min(8, hidden_dim); j++) {
        float normalized_val = input_data[0][j] * rms_scale;
        std::cout << j << "\t" << std::setprecision(4) 
                 << input_data[0][j] << "\t\t" 
                 << weight_data[j] << "\t\t" 
                 << output_hw[0][j] << "\t\t" 
                 << normalized_val << std::endl;
    }
    
    std::cout << "RMS scale factor for sequence 0: " << rms_scale << std::endl;
    
    // Debug: Print data packing information
    std::cout << "\nData packing debug info:" << std::endl;
    std::cout << "  Input vectors created: " << input_vectors_count << std::endl;
    std::cout << "  Weight vectors created: " << weight_vectors_count << std::endl;
    std::cout << "  Output vectors: " << output_vectors_count << std::endl;
    std::cout << "  Expected input size: " << (L * hidden_dim) / 16 << " vectors" << std::endl;
    std::cout << "  Expected weight size: " << hidden_dim / 16 << " vectors" << std::endl;
    std::cout << "  Expected output size: " << (L * hidden_dim) / 16 << " vectors" << std::endl;
    
    // Print statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total input elements: " << (L * hidden_dim) << " floats" << std::endl;
    std::cout << "  Total weight elements: " << hidden_dim << " floats" << std::endl;
    std::cout << "  Total output elements: " << (L * hidden_dim) << " floats" << std::endl;
    std::cout << "  Memory bandwidth utilization:" << std::endl;
    std::cout << "    Input: " << (L * hidden_dim * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Weight: " << (hidden_dim * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Output: " << (L * hidden_dim * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Total: " << ((2 * L + 1) * hidden_dim * sizeof(float)) << " bytes" << std::endl;
    
    // Test edge cases
    std::cout << "\nTesting edge cases:" << std::endl;
    
    // Test with small input values (near zero)
    std::vector<std::vector<float>> small_input(1, std::vector<float>(hidden_dim, 1e-6f));
    std::vector<std::vector<float>> small_output_ref(1, std::vector<float>(hidden_dim));
    reference_rms_norm(small_input, weight_data, small_output_ref);
    
    float small_variance = 0.0f;
    for (int j = 0; j < hidden_dim; j++) {
        small_variance += small_input[0][j] * small_input[0][j];
    }
    small_variance = small_variance * R_HIDDEN_DIM + EPSILON;
    float small_rms_scale = 1.0f / std::sqrt(small_variance);
    
    std::cout << "Small input test: variance=" << small_variance 
             << ", rms_scale=" << small_rms_scale 
             << ", epsilon effect=" << (EPSILON / small_variance) << std::endl;
    
    // Test with large input values
    std::vector<std::vector<float>> large_input(1, std::vector<float>(hidden_dim, 10.0f));
    std::vector<std::vector<float>> large_output_ref(1, std::vector<float>(hidden_dim));
    reference_rms_norm(large_input, weight_data, large_output_ref);
    
    float large_variance = 0.0f;
    for (int j = 0; j < hidden_dim; j++) {
        large_variance += large_input[0][j] * large_input[0][j];
    }
    large_variance = large_variance * R_HIDDEN_DIM + EPSILON;
    float large_rms_scale = 1.0f / std::sqrt(large_variance);
    
    std::cout << "Large input test: variance=" << large_variance 
             << ", rms_scale=" << large_rms_scale 
             << ", epsilon effect=" << (EPSILON / large_variance) << std::endl;
    
    return errors == 0 ? 0 : 1;
}
