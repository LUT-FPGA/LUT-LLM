#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "imm.h"

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

// Helper function to check if two floating point numbers are close
bool isClose(float a, float b, float tolerance = 1e-5) {
    return std::abs(a - b) < tolerance;
}

// Reference implementation with weight quantization
void reference_matrix_multiply_with_weight_vq(
    const std::vector<std::vector<int>>& act_indices,  // L x in_size (activation centroid indices)
    const std::vector<std::vector<std::vector<int>>>& weight_indices,  // in_size x (out_size/256) x 256 (weight centroid indices per submatrix)
    const std::vector<std::vector<std::vector<float>>>& act_centroids,  // in_size x num_act_centroids x vector_dim
    const std::vector<std::vector<std::vector<std::vector<float>>>>& weight_centroids,  // in_size x (out_size/256) x num_weight_centroids x vector_dim
    std::vector<std::vector<float>>& output  // L x out_size
) {
    int L = act_indices.size();
    int in_size = act_indices[0].size();
    int out_size = output[0].size();
    int vector_dim = 2;
    int num_submatrices = out_size / 256;
    
    // Initialize output
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size; j++) {
            output[i][j] = 0.0f;
        }
    }
    
    // Compute output
    for (int i = 0; i < L; i++) {
        for (int r = 0; r < in_size; r++) {
            int act_idx = act_indices[i][r];
            for (int sub = 0; sub < num_submatrices; sub++) {
                for (int col = 0; col < 256; col++) {
                    int out_col = sub * 256 + col;
                    if (out_col < out_size) {  // Handle case where out_size is not multiple of 256
                        int weight_idx = weight_indices[r][sub][col];
                        
                        // Dot product of activation centroid and weight centroid
                        for (int k = 0; k < vector_dim; k++) {
                            output[i][out_col] += act_centroids[r][act_idx][k] * weight_centroids[r][sub][weight_idx][k];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters
    const int L = 128;  // Number of input sequences (sequence length)
    const int input_dim = 16;  // Input dimension of weight matrix (changed from 8 to 16 to get in_size=8)
    const int out_size = 4864;  // Output dimension of weight matrix (changed from 896 to 4864)
    const int vector_dim = 2;  // Dimension of each centroid
    const int in_size = input_dim / vector_dim;  // Number of 2-element positions = 8
    const int num_act_centroids = 64;  // Number of activation centroids per position
    const int num_weight_centroids = 16;  // Number of weight centroids per position
    const int num_submatrices = (out_size + 255) / 256;  // Number of 256-column submatrices = 19
    
    std::cout << "Testing IMM kernel with weight vector quantization:" << std::endl;
    std::cout << "  L (sequence length): " << L << std::endl;
    std::cout << "  Input dimension: " << input_dim << std::endl;
    std::cout << "  in_size (number of 2-element positions): " << in_size << std::endl;
    std::cout << "  Number of activation centroids per position: " << num_act_centroids << std::endl;
    std::cout << "  Number of weight centroids per position: " << num_weight_centroids << std::endl;
    std::cout << "  Output dimension: " << out_size << std::endl;
    std::cout << "  Number of 256-column submatrices: " << num_submatrices << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> centroid_dis(-10.0f, 10.0f);
    std::uniform_real_distribution<float> weight_dis(-0.5f, 0.5f);
    std::uniform_real_distribution<float> input_dis(-10.0f, 10.0f);
    
    // Generate activation centroids for each 2-element position
    std::vector<std::vector<std::vector<float>>> act_centroids(in_size, 
        std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(vector_dim)));
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                act_centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }
    
    // Generate weight centroids for each position and submatrix
    std::vector<std::vector<std::vector<std::vector<float>>>> weight_centroids(in_size,
        std::vector<std::vector<std::vector<float>>>(num_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int sub = 0; sub < num_submatrices; sub++) {
            for (int i = 0; i < num_weight_centroids; i++) {
                for (int j = 0; j < vector_dim; j++) {
                    weight_centroids[pos][sub][i][j] = weight_dis(gen);
                }
            }
        }
    }
    
    // Generate random input vectors
    std::vector<std::vector<std::vector<float>>> input_vectors(L, 
        std::vector<std::vector<float>>(in_size, std::vector<float>(vector_dim)));
    
    for (int i = 0; i < L; i++) {
        for (int pos = 0; pos < in_size; pos++) {
            for (int j = 0; j < vector_dim; j++) {
                input_vectors[i][pos][j] = input_dis(gen);
            }
        }
    }
    
    // Find closest activation centroid indices
    std::cout << "Finding closest activation centroids..." << std::endl;
    std::vector<std::vector<int>> act_indices(L, std::vector<int>(in_size));
    std::vector<std::vector<int>> act_indices_hw(8, std::vector<int>(L * in_size / 8));
    
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int i = 0; i < L; i++) {
            act_indices[i][pos] = find_closest_centroid(input_vectors[i][pos], act_centroids[pos]);
            // Distribute across 8 buffers - round-robin by position
            int buffer_idx = pos % 8;
            int local_pos = pos / 8;
            act_indices_hw[buffer_idx][local_pos * L + i] = act_indices[i][pos];
        }
    }
    
    // Generate random weight vectors and quantize them
    std::cout << "Generating and quantizing weight matrices..." << std::endl;
    std::vector<std::vector<std::vector<std::vector<float>>>> weight_vectors(in_size,
        std::vector<std::vector<std::vector<float>>>(num_submatrices,
            std::vector<std::vector<float>>(256, std::vector<float>(vector_dim))));
    
    std::vector<std::vector<std::vector<int>>> weight_indices(in_size,
        std::vector<std::vector<int>>(num_submatrices, std::vector<int>(256)));
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int sub = 0; sub < num_submatrices; sub++) {
            for (int col = 0; col < 256; col++) {
                // Generate random weight vector
                for (int j = 0; j < vector_dim; j++) {
                    weight_vectors[pos][sub][col][j] = weight_dis(gen);
                }
                // Find closest weight centroid
                weight_indices[pos][sub][col] = find_closest_centroid(weight_vectors[pos][sub][col], weight_centroids[pos][sub]);
            }
        }
    }
    
    // Pack weight indices into hardware format
    std::cout << "Packing weight indices..." << std::endl;
    std::vector<std::vector<tapa::vec_t<ap_uint<4>, 128>>> weight_idx_hw(8, std::vector<tapa::vec_t<ap_uint<4>, 128>>(out_size * in_size / 8 / 128));
    
    for (int pos = 0; pos < in_size; pos++) {
        int buffer_idx = pos % 8;
        int local_pos = pos / 8;
        
        for (int sub = 0; sub < num_submatrices; sub++) {
            for (int vec_idx = 0; vec_idx < 2; vec_idx++) { // 256 indices = 2 x 128-element vectors
                int hw_idx = local_pos * num_submatrices * 2 + sub * 2 + vec_idx;
                for (int k = 0; k < 128; k++) {
                    int col = vec_idx * 128 + k;
                    if (col < 256 && sub * 256 + col < out_size) {
                        weight_idx_hw[buffer_idx][hw_idx][k] = weight_indices[pos][sub][col];
                    } else {
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0; // Padding
                    }
                }
            }
        }
    }
    
    // Precompute 2D lookup tables (activation centroids x weight centroids)
    std::cout << "Precomputing 2D lookup tables..." << std::endl;
    std::vector<std::vector<std::vector<std::vector<float>>>> lut_2d(in_size,
        std::vector<std::vector<std::vector<float>>>(num_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int sub = 0; sub < num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    // Compute dot product of activation and weight centroids
                    float dot_product = 0.0f;
                    for (int k = 0; k < vector_dim; k++) {
                        dot_product += act_centroids[pos][act_idx][k] * weight_centroids[pos][sub][weight_idx][k];
                    }
                    lut_2d[pos][sub][act_idx][weight_idx] = dot_product;
                }
            }
        }
    }
    
    // Pack LUT into hardware format
    std::cout << "Packing LUT into hardware format..." << std::endl;
    std::vector<std::vector<tapa::vec_t<float, 16>>> lut_hw(8, std::vector<tapa::vec_t<float, 16>>(num_submatrices * num_act_centroids * num_weight_centroids * in_size / 8 / 16));
    
    for (int pos = 0; pos < in_size; pos++) {
        int buffer_idx = pos % 8;
        int local_pos = pos / 8;
        
        for (int sub = 0; sub < num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    lut_hw[buffer_idx][local_pos * num_act_centroids * num_submatrices + act_idx * num_submatrices + sub][weight_idx] = lut_2d[pos][sub][act_idx][weight_idx];
                }
            }
        }
    }
    
    // Allocate output array
    int output_elements = L * out_size;
    int num_output_vectors = (output_elements + 15) / 16;
    std::vector<tapa::vec_t<float, 16>> output_hw_raw(num_output_vectors);
    
    std::vector<int> cycle_count_hw(1);
    
    // Compute reference results
    std::cout << "Computing reference results..." << std::endl;
    std::vector<std::vector<float>> output_ref(L, std::vector<float>(out_size));
    reference_matrix_multiply_with_weight_vq(act_indices, weight_indices, act_centroids, weight_centroids, output_ref);
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(imm, FLAGS_bitstream,
                L,
                in_size,
                out_size,
                tapa::read_only_mmaps<int, 8>(act_indices_hw),
                tapa::read_only_mmaps<tapa::vec_t<float, 16>, 8>(lut_hw),
                tapa::read_only_mmaps<tapa::vec_t<ap_uint<4>, 128>, 8>(weight_idx_hw),
                tapa::write_only_mmap<tapa::vec_t<float, 16>>(output_hw_raw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Convert hardware output from tapa::vec_t<float, 16> vectors to 2D array
    // Hardware now writes in output-major order: for each output position, then for each sequence
    std::vector<std::vector<float>> output_hw(L, std::vector<float>(out_size));
    
    for (int i = 0; i < out_size / 16; i++) {        // For each output position group
        for (int j = 0; j < L; j++) {                // For each sequence
            for (int k = 0; k < 16; k++) {           // For each element in the 16-element vector
                int output_idx = i * 16 + k;        // Actual output index
                if (output_idx < out_size) {         // Bounds check
                    int linear_idx = i * L * 16 + j * 16 + k;  // Hardware output layout
                    int vec_idx = linear_idx / 16;
                    int elem_idx = linear_idx % 16;
                    output_hw[j][output_idx] = output_hw_raw[vec_idx][elem_idx];
                }
            }
        }
    }
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    float max_error = 0.0f;
    float tolerance = 1e-3f;  // Slightly relaxed tolerance for weight quantization
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size; j++) {
            float diff = std::abs(output_hw[i][j] - output_ref[i][j]);
            if (diff > max_error) {
                max_error = diff;
            }
            
            if (!isClose(output_hw[i][j], output_ref[i][j], tolerance)) {
                errors++;
                if (errors <= 10) {  // Print first 10 errors
                    std::cout << "Error at [" << i << "][" << j << "]: HW=" 
                             << output_hw[i][j] << ", REF=" << output_ref[i][j] 
                             << ", diff=" << diff << std::endl;
                }
            }
        }
    }
    
    std::cout << "Maximum error: " << max_error << std::endl;
    
    if (errors == 0) {
        std::cout << "SUCCESS: All " << (L * out_size) << " results match within tolerance!" << std::endl;
    } else {
        std::cout << "FAILURE: " << errors << " out of " << (L * out_size) 
                 << " results don't match!" << std::endl;
    }
    
    // Print sample results
    std::cout << "\nSample results (first sequence, first 10 outputs):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < std::min(10, out_size); j++) {
        std::cout << "Output [0][" << j << "]: HW=" << output_hw[0][j] 
                 << ", REF=" << output_ref[0][j] 
                 << ", diff=" << std::abs(output_hw[0][j] - output_ref[0][j]) << std::endl;
    }
    
    return errors == 0 ? 0 : 1;
}
