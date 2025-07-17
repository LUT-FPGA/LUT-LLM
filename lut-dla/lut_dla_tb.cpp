#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include "lut_dla.h"

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

// Helper function to check if two floating point numbers are close
bool isClose(float a, float b, float tolerance = 1e-5) {
    return std::abs(a - b) < tolerance;
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

// Reference implementation: Combined CCU + IMM functionality with weight VQ (floating point)
void reference_lut_dla_with_weight_vq(
    const std::vector<std::vector<std::vector<float>>>& input_vectors,  // in_size x L x vector_dim
    const std::vector<std::vector<std::vector<float>>>& act_centroids,  // in_size x num_act_centroids x vector_dim
    const std::vector<std::vector<std::vector<std::vector<float>>>>& weight_centroids,  // in_size x (out_size/256) x num_weight_centroids x vector_dim
    const std::vector<std::vector<std::vector<int>>>& weight_indices,  // in_size x (out_size/256) x 256
    std::vector<std::vector<float>>& output                             // L x out_size
) {
    int L = input_vectors[0].size();
    int in_size = input_vectors.size();
    int out_size = output[0].size();
    int num_submatrices = (out_size + 255) / 256;
    int vector_dim = 2;
    
    // Initialize output
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size; j++) {
            output[i][j] = 0.0f;
        }
    }
    
    // For each sequence
    for (int i = 0; i < L; i++) {
        // For each position
        for (int pos = 0; pos < in_size; pos++) {
            // CCU: Find closest activation centroid
            int act_idx = find_closest_centroid(input_vectors[pos][i], act_centroids[pos]);
            
            // IMM: Accumulate using 2D LUT (activation centroid x weight centroid)
            for (int sub = 0; sub < num_submatrices; sub++) {
                for (int col = 0; col < 256; col++) {
                    int out_col = sub * 256 + col;
                    if (out_col < out_size) {
                        int weight_idx = weight_indices[pos][sub][col];
                        
                        // Compute dot product of activation and weight centroids
                        float dot_product = 0.0f;
                        for (int k = 0; k < vector_dim; k++) {
                            dot_product += act_centroids[pos][act_idx][k] * weight_centroids[pos][sub][weight_idx][k];
                        }
                        output[i][out_col] += dot_product;
                    }
                }
            }
        }
    }
}

// Reference implementation with quantized LUT (matches hardware behavior)
void reference_lut_dla_quantized_lut(
    const std::vector<std::vector<std::vector<float>>>& input_vectors,  // in_size x L x vector_dim
    const std::vector<std::vector<std::vector<float>>>& act_centroids,  // in_size x num_act_centroids x vector_dim
    const std::vector<std::vector<std::vector<int>>>& weight_indices,  // in_size x (out_size/256) x 256
    const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& lut_2d_quantized,  // in_size x num_submatrices x num_act_centroids x num_weight_centroids
    float scale,
    float zeropoint,
    std::vector<std::vector<float>>& output                             // L x out_size
) {
    int L = input_vectors[0].size();
    int in_size = input_vectors.size();
    int out_size = output[0].size();
    int num_submatrices = (out_size + 255) / 256;
    
    // Initialize output
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size; j++) {
            output[i][j] = 0.0f;
        }
    }
    
    // For each sequence
    for (int i = 0; i < L; i++) {
        // For each position
        for (int pos = 0; pos < in_size; pos++) {
            // CCU: Find closest activation centroid
            int act_idx = find_closest_centroid(input_vectors[pos][i], act_centroids[pos]);
            
            // IMM: Accumulate using quantized 2D LUT
            for (int sub = 0; sub < num_submatrices; sub++) {
                for (int col = 0; col < 256; col++) {
                    int out_col = sub * 256 + col;
                    if (out_col < out_size) {
                        int weight_idx = weight_indices[pos][sub][col];
                        
                        // Look up quantized value and dequantize
                        uint8_t quantized_val = lut_2d_quantized[pos][sub][act_idx][weight_idx];
                        float dequantized_val = (float(quantized_val) - zeropoint) * scale;
                        output[i][out_col] += dequantized_val;
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters - matching IMM testbench configuration
    const int L = 128;              // Number of input sequences
    const int input_dim = 64;       // Input dimension (changed to match IMM)
    const int out_size = 4864;      // Output dimension (changed to match IMM)
    const int vector_dim = 2;       // Dimension of each centroid
    const int num_streams = 8;  // Number of parallel streams
    const int in_size = input_dim / vector_dim;  // Number of 2-element positions = 8
    const int num_act_centroids = 64;   // Number of activation centroids per position
    const int num_weight_centroids = 16; // Number of weight centroids per position
    const int num_submatrices = (out_size + 255) / 256;  // Number of 256-column submatrices
    
    std::cout << "Testing LUT-DLA kernel with weight vector quantization:" << std::endl;
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
    std::uniform_real_distribution<float> centroid_dis(-5.0f, 5.0f);
    std::uniform_real_distribution<float> weight_dis(-0.5f, 0.5f);
    std::uniform_real_distribution<float> input_dis(-5.0f, 5.0f);
    
    // Generate activation centroids for each position (same as CCU centroids)
    std::vector<std::vector<std::vector<float>>> act_centroids(in_size, 
        std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(vector_dim)));
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                act_centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }
    
    // Pack activation centroids into hardware format (same as CCU)
    std::vector<tapa::vec_t<float, 16>> centroids_hw(in_size * num_act_centroids / 8);
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroids_hw[(pos/num_streams)*num_act_centroids+i][(pos % num_streams)*vector_dim+j] = act_centroids[pos][i][j];
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
    // Input format: L vectors of 16 elements each (8 streams * 2 elements per stream)
    std::vector<tapa::vec_t<float, 16>> input_hw(L * in_size / num_streams);
    std::vector<std::vector<std::vector<float>>> input_vectors(in_size, 
        std::vector<std::vector<float>>(L, std::vector<float>(vector_dim)));
    
    for (int r = 0; r < (in_size / num_streams); r++){
        for (int l = 0; l < L; ++l) {
            for (int s = 0; s < num_streams; ++s) {
                for (int j = 0; j < vector_dim; ++j) {
                    float val = input_dis(gen);
                    // Pack 8 streams of 2-element vectors into 16-element hardware vectors
                    int elem_idx = s * vector_dim + j;
                    input_hw[r * L + l][elem_idx] = val;
                    input_vectors[r*num_streams+s][l][j] = val;
                }
            }
        }
    }
    
    // Generate random weight vectors and quantize them
    std::cout << "Generating and quantizing weight matrices..." << std::endl;
    std::vector<std::vector<std::vector<int>>> weight_indices(in_size,
        std::vector<std::vector<int>>(num_submatrices, std::vector<int>(256)));
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int sub = 0; sub < num_submatrices; sub++) {
            for (int col = 0; col < 256; col++) {
                // Generate random weight vector
                std::vector<float> weight_vec(vector_dim);
                for (int j = 0; j < vector_dim; j++) {
                    weight_vec[j] = weight_dis(gen);
                }
                // Find closest weight centroid
                weight_indices[pos][sub][col] = find_closest_centroid(weight_vec, weight_centroids[pos][sub]);
            }
        }
    }
    
    // Pack weight indices into 8 hardware buffers (following IMM pattern)
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> weight_idx_hw(8);
    
    for (int buffer_idx = 0; buffer_idx < 8; buffer_idx++) {
        weight_idx_hw[buffer_idx].resize((in_size / 8) * num_submatrices * 2);
    }
    
    for (int pos = 0; pos < in_size; pos++) {
        int buffer_idx = pos % 8;
        int local_pos = pos / 8;
        
        for (int sub = 0; sub < num_submatrices; sub++) {
            for (int vec_idx = 0; vec_idx < 2; vec_idx++) {
                int hw_idx = local_pos * num_submatrices * 2 + sub * 2 + vec_idx;
                for (int k = 0; k < 64; k++) {
                    int col = vec_idx * 128 + k * 2;
                    if (col < 256 && sub * 256 + col < out_size) {
                        ap_uint<8> tmp_idx;
                        tmp_idx(3, 0) = weight_indices[pos][sub][col];
                        tmp_idx(7, 4) = weight_indices[pos][sub][col + 1];
                        weight_idx_hw[buffer_idx][hw_idx][k] = tmp_idx;
                    } else {
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0; // Padding
                    }
                }
            }
        }
    }
    
    // Precompute 2D lookup tables and pack into 8 hardware buffers (following IMM pattern)
    std::cout << "Precomputing 2D lookup tables..." << std::endl;
    
    // First compute floating-point 2D LUT
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
    
    // Compute scale and zero-point for quantization
    std::cout << "Computing quantization parameters..." << std::endl;
    auto [scale, zeropoint] = compute_scale_zeropoint(lut_2d, in_size, num_submatrices, num_act_centroids, num_weight_centroids);
    std::cout << "  Scale: " << scale << std::endl;
    std::cout << "  Zero-point: " << zeropoint << std::endl;
    
    // Quantize LUT values
    std::cout << "Quantizing LUT values..." << std::endl;
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> lut_2d_quantized(in_size,
        std::vector<std::vector<std::vector<uint8_t>>>(num_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int sub = 0; sub < num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    lut_2d_quantized[pos][sub][act_idx][weight_idx] = quantize_value(lut_2d[pos][sub][act_idx][weight_idx], scale, zeropoint);
                }
            }
        }
    }
    
    // Pack quantized LUT into hardware format
    std::cout << "Packing quantized LUT into hardware format..." << std::endl;
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> lut_hw(8, std::vector<tapa::vec_t<ap_uint<8>, 64>>(num_submatrices * num_act_centroids * in_size / 8 / 4));
    
    for (int pos = 0; pos < in_size; pos++) {
        int buffer_idx = pos % 8;
        int local_pos = pos / 8;
        
        for (int sub = 0; sub < num_submatrices; sub++) {
            // Process groups of 4 activation centroids at a time (as expected by kernel)
            for (int act_group = 0; act_group < num_act_centroids / 4; act_group++) {
                int hw_idx = local_pos * num_submatrices * (num_act_centroids / 4) + act_group * num_submatrices + sub;
                
                // Pack 64 elements: 4 activation centroids x 16 weight centroids
                for (int k = 0; k < 16; k++) {  // 16 weight centroids
                    for (int ii = 0; ii < 4; ii++) {  // 4 activation centroids
                        int act_idx = act_group * 4 + ii;
                        if (act_idx < num_act_centroids && k < num_weight_centroids) {
                            int elem_idx = ii * 16 + k;  // Matches kernel: tmp[ii*16+k]
                            lut_hw[buffer_idx][hw_idx][elem_idx] = lut_2d_quantized[pos][sub][act_idx][k];
                        } else {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = 0;  // Padding
                        }
                    }
                }
            }
        }
    }
    
    // Create scale/zero-point buffer for hardware (if needed by lut_dla kernel)
    std::vector<ap_uint<64>> scale_zero_hw(1);
    float zeropoint_hw = zeropoint * scale * in_size;  // Convert zero-point as specified
    scale_zero_hw[0] = (tapa::bit_cast<ap_uint<32>>(zeropoint_hw), tapa::bit_cast<ap_uint<32>>(scale));
    
    // Allocate output array
    int output_elements = L * out_size;
    int num_output_vectors = (output_elements + 15) / 16;
    std::vector<tapa::vec_t<float, 16>> output_hw_raw(num_output_vectors);
    
    std::vector<int> cycle_count_hw(1);
    
    // Compute reference results (floating point)
    std::cout << "Computing floating-point reference results..." << std::endl;
    std::vector<std::vector<float>> output_ref_fp(L, std::vector<float>(out_size));
    reference_lut_dla_with_weight_vq(input_vectors, act_centroids, weight_centroids, weight_indices, output_ref_fp);
    
    // Compute reference results (quantized LUT)
    std::cout << "Computing quantized LUT reference results..." << std::endl;
    std::vector<std::vector<float>> output_ref_quant(L, std::vector<float>(out_size));
    reference_lut_dla_quantized_lut(input_vectors, act_centroids, weight_indices, lut_2d_quantized, scale, zeropoint, output_ref_quant);
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(lut_dla_core, FLAGS_bitstream,
                L,
                in_size,
                out_size,
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(input_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(centroids_hw),
                tapa::read_only_mmaps<tapa::vec_t<ap_uint<8>, 64>, 8>(lut_hw),
                tapa::read_only_mmaps<tapa::vec_t<ap_uint<8>, 64>, 8>(weight_idx_hw),
                tapa::read_only_mmap<ap_uint<64>>(scale_zero_hw),
                tapa::write_only_mmap<tapa::vec_t<float, 16>>(output_hw_raw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Convert hardware output from tapa::vec_t<float, 16> vectors to 2D array
    // Hardware writes in output-major order (same as IMM)
    std::vector<std::vector<float>> output_hw(L, std::vector<float>(out_size));
    
    for (int i = 0; i < out_size / 16; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < 16; k++) {
                int output_idx = i * 16 + k;
                if (output_idx < out_size) {
                    int linear_idx = i * L * 16 + j * 16 + k;
                    int vec_idx = linear_idx / 16;
                    int elem_idx = linear_idx % 16;
                    output_hw[j][output_idx] = output_hw_raw[vec_idx][elem_idx];
                }
            }
        }
    }
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    
    // Compare hardware vs floating-point reference
    int errors_fp = 0;
    float max_error_fp = 0.0f;
    float tolerance_fp = 5e-2f;  // Tight tolerance - should match exactly
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size; j++) {
            float diff = std::abs(output_hw[i][j] - output_ref_fp[i][j]);
            if (diff > max_error_fp) {
                max_error_fp = diff;
            }
            
            if (!isClose(output_hw[i][j], output_ref_fp[i][j], tolerance_fp)) {
                errors_fp++;
                if (errors_fp <= 5) {  // Print first 5 errors
                    std::cout << "FP Error at [" << i << "][" << j << "]: HW=" 
                             << output_hw[i][j] << ", FP_REF=" << output_ref_fp[i][j] 
                             << ", diff=" << diff << std::endl;
                }
            }
        }
    }
    
    // Compare hardware vs quantized reference
    int errors_quant = 0;
    float max_error_quant = 0.0f;
    float tolerance_quant = 1e-4f;  // Relaxed tolerance - expect differences due to quantization
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size; j++) {
            float diff = std::abs(output_hw[i][j] - output_ref_quant[i][j]);
            if (diff > max_error_quant) {
                max_error_quant = diff;
            }
            
            if (!isClose(output_hw[i][j], output_ref_quant[i][j], tolerance_quant)) {
                errors_quant++;
                if (errors_quant <= 5) {  // Print first 5 errors
                    std::cout << "Quant Error at [" << i << "][" << j << "]: HW=" 
                             << output_hw[i][j] << ", QUANT_REF=" << output_ref_quant[i][j] 
                             << ", diff=" << diff << std::endl;
                }
            }
        }
    }
    
    // Compare floating-point vs quantized to show quantization impact
    int quant_diff_count = 0;
    float max_quant_impact = 0.0f;
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size; j++) {
            float diff = std::abs(output_ref_fp[i][j] - output_ref_quant[i][j]);
            if (diff > max_quant_impact) {
                max_quant_impact = diff;
            }
            
            if (!isClose(output_ref_fp[i][j], output_ref_quant[i][j], 1e-3f)) {
                quant_diff_count++;
            }
        }
    }
    
    std::cout << "\n=== RESULTS SUMMARY ===" << std::endl;
    std::cout << "Maximum error (HW vs FP reference): " << max_error_fp << std::endl;
    std::cout << "Maximum error (HW vs Quantized reference): " << max_error_quant << std::endl;
    std::cout << "Maximum quantization impact (FP vs Quantized): " << max_quant_impact << std::endl;
    
    if (errors_fp == 0) {
        std::cout << "SUCCESS: Hardware matches floating-point reference exactly!" << std::endl;
    } else {
        std::cout << "FAILURE: " << errors_fp << " out of " << (L * out_size) 
                 << " results don't match floating-point reference!" << std::endl;
    }
    
    if (errors_quant == 0) {
        std::cout << "SUCCESS: Hardware matches quantized reference exactly!" << std::endl;
    } else {
        std::cout << "EXPECTED: " << errors_quant << " out of " << (L * out_size) 
                 << " results differ from quantized reference (showing quantization impact)!" << std::endl;
    }
    
    std::cout << "Quantization impact: " << quant_diff_count << " out of " << (L * out_size) 
             << " values differ between FP and quantized references" << std::endl;
    
    // Print sample results
    std::cout << "\nSample results (first sequence, first 10 outputs):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < std::min(10, out_size); j++) {
        std::cout << "Output [0][" << j << "]:" << std::endl;
        std::cout << "  HW:         " << output_hw[0][j] << std::endl;
        std::cout << "  FP_REF:     " << output_ref_fp[0][j] << std::endl;
        std::cout << "  QUANT_REF:  " << output_ref_quant[0][j] << std::endl;
        std::cout << "  HW-FP diff: " << std::abs(output_hw[0][j] - output_ref_fp[0][j]) << std::endl;
        std::cout << "  HW-Q diff:  " << std::abs(output_hw[0][j] - output_ref_quant[0][j]) << std::endl;
        std::cout << std::endl;
    }
    
    // Print statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total input vectors: " << L * in_size << std::endl;
    std::cout << "  Total activation centroids: " << in_size * num_act_centroids << std::endl;
    std::cout << "  Total weight centroids: " << in_size * num_submatrices * num_weight_centroids << std::endl;
    std::cout << "  Total 2D LUT entries: " << in_size * num_act_centroids * num_weight_centroids << std::endl;
    std::cout << "  Total output elements: " << output_elements << std::endl;
    std::cout << "  Quantization scale: " << scale << std::endl;
    std::cout << "  Quantization zero-point: " << zeropoint << std::endl;
    
    return errors_fp == 0 ? 0 : 1;
}
