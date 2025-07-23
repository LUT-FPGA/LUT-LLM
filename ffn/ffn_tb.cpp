#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include "ffn.h"

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

// Helper function to check if two floating point numbers are close
bool isClose(float a, float b, float tolerance = 1e-4) {
    return std::abs(a - b) < tolerance;
}

// Reference implementation with weight vector quantization (floating point) 
void reference_linear_with_weight_vq(
    const std::vector<std::vector<std::vector<float>>>& input_vectors,  // in_size x L x vector_dim
    const std::vector<std::vector<std::vector<float>>>& act_centroids,  // in_size x num_act_centroids x vector_dim
    const std::vector<std::vector<std::vector<std::vector<float>>>>& weight_centroids,  // in_size x (out_size/256) x num_weight_centroids x vector_dim
    const std::vector<std::vector<std::vector<int>>>& weight_indices,  // in_size x (out_size/256) x 256
    std::vector<std::vector<float>>& output,                             // L x out_size
    int L, int in_size, int out_size
) {
    int vector_dim = 2;
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
                    
                    // Compute dot product between activation and weight centroids
                    float dot_product = 0.0f;
                    for (int k = 0; k < vector_dim; k++) {
                        dot_product += act_centroids[pos][act_centroid_idx][k] * 
                                     weight_centroids[pos][sub][weight_centroid_idx][k];
                    }

                    output[i][sub * 512 + j] += dot_product;
                }
            }
        }
    }
}

// Reference implementation using quantized LUT (matches hardware behavior)
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

// Reference FFN implementation using weight vector quantization  
void reference_ffn_with_weight_vq(
    const std::vector<std::vector<float>>& input,  // L x HIDDEN_DIM
    const std::vector<std::vector<std::vector<float>>>& up_act_centroids,    // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<std::vector<float>>>>& up_weight_centroids,  // (HIDDEN_DIM/2) x (INTERM_DIM/256) x num_weight_centroids x 2
    const std::vector<std::vector<std::vector<int>>>& up_weight_indices,  // (HIDDEN_DIM/2) x (INTERM_DIM/256) x 256
    const std::vector<std::vector<std::vector<float>>>& gate_act_centroids,  // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<std::vector<float>>>>& gate_weight_centroids,  // (HIDDEN_DIM/2) x (INTERM_DIM/256) x num_weight_centroids x 2
    const std::vector<std::vector<std::vector<int>>>& gate_weight_indices,  // (HIDDEN_DIM/2) x (INTERM_DIM/256) x 256
    const std::vector<std::vector<std::vector<float>>>& down_act_centroids,  // (INTERM_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<std::vector<float>>>>& down_weight_centroids,  // (INTERM_DIM/2) x (HIDDEN_DIM/256) x num_weight_centroids x 2
    const std::vector<std::vector<std::vector<int>>>& down_weight_indices,  // (INTERM_DIM/2) x (HIDDEN_DIM/256) x 256
    std::vector<std::vector<float>>& output,       // L x HIDDEN_DIM
    int L, bool use_direct_silu = false
) {
    int up_in_size = HIDDEN_DIM / 2;
    int down_in_size = INTERM_DIM / 2;
    
    // Convert input to vector format for up/gate projections
    std::vector<std::vector<std::vector<float>>> up_input_vectors(up_in_size, 
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < L; i++) {
            up_input_vectors[pos][i][0] = input[i][pos * 2];
            up_input_vectors[pos][i][1] = input[i][pos * 2 + 1];
        }
    }
    
    // Up projection
    std::vector<std::vector<float>> up_output(L, std::vector<float>(INTERM_DIM));
    reference_linear_with_weight_vq(up_input_vectors, up_act_centroids, up_weight_centroids, 
                                   up_weight_indices, up_output, L, up_in_size, INTERM_DIM);
    
    // Gate projection (same input as up)
    std::vector<std::vector<float>> gate_output(L, std::vector<float>(INTERM_DIM));
    reference_linear_with_weight_vq(up_input_vectors, gate_act_centroids, gate_weight_centroids,
                                   gate_weight_indices, gate_output, L, up_in_size, INTERM_DIM);
    
    // Debug logging for up and gate projections (first sequence, first 8 elements)
    std::cout << "[DEBUG] Reference Up projection output (seq 0, first 8 elements): ";
    for (int j = 0; j < std::min(8, INTERM_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << up_output[0][j] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "[DEBUG] Reference Gate projection output (seq 0, first 8 elements): ";
    for (int j = 0; j < std::min(8, INTERM_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << gate_output[0][j] << " ";
    }
    std::cout << std::endl;
    
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
    
    // Debug logging for gate after SiLU (first sequence, first 8 elements)
    std::cout << "[DEBUG] Reference Gate after SiLU (seq 0, first 8 elements): ";
    for (int j = 0; j < std::min(8, INTERM_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << gate_output[0][j] << " ";
    }
    std::cout << std::endl;
    
    // Element-wise multiplication
    std::vector<std::vector<float>> intermediate(L, std::vector<float>(INTERM_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < INTERM_DIM; j++) {
            intermediate[i][j] = up_output[i][j] * gate_output[i][j];
        }
    }
    
    // Debug logging for element-wise multiplication result (first sequence, first 8 elements)
    std::cout << "[DEBUG] Reference Element-wise multiplication result (seq 0, first 8 elements): ";
    for (int j = 0; j < std::min(8, INTERM_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << intermediate[0][j] << " ";
    }
    std::cout << std::endl;
    
    // Convert intermediate to vector format for down projection
    std::vector<std::vector<std::vector<float>>> down_input_vectors(down_in_size,
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int i = 0; i < L; i++) {
            down_input_vectors[pos][i][0] = intermediate[i][pos * 2];
            down_input_vectors[pos][i][1] = intermediate[i][pos * 2 + 1];
        }
    }
    
    // Down projection
    reference_linear_with_weight_vq(down_input_vectors, down_act_centroids, down_weight_centroids,
                                   down_weight_indices, output, L, down_in_size, HIDDEN_DIM);
}

// Reference implementation with quantized LUT (matches hardware behavior)
void reference_linear_with_quantized_lut(
    const std::vector<std::vector<std::vector<float>>>& input_vectors,  // in_size x L x vector_dim
    const std::vector<std::vector<std::vector<float>>>& act_centroids,  // in_size x num_act_centroids x vector_dim
    const std::vector<std::vector<std::vector<int>>>& weight_indices,  // in_size x (out_size/256) x 256
    const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& lut_2d_quantized,  // in_size x num_submatrices x num_act_centroids x num_weight_centroids
    float scale,
    float zeropoint,
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
                    
                    // Look up quantized value and dequantize
                    uint8_t quantized_val = lut_2d_quantized[pos][sub][act_centroid_idx][weight_centroid_idx];
                    float dequantized_val = (float(quantized_val) - zeropoint) * scale;

                    output[i][sub * 512 + j] += dequantized_val;
                }
            }
        }
    }
}

// Reference FFN implementation using quantized LUT
void reference_ffn_with_quantized_lut(
    const std::vector<std::vector<float>>& input,  // L x HIDDEN_DIM
    const std::vector<std::vector<std::vector<float>>>& up_act_centroids,    // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<int>>>& up_weight_indices,  // (HIDDEN_DIM/2) x (INTERM_DIM/256) x 256
    const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& up_lut_2d_quantized,  // (HIDDEN_DIM/2) x (INTERM_DIM/256) x num_act_centroids x num_weight_centroids
    float up_scale, float up_zeropoint,
    const std::vector<std::vector<std::vector<float>>>& gate_act_centroids,  // (HIDDEN_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<int>>>& gate_weight_indices,  // (HIDDEN_DIM/2) x (INTERM_DIM/256) x 256
    const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& gate_lut_2d_quantized,  // (HIDDEN_DIM/2) x (INTERM_DIM/256) x num_act_centroids x num_weight_centroids
    float gate_scale, float gate_zeropoint,
    const std::vector<std::vector<std::vector<float>>>& down_act_centroids,  // (INTERM_DIM/2) x num_centroids x 2
    const std::vector<std::vector<std::vector<int>>>& down_weight_indices,  // (INTERM_DIM/2) x (HIDDEN_DIM/256) x 256
    const std::vector<std::vector<std::vector<std::vector<uint8_t>>>>& down_lut_2d_quantized,  // (INTERM_DIM/2) x (HIDDEN_DIM/256) x num_act_centroids x num_weight_centroids
    float down_scale, float down_zeropoint,
    std::vector<std::vector<float>>& output,       // L x HIDDEN_DIM
    int L, bool use_direct_silu = false
) {
    int up_in_size = HIDDEN_DIM / 2;
    int down_in_size = INTERM_DIM / 2;
    
    // Convert input to vector format for up/gate projections
    std::vector<std::vector<std::vector<float>>> up_input_vectors(up_in_size, 
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < L; i++) {
            up_input_vectors[pos][i][0] = input[i][pos * 2];
            up_input_vectors[pos][i][1] = input[i][pos * 2 + 1];
        }
    }
    
    // Up projection with quantized LUT
    std::vector<std::vector<float>> up_output(L, std::vector<float>(INTERM_DIM));
    reference_linear_with_quantized_lut(up_input_vectors, up_act_centroids, up_weight_indices, 
                                       up_lut_2d_quantized, up_scale, up_zeropoint,
                                       up_output, L, up_in_size, INTERM_DIM);
    
    // Gate projection with quantized LUT (same input as up)
    std::vector<std::vector<float>> gate_output(L, std::vector<float>(INTERM_DIM));
    reference_linear_with_quantized_lut(up_input_vectors, gate_act_centroids, gate_weight_indices,
                                       gate_lut_2d_quantized, gate_scale, gate_zeropoint,
                                       gate_output, L, up_in_size, INTERM_DIM);
    
    // Debug logging for up and gate projections (first sequence, first 8 elements)
    std::cout << "[DEBUG] Quantized Up projection output (seq 0, first 8 elements): ";
    for (int j = 0; j < std::min(8, INTERM_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << up_output[0][j] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "[DEBUG] Quantized Gate projection output (seq 0, first 8 elements): ";
    for (int j = 0; j < std::min(8, INTERM_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << gate_output[0][j] << " ";
    }
    std::cout << std::endl;
    
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
    
    // Debug logging for gate after SiLU (first sequence, first 8 elements)
    std::cout << "[DEBUG] Quantized Gate after SiLU (seq 0, first 8 elements): ";
    for (int j = 0; j < std::min(8, INTERM_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << gate_output[0][j] << " ";
    }
    std::cout << std::endl;
    
    // Element-wise multiplication
    std::vector<std::vector<float>> intermediate(L, std::vector<float>(INTERM_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < INTERM_DIM; j++) {
            intermediate[i][j] = up_output[i][j] * gate_output[i][j];
        }
    }
    
    // Debug logging for element-wise multiplication result (first sequence, first 8 elements)
    std::cout << "[DEBUG] Quantized Element-wise multiplication result (seq 0, first 8 elements): ";
    for (int j = 0; j < std::min(8, INTERM_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << intermediate[0][j] << " ";
    }
    std::cout << std::endl;
    
    // Convert intermediate to vector format for down projection
    std::vector<std::vector<std::vector<float>>> down_input_vectors(down_in_size,
        std::vector<std::vector<float>>(L, std::vector<float>(2)));
    
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int i = 0; i < L; i++) {
            down_input_vectors[pos][i][0] = intermediate[i][pos * 2];
            down_input_vectors[pos][i][1] = intermediate[i][pos * 2 + 1];
        }
    }
    
    // Down projection with quantized LUT
    reference_linear_with_quantized_lut(down_input_vectors, down_act_centroids, down_weight_indices,
                                       down_lut_2d_quantized, down_scale, down_zeropoint,
                                       output, L, down_in_size, HIDDEN_DIM);
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters - matching IMM testbench configuration
    const int L = 32;              // Sequence length
    const int num_act_centroids = 64;   // Number of activation centroids per position
    const int num_weight_centroids = 16; // Number of weight centroids per position
    const int vector_dim = 2;       // Dimension of each centroid
    const int num_streams = 8;      // Number of parallel streams
    
    std::cout << "Testing FFN kernel with weight vector quantization:" << std::endl;
    std::cout << "  L (sequence length): " << L << std::endl;
    std::cout << "  Hidden dimension: " << HIDDEN_DIM << std::endl;
    std::cout << "  Intermediate dimension: " << INTERM_DIM << std::endl;
    std::cout << "  Number of activation centroids per position: " << num_act_centroids << std::endl;
    std::cout << "  Number of weight centroids per position: " << num_weight_centroids << std::endl;
    std::cout << "  Vector dimension: " << vector_dim << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> centroid_dis(-2.0f, 2.0f);
    std::uniform_real_distribution<float> weight_dis(-0.2f, 0.2f);
    std::uniform_real_distribution<float> input_dis(-2.0f, 2.0f);
    
    // Generate random input
    std::vector<std::vector<float>> input(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            input[i][j] = input_dis(gen);
        }
    }
    
    // Pack input into hardware format (L x HIDDEN_DIM/16 vectors of 16 elements)
    std::vector<tapa::vec_t<float, 16>> input_hw(L * HIDDEN_DIM / 16);
    for (int i = 0; i < (HIDDEN_DIM / 16); i++) {
        for (int j = 0; j < L; j++) {
            int hw_idx = i * L + j;
            for (int k = 0; k < 16; k++) {
                input_hw[hw_idx][k] = input[j][i * 16 + k];
            }
        }
    }
    
    // Calculate dimensions for different projections
    int up_in_size = HIDDEN_DIM / 2;    // Input positions for up/gate projections
    int down_in_size = INTERM_DIM / 2;  // Input positions for down projection
    int up_num_submatrices = (INTERM_DIM + 511) / 512;  // Number of 256-column submatrices for up/gate
    int down_num_submatrices = (HIDDEN_DIM + 511) / 512; // Number of 256-column submatrices for down
    
    std::cout << "  Up projection: " << up_in_size << " positions -> " << INTERM_DIM << " outputs (" << up_num_submatrices << " submatrices)" << std::endl;
    std::cout << "  Down projection: " << down_in_size << " positions -> " << HIDDEN_DIM << " outputs (" << down_num_submatrices << " submatrices)" << std::endl;
    
    
    // Generate activation centroids for each position 
    // For up/gate projections: HIDDEN_DIM/2 positions
    // For down projection: INTERM_DIM/2 positions
    // Concatenated order: [up_centroids, down_centroids]
    std::cout << "Generating activation centroids..." << std::endl;
    
    std::vector<std::vector<std::vector<float>>> up_act_centroids(up_in_size,
        std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(vector_dim)));
    std::vector<std::vector<std::vector<float>>> down_act_centroids(down_in_size,
        std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(vector_dim)));
    
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                up_act_centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }
    
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                down_act_centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }
    
    // Pack activation centroids into hardware format (CENTROID_SIZE = HIDDEN_DIM_DIV_2 + INTERM_DIM_DIV_2)
    int total_centroid_positions = up_in_size + down_in_size; // CENTROID_SIZE
    std::vector<tapa::vec_t<float, 16>> centroid_hw(total_centroid_positions * num_act_centroids / 8);
    
    // Pack up centroids first
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroid_hw[(pos/num_streams)*num_act_centroids+i][(pos % num_streams)*vector_dim+j] = up_act_centroids[pos][i][j];
            }
        }
    }
    
    // Pack down centroids second
    int down_offset = (up_in_size / num_streams) * num_act_centroids;
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int i = 0; i < num_act_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroid_hw[down_offset+(pos/num_streams)*num_act_centroids+i][(pos % num_streams)*vector_dim+j] = down_act_centroids[pos][i][j];
            }
        }
    }
    
    // Generate weight centroids and indices for each projection
    std::cout << "Generating weight centroids and indices..." << std::endl;
    
    // Up projection weight centroids and indices
    std::vector<std::vector<std::vector<std::vector<float>>>> up_weight_centroids(up_in_size,
        std::vector<std::vector<std::vector<float>>>(up_num_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> up_weight_indices(up_in_size,
        std::vector<std::vector<int>>(up_num_submatrices, std::vector<int>(512)));
    
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int sub = 0; sub < up_num_submatrices; sub++) {
            for (int i = 0; i < num_weight_centroids; i++) {
                for (int j = 0; j < vector_dim; j++) {
                    up_weight_centroids[pos][sub][i][j] = weight_dis(gen);
                }
            }
            for (int col = 0; col < 512; col++) {
                // Generate random weight vector
                std::vector<float> weight_vec(vector_dim);
                for (int j = 0; j < vector_dim; j++) {
                    weight_vec[j] = weight_dis(gen);
                }
                // Find closest weight centroid
                up_weight_indices[pos][sub][col] = find_closest_centroid(weight_vec, up_weight_centroids[pos][sub]);
            }
        }
    }
    
    // Gate projection weight centroids and indices (same structure as up)
    std::vector<std::vector<std::vector<std::vector<float>>>> gate_weight_centroids(up_in_size,
        std::vector<std::vector<std::vector<float>>>(up_num_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> gate_weight_indices(up_in_size,
        std::vector<std::vector<int>>(up_num_submatrices, std::vector<int>(512)));
    
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int sub = 0; sub < up_num_submatrices; sub++) {
            for (int i = 0; i < num_weight_centroids; i++) {
                for (int j = 0; j < vector_dim; j++) {
                    gate_weight_centroids[pos][sub][i][j] = weight_dis(gen);
                }
            }
            for (int col = 0; col < 512; col++) {
                // Generate random weight vector
                std::vector<float> weight_vec(vector_dim);
                for (int j = 0; j < vector_dim; j++) {
                    weight_vec[j] = weight_dis(gen);
                }
                // Find closest weight centroid
                gate_weight_indices[pos][sub][col] = find_closest_centroid(weight_vec, gate_weight_centroids[pos][sub]);
            }
        }
    }
    
    // Down projection weight centroids and indices
    std::vector<std::vector<std::vector<std::vector<float>>>> down_weight_centroids(down_in_size,
        std::vector<std::vector<std::vector<float>>>(down_num_submatrices,
            std::vector<std::vector<float>>(num_weight_centroids, std::vector<float>(vector_dim))));
    std::vector<std::vector<std::vector<int>>> down_weight_indices(down_in_size,
        std::vector<std::vector<int>>(down_num_submatrices, std::vector<int>(512)));
    
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int sub = 0; sub < down_num_submatrices; sub++) {
            for (int i = 0; i < num_weight_centroids; i++) {
                for (int j = 0; j < vector_dim; j++) {
                    down_weight_centroids[pos][sub][i][j] = weight_dis(gen);
                }
            }
            for (int col = 0; col < 512; col++) {
                // Generate random weight vector
                std::vector<float> weight_vec(vector_dim);
                for (int j = 0; j < vector_dim; j++) {
                    weight_vec[j] = weight_dis(gen);
                }
                // Find closest weight centroid
                down_weight_indices[pos][sub][col] = find_closest_centroid(weight_vec, down_weight_centroids[pos][sub]);
            }
        }
    }
    
    
    // Precompute floating-point 2D LUTs for all three projections
    std::cout << "Precomputing 2D lookup tables..." << std::endl;
    
    // Up projection LUT
    std::vector<std::vector<std::vector<std::vector<float>>>> up_lut_2d(up_in_size,
        std::vector<std::vector<std::vector<float>>>(up_num_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int sub = 0; sub < up_num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    // Compute dot product between activation and weight centroids
                    float dot_product = 0.0f;
                    for (int k = 0; k < vector_dim; k++) {
                        dot_product += up_act_centroids[pos][act_idx][k] * 
                                     up_weight_centroids[pos][sub][weight_idx][k];
                    }
                    up_lut_2d[pos][sub][act_idx][weight_idx] = dot_product;
                }
            }
        }
    }
    
    // Gate projection LUT (same structure)
    std::vector<std::vector<std::vector<std::vector<float>>>> gate_lut_2d(up_in_size,
        std::vector<std::vector<std::vector<float>>>(up_num_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int sub = 0; sub < up_num_submatrices; sub++) {
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
    
    // Down projection LUT
    std::vector<std::vector<std::vector<std::vector<float>>>> down_lut_2d(down_in_size,
        std::vector<std::vector<std::vector<float>>>(down_num_submatrices,
            std::vector<std::vector<float>>(num_act_centroids, std::vector<float>(num_weight_centroids))));
    
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int sub = 0; sub < down_num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    float dot_product = 0.0f;
                    for (int k = 0; k < vector_dim; k++) {
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
    auto [up_scale, up_zeropoint] = compute_scale_zeropoint(up_lut_2d, up_in_size, up_num_submatrices, num_act_centroids, num_weight_centroids);
    auto [gate_scale, gate_zeropoint] = compute_scale_zeropoint(gate_lut_2d, up_in_size, up_num_submatrices, num_act_centroids, num_weight_centroids);
    auto [down_scale, down_zeropoint] = compute_scale_zeropoint(down_lut_2d, down_in_size, down_num_submatrices, num_act_centroids, num_weight_centroids);
    
    std::cout << "  Up scale: " << up_scale << ", zeropoint: " << up_zeropoint << std::endl;
    std::cout << "  Gate scale: " << gate_scale << ", zeropoint: " << gate_zeropoint << std::endl;
    std::cout << "  Down scale: " << down_scale << ", zeropoint: " << down_zeropoint << std::endl;
    
    // Create scale/zero-point buffer for hardware (3 packets: up, gate, down)
    std::vector<ap_uint<64>> scale_zero_hw(3);
    float up_zeropoint_hw = up_zeropoint * up_scale * up_in_size;  // Convert zero-point as in IMM
    float gate_zeropoint_hw = gate_zeropoint * gate_scale * up_in_size;
    float down_zeropoint_hw = down_zeropoint * down_scale * down_in_size;
    
    scale_zero_hw[0] = (tapa::bit_cast<ap_uint<32>>(up_zeropoint_hw), tapa::bit_cast<ap_uint<32>>(up_scale));
    scale_zero_hw[1] = (tapa::bit_cast<ap_uint<32>>(gate_zeropoint_hw), tapa::bit_cast<ap_uint<32>>(gate_scale));
    scale_zero_hw[2] = (tapa::bit_cast<ap_uint<32>>(down_zeropoint_hw), tapa::bit_cast<ap_uint<32>>(down_scale));
    
    // Quantize LUT values
    std::cout << "Quantizing LUT values..." << std::endl;
    
    // Create concatenated up+gate LUT (concatenated in submatrix dimension: first up, then gate)
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> up_gate_lut_2d_quantized(up_in_size,
        std::vector<std::vector<std::vector<uint8_t>>>(up_num_submatrices * 2,  // Double the submatrix dimension
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    // Fill up submatrices first (index 0 to up_num_submatrices-1)
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int sub = 0; sub < up_num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    up_gate_lut_2d_quantized[pos][sub][act_idx][weight_idx] = 
                        quantize_value(up_lut_2d[pos][sub][act_idx][weight_idx], up_scale, up_zeropoint);
                }
            }
        }
    }
    
    // Fill gate submatrices second (index up_num_submatrices to 2*up_num_submatrices-1)
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int sub = 0; sub < up_num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    up_gate_lut_2d_quantized[pos][sub + up_num_submatrices][act_idx][weight_idx] = 
                        quantize_value(gate_lut_2d[pos][sub][act_idx][weight_idx], gate_scale, gate_zeropoint);
                }
            }
        }
    }
    
    // Quantize down LUT
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> down_lut_2d_quantized(down_in_size,
        std::vector<std::vector<std::vector<uint8_t>>>(down_num_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int sub = 0; sub < down_num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    down_lut_2d_quantized[pos][sub][act_idx][weight_idx] = 
                        quantize_value(down_lut_2d[pos][sub][act_idx][weight_idx], down_scale, down_zeropoint);
                }
            }
        }
    }
    
    
    // Pack quantized LUT into hardware format (concatenated: up+gate, down)
    std::cout << "Packing quantized LUT into hardware format..." << std::endl;
    
    // Calculate total LUT size: FFN_LUT_SIZE = HIDDEN_DIM_DIV_2 * INTERM_DIM_MUL_2 + INTERM_DIM_DIV_2 * HIDDEN_DIM
    int up_gate_lut_size = up_in_size * up_num_submatrices * 2 * num_act_centroids * num_weight_centroids;  // Double for up+gate
    int down_lut_size = down_in_size * down_num_submatrices * num_act_centroids * num_weight_centroids;
    int total_lut_size = up_gate_lut_size + down_lut_size; 
    
    std::cout << "  Up+Gate LUT size: " << up_gate_lut_size << std::endl;
    std::cout << "  Down LUT size: " << down_lut_size << std::endl;
    std::cout << "  Total LUT size: " << total_lut_size << " (FFN_LUT_SIZE = " << FFN_LUT_SIZE << ")" << std::endl;
    std::cout << "  Note: Up and Gate LUTs are concatenated in submatrix dimension (up: 0-" << (up_num_submatrices-1) << ", gate: " << up_num_submatrices << "-" << (2*up_num_submatrices-1) << ")" << std::endl;
    
    
    // Pack into 8 hardware buffers (following LUT-DLA pattern)
    // Calculate total LUT vectors needed for hardware format
    int up_gate_lut_vectors = (up_in_size / 8) * up_num_submatrices * 2 * (num_act_centroids / 4);
    int down_lut_vectors = (down_in_size / 8) * down_num_submatrices * (num_act_centroids / 4);
    int total_lut_vectors = up_gate_lut_vectors + down_lut_vectors;
    
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> lut_hw(8);
    for (int buffer_idx = 0; buffer_idx < 8; buffer_idx++) {
        lut_hw[buffer_idx].resize(total_lut_vectors);
    }
    
    std::cout << "  Up+Gate LUT vectors: " << up_gate_lut_vectors << std::endl;
    std::cout << "  Down LUT vectors: " << down_lut_vectors << std::endl;
    std::cout << "  Total LUT vectors: " << total_lut_vectors << std::endl;
    
    int vector_offset = 0;
    
    // Pack up+gate LUT first (following LUT-DLA pattern)
    for (int pos = 0; pos < up_in_size; pos++) {
        int buffer_idx = pos % 8;
        int local_pos = pos / 8;
        
        for (int sub = 0; sub < up_num_submatrices * 2; sub++) {  // Iterate through all concatenated submatrices
            // Process groups of 4 activation centroids at a time (as expected by kernel)
            for (int act_group = 0; act_group < num_act_centroids / 4; act_group++) {
                int hw_idx = vector_offset + local_pos * up_num_submatrices * 2 * (num_act_centroids / 4) + act_group * up_num_submatrices * 2 + sub;
                
                // Pack 64 elements: 4 activation centroids x 16 weight centroids
                for (int k = 0; k < 16; k++) {  // 16 weight centroids
                    for (int ii = 0; ii < 4; ii++) {  // 4 activation centroids
                        int act_idx = act_group * 4 + ii;
                        if (act_idx < num_act_centroids && k < num_weight_centroids) {
                            int elem_idx = ii * 16 + k;  // Matches kernel: tmp[ii*16+k]
                            lut_hw[buffer_idx][hw_idx][elem_idx] = up_gate_lut_2d_quantized[pos][sub][act_idx][k];
                        } else {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = 0;  // Padding
                        }
                    }
                }
            }
        }
    }
    
    // Update vector offset for down LUT
    vector_offset += up_gate_lut_vectors;
    
    // Pack down LUT second (following LUT-DLA pattern)
    for (int pos = 0; pos < down_in_size; pos++) {
        int buffer_idx = pos % 8;
        int local_pos = pos / 8;
        
        for (int sub = 0; sub < down_num_submatrices; sub++) {
            // Process groups of 4 activation centroids at a time (as expected by kernel)
            for (int act_group = 0; act_group < num_act_centroids / 4; act_group++) {
                int hw_idx = vector_offset + local_pos * down_num_submatrices * (num_act_centroids / 4) + act_group * down_num_submatrices + sub;
                
                // Pack 64 elements: 4 activation centroids x 16 weight centroids
                for (int k = 0; k < 16; k++) {  // 16 weight centroids
                    for (int ii = 0; ii < 4; ii++) {  // 4 activation centroids
                        int act_idx = act_group * 4 + ii;
                        if (act_idx < num_act_centroids && k < num_weight_centroids) {
                            int elem_idx = ii * 16 + k;  // Matches kernel: tmp[ii*16+k]
                            lut_hw[buffer_idx][hw_idx][elem_idx] = down_lut_2d_quantized[pos][sub][act_idx][k];
                        } else {
                            int elem_idx = ii * 16 + k;
                            lut_hw[buffer_idx][hw_idx][elem_idx] = 0;  // Padding
                        }
                    }
                }
            }
        }
    }
    
    // Create concatenated up+gate weight indices (concatenated in submatrix dimension: first up, then gate)
    std::vector<std::vector<std::vector<int>>> up_gate_weight_indices(up_in_size,
        std::vector<std::vector<int>>(up_num_submatrices * 2, std::vector<int>(512)));  // Double the submatrix dimension
    
    // Fill up weight indices first (index 0 to up_num_submatrices-1)
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int sub = 0; sub < up_num_submatrices; sub++) {
            for (int i = 0; i < 512; i++) {
                up_gate_weight_indices[pos][sub][i] = up_weight_indices[pos][sub][i];
            }
        }
    }
    
    // Fill gate weight indices second (index up_num_submatrices to 2*up_num_submatrices-1)
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int sub = 0; sub < up_num_submatrices; sub++) {
            for (int i = 0; i < 512; i++) {
                up_gate_weight_indices[pos][sub + up_num_submatrices][i] = gate_weight_indices[pos][sub][i];
            }
        }
    }
    
    // Pack weight indices into 8 hardware buffers (following LUT-DLA pattern)
    std::cout << "Packing weight indices into hardware format..." << std::endl;
    std::cout << "  Note: Up and Gate weight indices are concatenated in submatrix dimension (up: 0-" << (up_num_submatrices-1) << ", gate: " << up_num_submatrices << "-" << (2*up_num_submatrices-1) << ")" << std::endl;
    
    // Calculate total weight index vectors needed for hardware format
    int up_gate_weight_vectors = (up_in_size / 8) * up_num_submatrices * 2 * 4;  // *2 for up+gate, *2 for vec_idx
    int down_weight_vectors = (down_in_size / 8) * down_num_submatrices * 4;     // *2 for vec_idx
    int total_weight_vectors = up_gate_weight_vectors + down_weight_vectors;
    
    std::vector<std::vector<tapa::vec_t<ap_uint<8>, 64>>> weight_idx_hw(8);
    for (int buffer_idx = 0; buffer_idx < 8; buffer_idx++) {
        weight_idx_hw[buffer_idx].resize(total_weight_vectors);
    }
    
    std::cout << "  Up+Gate weight vectors: " << up_gate_weight_vectors << std::endl;
    std::cout << "  Down weight vectors: " << down_weight_vectors << std::endl;
    std::cout << "  Total weight vectors: " << total_weight_vectors << std::endl;
    
    vector_offset = 0;
    
    // Pack up+gate weight indices first (following LUT-DLA pattern)
    for (int pos = 0; pos < up_in_size; pos++) {
        int buffer_idx = pos % 8;
        int local_pos = pos / 8;
        
        for (int sub = 0; sub < up_num_submatrices * 2; sub++) {  // Iterate through all concatenated submatrices
            for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                int hw_idx = vector_offset + local_pos * up_num_submatrices * 2 * 4 + sub * 4 + vec_idx;
                for (int k = 0; k < 64; k++) {
                    int col = vec_idx * 128 + k * 2;
                    if (col < 512) {
                        ap_uint<8> tmp_idx;
                        tmp_idx(3, 0) = up_gate_weight_indices[pos][sub][col];
                        tmp_idx(7, 4) = up_gate_weight_indices[pos][sub][col + 1];
                        weight_idx_hw[buffer_idx][hw_idx][k] = tmp_idx;
                    } else {
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0; // Padding
                    }
                }
            }
        }
    }
    
    // Update vector offset for down weight indices
    vector_offset += up_gate_weight_vectors;
    
    // Pack down weight indices second (following LUT-DLA pattern)
    for (int pos = 0; pos < down_in_size; pos++) {
        int buffer_idx = pos % 8;
        int local_pos = pos / 8;
        
        for (int sub = 0; sub < down_num_submatrices; sub++) {
            for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                int hw_idx = vector_offset + local_pos * down_num_submatrices * 4 + sub * 4 + vec_idx;
                for (int k = 0; k < 64; k++) {
                    int col = vec_idx * 128 + k * 2;
                    if (col < 512) {
                        ap_uint<8> tmp_idx;
                        tmp_idx(3, 0) = down_weight_indices[pos][sub][col];
                        tmp_idx(7, 4) = down_weight_indices[pos][sub][col + 1];
                        weight_idx_hw[buffer_idx][hw_idx][k] = tmp_idx;
                    } else {
                        weight_idx_hw[buffer_idx][hw_idx][k] = 0; // Padding
                    }
                }
            }
        }
    }
    
    // Allocate output array
    int output_elements = L * HIDDEN_DIM;
    int output_vectors = output_elements / 16;
    std::vector<tapa::vec_t<float, 16>> output_hw_raw(output_vectors);
    std::vector<int> cycle_count_hw(1);
    
    
    // Compute reference results using weight vector quantization
    std::cout << "Computing reference results with weight vector quantization..." << std::endl;
    std::vector<std::vector<float>> output_ref(L, std::vector<float>(HIDDEN_DIM));
    reference_ffn_with_weight_vq(input, up_act_centroids, up_weight_centroids, up_weight_indices,
                                up_act_centroids, gate_weight_centroids, gate_weight_indices,  // Note: same activation centroids for gate
                                down_act_centroids, down_weight_centroids, down_weight_indices,
                                output_ref, L, false);  // Use piece-wise SiLU
    
    // Compute reference results using quantized LUT
    std::cout << "Computing reference results with quantized LUT..." << std::endl;
    std::vector<std::vector<float>> output_ref_quant(L, std::vector<float>(HIDDEN_DIM));
    
    // Extract individual projection LUTs from concatenated structures
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> up_lut_extracted(up_in_size,
        std::vector<std::vector<std::vector<uint8_t>>>(up_num_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    std::vector<std::vector<std::vector<std::vector<uint8_t>>>> gate_lut_extracted(up_in_size,
        std::vector<std::vector<std::vector<uint8_t>>>(up_num_submatrices,
            std::vector<std::vector<uint8_t>>(num_act_centroids, std::vector<uint8_t>(num_weight_centroids))));
    
    // Extract up and gate LUTs from concatenated structure
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int sub = 0; sub < up_num_submatrices; sub++) {
            for (int act_idx = 0; act_idx < num_act_centroids; act_idx++) {
                for (int weight_idx = 0; weight_idx < num_weight_centroids; weight_idx++) {
                    up_lut_extracted[pos][sub][act_idx][weight_idx] = up_gate_lut_2d_quantized[pos][sub][act_idx][weight_idx];
                    gate_lut_extracted[pos][sub][act_idx][weight_idx] = up_gate_lut_2d_quantized[pos][sub + up_num_submatrices][act_idx][weight_idx];
                }
            }
        }
    }
    
    reference_ffn_with_quantized_lut(input, up_act_centroids, up_weight_indices, up_lut_extracted, up_scale, up_zeropoint,
                                     up_act_centroids, gate_weight_indices, gate_lut_extracted, gate_scale, gate_zeropoint,
                                     down_act_centroids, down_weight_indices, down_lut_2d_quantized, down_scale, down_zeropoint,
                                     output_ref_quant, L, false);  // Use piece-wise SiLU
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    // Debug logging for input values (first sequence, first 8 elements)
    std::cout << "[DEBUG] Input values (seq 0, first 8 elements): ";
    for (int j = 0; j < std::min(8, HIDDEN_DIM); j++) {
        std::cout << std::fixed << std::setprecision(6) << input[0][j] << " ";
    }
    std::cout << std::endl;
    
    tapa::invoke(ffn_core, FLAGS_bitstream,
                L,
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(input_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(centroid_hw),
                tapa::read_only_mmaps<tapa::vec_t<ap_uint<8>, 64>, 8>(lut_hw),
                tapa::read_only_mmaps<tapa::vec_t<ap_uint<8>, 64>, 8>(weight_idx_hw),
                tapa::read_only_mmap<ap_uint<64>>(scale_zero_hw),
                tapa::write_only_mmap<tapa::vec_t<float, 16>>(output_hw_raw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Convert hardware output from tapa::vec_t<float, 16> vectors to 2D array
    std::cout << "Converting hardware output..." << std::endl;
    std::vector<std::vector<float>> output_hw(L, std::vector<float>(HIDDEN_DIM));
    
    // Hardware writes in sequence-major order
    for (int i = 0; i < (HIDDEN_DIM / 16); i++) {
        for (int j = 0; j < L; j++) {
            int vec_idx = i * L + j;
            for (int k = 0; k < 16; k++) {
                output_hw[j][i * 16 + k] = output_hw_raw[vec_idx][k];
            }
        }
    }
    
    // Verify results against reference
    std::cout << "Verifying results..." << std::endl;
    
    // Compare hardware vs floating-point reference
    int errors_fp = 0;
    float max_error_fp = 0.0f;
    float tolerance_fp = 5e-2f;  // Relaxed tolerance for FP comparison
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float diff = std::abs(output_hw[i][j] - output_ref[i][j]);
            if (diff > max_error_fp) {
                max_error_fp = diff;
            }
            
            if (!isClose(output_hw[i][j], output_ref[i][j], tolerance_fp)) {
                errors_fp++;
                if (errors_fp <= 5) {  // Print first 5 errors
                    std::cout << "FP Error at [" << i << "][" << j << "]: HW=" 
                             << output_hw[i][j] << ", FP_REF=" << output_ref[i][j] 
                             << ", diff=" << diff << std::endl;
                }
            }
        }
    }
    
    // Compare hardware vs quantized reference
    int errors_quant = 0;
    float max_error_quant = 0.0f;
    float tolerance_quant = 5e-1f;  // Tight tolerance - should match very closely
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
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

    // Find and print top 5 quantization errors
    std::vector<std::tuple<float, int, int, float, float>> quant_errors; // (error, i, j, hw_val, ref_val)
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float diff = std::abs(output_hw[i][j] - output_ref_quant[i][j]);
            if (diff > 0) {
                quant_errors.push_back(std::make_tuple(diff, i, j, output_hw[i][j], output_ref_quant[i][j]));
            }
        }
    }
    
    // Sort by error magnitude (descending)
    std::sort(quant_errors.begin(), quant_errors.end(), 
              [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    
    // Print top 5 errors
    std::cout << "\n=== TOP 5 QUANTIZATION ERRORS ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int k = 0; k < std::min(5, (int)quant_errors.size()); k++) {
        auto [error, i, j, hw_val, ref_val] = quant_errors[k];
        std::cout << "Error #" << (k+1) << ":" << std::endl;
        std::cout << "  Position: [" << i << "][" << j << "]" << std::endl;
        std::cout << "  HW value: " << hw_val << std::endl;
        std::cout << "  Quantized REF value: " << ref_val << std::endl;
        std::cout << "  Error magnitude: " << error << std::endl;
        std::cout << "  Relative error: " << (ref_val != 0 ? error/std::abs(ref_val) * 100 : 0) << "%" << std::endl;
        std::cout << std::endl;
    }
    
    // Compare floating-point vs quantized to show quantization impact
    int quant_diff_count = 0;
    float max_quant_impact = 0.0f;
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float diff = std::abs(output_ref[i][j] - output_ref_quant[i][j]);
            if (diff > max_quant_impact) {
                max_quant_impact = diff;
            }
            
            if (!isClose(output_ref[i][j], output_ref_quant[i][j], 1e-3f)) {
                quant_diff_count++;
            }
        }
    }
    
    std::cout << "\n=== RESULTS SUMMARY ===" << std::endl;
    std::cout << "Maximum error (HW vs FP reference): " << max_error_fp << std::endl;
    std::cout << "Maximum error (HW vs Quantized reference): " << max_error_quant << std::endl;
    std::cout << "Maximum quantization impact (FP vs Quantized): " << max_quant_impact << std::endl;
    
    if (errors_fp == 0) {
        std::cout << "SUCCESS: Hardware matches floating-point reference within tolerance!" << std::endl;
    } else {
        std::cout << "WARNING: " << errors_fp << " out of " << (L * HIDDEN_DIM) 
                 << " results don't match floating-point reference (expected due to quantization)!" << std::endl;
    }
    
    if (errors_quant == 0) {
        std::cout << "SUCCESS: Hardware matches quantized reference exactly!" << std::endl;
    } else {
        std::cout << "FAILURE: " << errors_quant << " out of " << (L * HIDDEN_DIM) 
                 << " results don't match quantized reference!" << std::endl;
    }
    
    std::cout << "Quantization impact: " << quant_diff_count << " out of " << (L * HIDDEN_DIM) 
             << " values differ between FP and quantized references" << std::endl;
    
    // Print some sample results
    std::cout << "\nSample results (first sequence, first 10 outputs):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < std::min(10, HIDDEN_DIM); j++) {
        std::cout << "Output [0][" << j << "]:" << std::endl;
        std::cout << "  HW:         " << output_hw[0][j] << std::endl;
        std::cout << "  FP_REF:     " << output_ref[0][j] << std::endl;
        std::cout << "  QUANT_REF:  " << output_ref_quant[0][j] << std::endl;
        std::cout << "  HW-FP diff: " << std::abs(output_hw[0][j] - output_ref[0][j]) << std::endl;
        std::cout << "  HW-Q diff:  " << std::abs(output_hw[0][j] - output_ref_quant[0][j]) << std::endl;
        std::cout << std::endl;
    }
    
    // Print SiLU activation comparison for some sample values
    std::cout << "\nSiLU activation comparison (sample values):" << std::endl;
    std::vector<float> sample_values = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    for (float val : sample_values) {
        float direct = silu_direct(val);
        float piecewise = silu_piecewise(val);
        std::cout << "SiLU(" << val << "): direct=" << direct 
                 << ", piecewise=" << piecewise 
                 << ", diff=" << std::abs(direct - piecewise) << std::endl;
    }
    
    // Print statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total input elements: " << (L * HIDDEN_DIM) << std::endl;
    std::cout << "  Total activation centroids: " << ((up_in_size + down_in_size) * num_act_centroids) << std::endl;
    std::cout << "  Total weight centroids: " << (up_in_size * up_num_submatrices * num_weight_centroids * 2 + down_in_size * down_num_submatrices * num_weight_centroids) << std::endl;
    //std::cout << "  Total LUT entries: " << total_lut_size << std::endl;
    //std::cout << "  Total weight indices: " << total_weight_size << std::endl;
    std::cout << "  Total output elements: " << (L * HIDDEN_DIM) << std::endl;
    std::cout << "  Memory usage:" << std::endl;
    std::cout << "    Input: " << (input_hw.size() * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Centroids: " << (centroid_hw.size() * 16 * sizeof(float)) << " bytes" << std::endl;
    //std::cout << "    LUTs: " << (total_lut_size * sizeof(uint8_t)) << " bytes" << std::endl;
    //std::cout << "    Weight indices: " << (total_weight_size * sizeof(uint8_t)) << " bytes" << std::endl;
    std::cout << "    Output: " << (output_hw_raw.size() * 16 * sizeof(float)) << " bytes" << std::endl;
    
    return errors_quant == 0 ? 0 : 1;
}
