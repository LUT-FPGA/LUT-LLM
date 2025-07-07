#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
#include "ffn.h"

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
bool isClose(float a, float b, float tolerance = 1e-4) {
    return std::abs(a - b) < tolerance;
}

// Direct SiLU activation function
float silu_direct(float x) {
    return x / (1.0f + std::exp(-x));
}

// Piece-wise SiLU activation function (for testing against hardware)
float silu_piecewise(float x) {
    // This should match the hardware implementation's piece-wise approximation
    // For now, using direct computation - actual hardware may use lookup tables
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
            // Extract 2-element vector from input
            std::vector<float> input_vec = {input[i][pos*2], input[i][pos*2 + 1]};
            
            // Find closest centroid for this position
            int centroid_idx = find_closest_centroid(input_vec, centroids[pos]);
            
            // Accumulate using LUT values
            for (int j = 0; j < out_dim; j++) {
                output[i][j] += lut[pos][centroid_idx][j];
            }
        }
    }
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

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters
    const int L = 32;              // Sequence length
    const int num_centroids = 64;   // Number of centroids per position
    const int vector_dim = 2;       // Dimension of each centroid
    
    std::cout << "Testing FFN kernel with:" << std::endl;
    std::cout << "  L (sequence length): " << L << std::endl;
    std::cout << "  Hidden dimension: " << HIDDEN_DIM << std::endl;
    std::cout << "  Intermediate dimension: " << INTERM_DIM << std::endl;
    std::cout << "  Number of centroids per position: " << num_centroids << std::endl;
    std::cout << "  Vector dimension: " << vector_dim << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> centroid_dis(-10.0f, 10.0f);
    std::uniform_real_distribution<float> lut_dis(-1.0f, 1.0f);
    std::uniform_real_distribution<float> input_dis(-10.0f, 10.0f);
    
    // Generate random input
    std::vector<std::vector<float>> input(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            input[i][j] = input_dis(gen);
        }
    }
    
    // Pack input into hardware format
    std::vector<tapa::vec_t<float, 2>> input_hw(HIDDEN_DIM_DIV_2 * L);
    for (int pos = 0; pos < HIDDEN_DIM_DIV_2; pos++) {
        for (int i = 0; i < L; i++) {
            int hw_idx = pos * L + i;
            input_hw[hw_idx][0] = input[i][pos * 2];
            input_hw[hw_idx][1] = input[i][pos * 2 + 1];
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
    
    // Pack centroids into hardware format
    std::cout << "Packing centroids into hardware format..." << std::endl;
    
    // Up centroids
    std::vector<tapa::vec_t<float, 16>> up_centroid_hw(up_in_size * 8);
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                int vec_idx = pos * 8 + (i / 8);
                int elem_idx = (i % 8) * 2 + j;
                up_centroid_hw[vec_idx][elem_idx] = up_centroids[pos][i][j];
            }
        }
    }
    
    // Gate centroids
    std::vector<tapa::vec_t<float, 16>> gate_centroid_hw(up_in_size * 8);
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                int vec_idx = pos * 8 + (i / 8);
                int elem_idx = (i % 8) * 2 + j;
                gate_centroid_hw[vec_idx][elem_idx] = gate_centroids[pos][i][j];
            }
        }
    }
    
    // Down centroids
    std::vector<tapa::vec_t<float, 16>> down_centroid_hw(down_in_size * 8);
    for (int pos = 0; pos < down_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                int vec_idx = pos * 8 + (i / 8);
                int elem_idx = (i % 8) * 2 + j;
                down_centroid_hw[vec_idx][elem_idx] = down_centroids[pos][i][j];
            }
        }
    }
    
    // Pack LUTs into hardware format
    std::cout << "Packing LUTs into hardware format..." << std::endl;
    
    // Up LUT
    int up_lut_elements = up_in_size * num_centroids * INTERM_DIM;
    int up_lut_vectors = up_lut_elements / 16;
    std::vector<tapa::vec_t<float, 16>> up_lut_hw(up_lut_vectors);
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < INTERM_DIM; j++) {
                int linear_idx = pos * (num_centroids * INTERM_DIM) + i * INTERM_DIM + j;
                int vec_idx = linear_idx / 16;
                int elem_idx = linear_idx % 16;
                up_lut_hw[vec_idx][elem_idx] = up_lut[pos][i][j];
            }
        }
    }
    
    // Gate LUT
    std::vector<tapa::vec_t<float, 16>> gate_lut_hw(up_lut_vectors);
    for (int pos = 0; pos < up_in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < INTERM_DIM; j++) {
                int linear_idx = pos * (num_centroids * INTERM_DIM) + i * INTERM_DIM + j;
                int vec_idx = linear_idx / 16;
                int elem_idx = linear_idx % 16;
                gate_lut_hw[vec_idx][elem_idx] = gate_lut[pos][i][j];
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
                int linear_idx = pos * (num_centroids * HIDDEN_DIM) + i * HIDDEN_DIM + j;
                int vec_idx = linear_idx / 16;
                int elem_idx = linear_idx % 16;
                down_lut_hw[vec_idx][elem_idx] = down_lut[pos][i][j];
            }
        }
    }
    
    // Allocate output arrays
    int output_vectors = (L * HIDDEN_DIM) / 16;
    std::vector<tapa::vec_t<float, 16>> output_hw_raw(output_vectors);
    std::vector<int> cycle_count_hw(1);
    
    // Compute reference results with piece-wise SiLU
    std::cout << "Computing reference results with piece-wise SiLU..." << std::endl;
    std::vector<std::vector<float>> output_ref_piecewise(L, std::vector<float>(HIDDEN_DIM));
    reference_ffn(input, up_centroids, up_lut, gate_centroids, gate_lut, 
                  down_centroids, down_lut, output_ref_piecewise, L, false);
    
    // Compute reference results with direct SiLU for error measurement
    std::cout << "Computing reference results with direct SiLU..." << std::endl;
    std::vector<std::vector<float>> output_ref_direct(L, std::vector<float>(HIDDEN_DIM));
    reference_ffn(input, up_centroids, up_lut, gate_centroids, gate_lut, 
                  down_centroids, down_lut, output_ref_direct, L, true);
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(ffn_core, FLAGS_bitstream,
                L,
                tapa::read_only_mmap<tapa::vec_t<float, 2>>(input_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(up_centroid_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(gate_centroid_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(down_centroid_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(up_lut_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(gate_lut_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(down_lut_hw),
                tapa::write_only_mmap<tapa::vec_t<float, 16>>(output_hw_raw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    
    // Convert hardware output to structured format
    std::cout << "Converting hardware output..." << std::endl;
    std::vector<std::vector<float>> output_hw(L, std::vector<float>(HIDDEN_DIM));
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            int linear_idx = i * HIDDEN_DIM + j;
            int vec_idx = linear_idx / 16;
            int elem_idx = linear_idx % 16;
            output_hw[i][j] = output_hw_raw[vec_idx][elem_idx];
        }
    }
    
    // Verify results against piece-wise SiLU reference
    std::cout << "Verifying results against piece-wise SiLU reference..." << std::endl;
    int errors = 0;
    float max_error = 0.0f;
    float tolerance = 1e-3f;
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float diff = std::abs(output_hw[i][j] - output_ref_piecewise[i][j]);
            if (diff > max_error) {
                max_error = diff;
            }
            
            if (!isClose(output_hw[i][j], output_ref_piecewise[i][j], tolerance)) {
                errors++;
                if (errors <= 10) {
                    std::cout << "Error at [" << i << "][" << j << "]: HW=" 
                             << output_hw[i][j] << ", REF=" << output_ref_piecewise[i][j] 
                             << ", diff=" << diff << std::endl;
                }
            }
        }
    }
    
    std::cout << "Maximum error (vs piece-wise SiLU): " << max_error << std::endl;
    
    // Also measure error against direct SiLU
    std::cout << "Measuring error against direct SiLU reference..." << std::endl;
    int direct_errors = 0;
    float max_direct_error = 0.0f;
    float direct_tolerance = 1e-2f;  // More relaxed tolerance
    
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < HIDDEN_DIM; j++) {
            float diff = std::abs(output_hw[i][j] - output_ref_direct[i][j]);
            if (diff > max_direct_error) {
                max_direct_error = diff;
            }
            
            if (!isClose(output_hw[i][j], output_ref_direct[i][j], direct_tolerance)) {
                direct_errors++;
            }
        }
    }
    
    std::cout << "Maximum error (vs direct SiLU): " << max_direct_error << std::endl;
    
    if (errors == 0) {
        std::cout << "SUCCESS: All " << (L * HIDDEN_DIM) 
                 << " results match piece-wise SiLU reference within tolerance!" << std::endl;
    } else {
        std::cout << "FAILURE: " << errors << " out of " << (L * HIDDEN_DIM) 
                 << " results don't match piece-wise SiLU reference!" << std::endl;
    }
    
    std::cout << "Direct SiLU comparison: " << direct_errors << " out of " << (L * HIDDEN_DIM) 
             << " results exceed direct SiLU tolerance." << std::endl;
    
    // Print some sample results
    std::cout << "\nSample results (first sequence, first 10 outputs):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < std::min(10, HIDDEN_DIM); j++) {
        std::cout << "Output [0][" << j << "]: HW=" << output_hw[0][j] 
                 << ", REF_PW=" << output_ref_piecewise[0][j]
                 << ", REF_DIRECT=" << output_ref_direct[0][j]
                 << ", diff_PW=" << std::abs(output_hw[0][j] - output_ref_piecewise[0][j])
                 << ", diff_DIRECT=" << std::abs(output_hw[0][j] - output_ref_direct[0][j]) << std::endl;
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
    std::cout << "  Total centroids: " << ((up_in_size + down_in_size) * num_centroids) << std::endl;
    std::cout << "  Total LUT entries: " << (up_lut_elements + up_lut_elements + down_lut_elements) << std::endl;
    std::cout << "  Total output elements: " << (L * HIDDEN_DIM) << std::endl;
    std::cout << "  Memory usage:" << std::endl;
    std::cout << "    Input: " << (input_hw.size() * 2 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Centroids: " << ((up_centroid_hw.size() + gate_centroid_hw.size() + down_centroid_hw.size()) * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    LUTs: " << ((up_lut_hw.size() + gate_lut_hw.size() + down_lut_hw.size()) * 16 * sizeof(float)) << " bytes" << std::endl;
    std::cout << "    Output: " << (output_hw_raw.size() * 16 * sizeof(float)) << " bytes" << std::endl;
    
    return errors == 0 ? 0 : 1;
}
