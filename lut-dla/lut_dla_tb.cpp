#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <iomanip>
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

// Reference implementation: Combined CCU + IMM functionality
void reference_lut_dla(
    const std::vector<std::vector<std::vector<float>>>& input_vectors,  // L x in_size x vector_dim
    const std::vector<std::vector<std::vector<float>>>& centroids,      // in_size x num_centroids x vector_dim
    const std::vector<std::vector<std::vector<float>>>& lut,            // in_size x num_centroids x out_size
    std::vector<std::vector<float>>& output                             // L x out_size
) {
    int L = input_vectors.size();
    int in_size = input_vectors[0].size();
    int out_size = lut[0][0].size();
    
    // Initialize output
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < out_size; j++) {
            output[i][j] = 0.0f;
        }
    }
    
    // For each sequence and each position
    for (int i = 0; i < L; i++) {
        for (int pos = 0; pos < in_size; pos++) {
            // Find closest centroid for this position (CCU functionality)
            int centroid_idx = find_closest_centroid(input_vectors[i][pos], centroids[pos]);
            
            // Accumulate using LUT values (IMM functionality)
            for (int j = 0; j < out_size; j++) {
                output[i][j] += lut[pos][centroid_idx][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters - matching the configuration from both testbenches
    const int L = 128;              // Number of input sequences (sequence length)
    const int input_dim = 8;        // Input dimension for weight matrix
    const int out_size = 896;      // Output dimension
    const int vector_dim = 2;       // Dimension of each centroid (2-element vectors)
    const int in_size = input_dim / vector_dim;  // Number of 2-element positions = 4
    const int num_centroids = 64;   // Number of centroids per position
    
    std::cout << "Testing LUT-DLA kernel with:" << std::endl;
    std::cout << "  L (sequence length): " << L << std::endl;
    std::cout << "  Input dimension: " << input_dim << std::endl;
    std::cout << "  in_size (number of 2-element positions): " << in_size << std::endl;
    std::cout << "  Number of centroids per position: " << num_centroids << std::endl;
    std::cout << "  Vector dimension: " << vector_dim << std::endl;
    std::cout << "  Output dimension: " << out_size << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> centroid_dis(-10.0f, 10.0f);
    std::uniform_real_distribution<float> lut_dis(-0.5f, 0.5f);
    std::uniform_real_distribution<float> input_dis(-10.0f, 10.0f);
    
    // Generate random centroids for each 2-element position
    // centroids[position][centroid_idx][dim] - different centroids for each position
    std::vector<std::vector<std::vector<float>>> centroids(in_size, 
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(vector_dim)));
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                centroids[pos][i][j] = centroid_dis(gen);
            }
        }
    }
    
    // Pack centroids into hardware format (vec_t<float, 16>)
    // Each position has 64 centroids, each with 2 elements
    // Need 8 reads of 16 floats for each position: 8 centroids per 16-float vector
    std::vector<tapa::vec_t<float, 16>> centroids_hw(in_size * 8);
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < vector_dim; j++) {
                int vec_idx = pos * 8 + (i / 8);           // Which 16-float vector
                int elem_idx = (i % 8) * 2 + j;            // Position within the vector
                centroids_hw[vec_idx][elem_idx] = centroids[pos][i][j];
            }
        }
    }
    
    // Generate random input vectors (L sequences x in_size positions x vector_dim)
    std::vector<std::vector<std::vector<float>>> input_vectors(L, 
        std::vector<std::vector<float>>(in_size, std::vector<float>(vector_dim)));
    
    for (int i = 0; i < L; i++) {
        for (int pos = 0; pos < in_size; pos++) {
            for (int j = 0; j < vector_dim; j++) {
                input_vectors[i][pos][j] = input_dis(gen);
            }
        }
    }
    
    // Pack input into hardware format
    std::vector<tapa::vec_t<float, 2>> input_hw(L * in_size);
    for (int pos = 0; pos < in_size; pos++) {
        for (int i = 0; i < L; i++) {
            int hw_idx = pos * L + i;
            input_hw[hw_idx][0] = input_vectors[i][pos][0];
            input_hw[hw_idx][1] = input_vectors[i][pos][1];
        }
    }
    
    // Generate random LUT values for each position and centroid
    // lut[position][centroid_idx][output_idx] 
    std::cout << "Generating lookup tables for all positions..." << std::endl;
    std::vector<std::vector<std::vector<float>>> lut(in_size, 
        std::vector<std::vector<float>>(num_centroids, std::vector<float>(out_size)));
    
    for (int pos = 0; pos < in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < out_size; j++) {
                lut[pos][i][j] = lut_dis(gen);
            }
        }
    }
    
    // Pack LUT into hardware format (vec_t<float, 16>)
    // Total elements: in_size * num_centroids * out_size
    int total_lut_elements = in_size * num_centroids * out_size;
    int num_lut_vectors = total_lut_elements / 16;
    std::vector<tapa::vec_t<float, 16>> lut_hw(num_lut_vectors);
    
    std::cout << "Packing LUT into hardware format..." << std::endl;
    for (int pos = 0; pos < in_size; pos++) {
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < out_size; j++) {
                int linear_idx = pos * (num_centroids * out_size) + i * out_size + j;
                int vec_idx = linear_idx / 16;
                int elem_idx = linear_idx % 16;
                lut_hw[vec_idx][elem_idx] = lut[pos][i][j];
            }
        }
    }
    
    // Allocate output arrays
    // Hardware output: (L * out_size) / 16 vectors of tapa::vec_t<float, 16>
    int output_elements = L * out_size;
    int num_output_vectors = output_elements / 16;
    std::vector<tapa::vec_t<float, 16>> output_hw_raw(num_output_vectors);
    
    std::vector<int> cycle_count_hw(1);
    
    // Compute reference results
    std::cout << "Computing reference results..." << std::endl;
    std::vector<std::vector<float>> output_ref(L, std::vector<float>(out_size));
    reference_lut_dla(input_vectors, centroids, lut, output_ref);
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(lut_dla_core, FLAGS_bitstream,
                L,
                in_size,
                out_size,
                tapa::read_only_mmap<tapa::vec_t<float, 2>>(input_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(centroids_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(lut_hw),
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
    float tolerance = 1e-4f;  // Tolerance for floating point comparison
    
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
                    
                    // Print debugging info
                    std::cout << "  Input sequence: " << i << ", output position: " << j << std::endl;
                    std::cout << "  First few input vectors: ";
                    for (int pos = 0; pos < std::min(3, in_size); pos++) {
                        std::cout << "[" << input_vectors[i][pos][0] << "," << input_vectors[i][pos][1] << "] ";
                    }
                    std::cout << std::endl;
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
    
    // Print some sample results
    std::cout << "\nSample results (first sequence, first 10 outputs):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int j = 0; j < std::min(10, out_size); j++) {
        std::cout << "Output [0][" << j << "]: HW=" << output_hw[0][j] 
                 << ", REF=" << output_ref[0][j] 
                 << ", diff=" << std::abs(output_hw[0][j] - output_ref[0][j]) << std::endl;
    }
    
    // Print some sample input-to-centroid mappings for first sequence
    std::cout << "\nSample input-to-centroid mappings (first sequence, all positions):" << std::endl;
    for (int pos = 0; pos < in_size; pos++) {
        int centroid_idx = find_closest_centroid(input_vectors[0][pos], centroids[pos]);
        std::cout << "Position " << pos << ": Input [" << input_vectors[0][pos][0] << ", " << input_vectors[0][pos][1] 
                 << "] -> Centroid " << centroid_idx << " ["
                 << centroids[pos][centroid_idx][0] << ", " << centroids[pos][centroid_idx][1] 
                 << "] (dist=" << chebyshev_distance(input_vectors[0][pos], centroids[pos][centroid_idx]) << ")" << std::endl;
    }
    
    // Print first sequence summary
    std::cout << "\nFirst sequence summary:" << std::endl;
    std::cout << "Total positions: " << in_size << std::endl;
    std::cout << "All centroid indices: ";
    for (int pos = 0; pos < in_size; pos++) {
        int centroid_idx = find_closest_centroid(input_vectors[0][pos], centroids[pos]);
        std::cout << centroid_idx << " ";
    }
    std::cout << std::endl;
    
    // Print statistics
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total input vectors: " << L * in_size << std::endl;
    std::cout << "  Total centroids: " << in_size * num_centroids << std::endl;
    std::cout << "  Total LUT entries: " << total_lut_elements << std::endl;
    std::cout << "  Total output elements: " << output_elements << std::endl;
    
    return errors == 0 ? 0 : 1;
}
