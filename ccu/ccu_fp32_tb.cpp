#include <gflags/gflags.h>
#include <tapa.h>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include "ccu_fp32.h"

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

// Reference implementation for verification
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

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Test parameters
    const int L = 100;  // Number of input vectors
    const int num_centroids = 64;
    const int vector_dim = 2;
    
    std::cout << "Testing CCU FP32 with " << L << " input vectors and " 
              << num_centroids << " centroids" << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    // Generate random centroids (64 centroids, each with 2 elements)
    std::vector<std::vector<float>> centroids_ref(num_centroids, std::vector<float>(vector_dim));
    std::vector<tapa::vec_t<float, 16>> centroids_hw(8);  // 8 vectors of 16 floats each
    
    for (int i = 0; i < num_centroids; ++i) {
        for (int j = 0; j < vector_dim; ++j) {
            float val = dis(gen);
            centroids_ref[i][j] = val;
            
            // Pack into hardware format: 8 reads of 16 floats
            // Each centroid has 2 elements, so 8 centroids per 16-float vector
            int vec_idx = i / 8;  // Which 16-float vector
            int elem_idx = (i % 8) * 2 + j;  // Position within the 16-float vector
            centroids_hw[vec_idx][elem_idx] = val;
        }
    }
    
    // Generate random input vectors
    std::vector<tapa::vec_t<float, 2>> input_hw(L);
    std::vector<std::vector<float>> input_ref(L, std::vector<float>(vector_dim));
    
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < vector_dim; ++j) {
            float val = dis(gen);
            input_hw[i][j] = val;
            input_ref[i][j] = val;
        }
    }
    
    // Allocate output arrays
    std::vector<int> idx_out_hw(L);
    std::vector<int> idx_out_ref(L);

    std::vector<int> cycle_count_hw(1);
    
    // Compute reference results
    std::cout << "Computing reference results..." << std::endl;
    for (int i = 0; i < L; ++i) {
        idx_out_ref[i] = find_closest_centroid(input_ref[i], centroids_ref);
    }
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(ccu_fp32_top, FLAGS_bitstream,
                L,
                tapa::read_only_mmap<tapa::vec_t<float, 2>>(input_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(centroids_hw),
                tapa::write_only_mmap<int>(idx_out_hw),
                tapa::write_only_mmap<int>(cycle_count_hw));
    
    std::cout << "Cycle count: " << cycle_count_hw[0] << std::endl;
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    int errors = 0;
    for (int i = 0; i < L; ++i) {
        if (idx_out_hw[i] != idx_out_ref[i]) {
            errors++;
            if (errors <= 10) {  // Print first 10 errors
                std::cout << "Error at index " << i << ": HW=" << idx_out_hw[i] 
                         << ", REF=" << idx_out_ref[i] << std::endl;
                
                // Print input vector and distances for debugging
                std::cout << "  Input vector: [" << input_ref[i][0] << ", " << input_ref[i][1] << "]" << std::endl;
                std::cout << "  HW chosen centroid " << idx_out_hw[i] << ": ["
                         << centroids_ref[idx_out_hw[i]][0] << ", " 
                         << centroids_ref[idx_out_hw[i]][1] << "] dist="
                         << chebyshev_distance(input_ref[i], centroids_ref[idx_out_hw[i]]) << std::endl;
                std::cout << "  REF chosen centroid " << idx_out_ref[i] << ": ["
                         << centroids_ref[idx_out_ref[i]][0] << ", " 
                         << centroids_ref[idx_out_ref[i]][1] << "] dist="
                         << chebyshev_distance(input_ref[i], centroids_ref[idx_out_ref[i]]) << std::endl;
            }
        }
    }
    
    if (errors == 0) {
        std::cout << "SUCCESS: All " << L << " results match!" << std::endl;
    } else {
        std::cout << "FAILURE: " << errors << " out of " << L << " results don't match!" << std::endl;
    }
    
    // Print some sample results
    std::cout << "\nSample results (first 10):" << std::endl;
    for (int i = 0; i < std::min(10, L); ++i) {
        std::cout << "Input " << i << ": [" << input_ref[i][0] << ", " << input_ref[i][1] 
                 << "] -> Centroid " << idx_out_hw[i] << " ["
                 << centroids_ref[idx_out_hw[i]][0] << ", " 
                 << centroids_ref[idx_out_hw[i]][1] << "]" << std::endl;
    }
    
    return errors == 0 ? 0 : 1;
}
