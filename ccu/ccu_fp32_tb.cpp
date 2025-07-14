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
    const int L = 128;  // Number of input vectors per stream
    const int num_streams = 8;  // Number of parallel streams
    const int num_centroids = 64;
    const int vector_dim = 2;
    
    std::cout << "Testing CCU FP32 with " << L << " input vectors per stream, " 
              << num_streams << " streams, and " << num_centroids << " centroids" << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    // Generate random centroids (64 centroids, each with 2 elements)
    std::vector<std::vector<std::vector<float>>> centroids_ref(num_streams, std::vector<std::vector<float>>(num_centroids, std::vector<float>(vector_dim)));
    std::vector<tapa::vec_t<float, 16>> centroids_hw(64);  // 8 vectors of 16 floats each

    for (int s = 0; s < num_streams; ++s) {
        for (int i = 0; i < num_centroids; ++i) {
            for (int k = 0; k < vector_dim; ++k) {
                float val = dis(gen);
                centroids_ref[s][i][k] = val;
                centroids_hw[i][s*vector_dim+k] = val;
            }
        }
    }
    
    // Generate random input vectors
    // Input format: L vectors of 16 elements each (8 streams * 2 elements per stream)
    std::vector<tapa::vec_t<float, 16>> input_hw(L);
    std::vector<std::vector<std::vector<float>>> input_ref(num_streams, 
        std::vector<std::vector<float>>(L, std::vector<float>(vector_dim)));
    
    for (int l = 0; l < L; ++l) {
        for (int s = 0; s < num_streams; ++s) {
            for (int j = 0; j < vector_dim; ++j) {
                float val = dis(gen);
                // Pack 8 streams of 2-element vectors into 16-element hardware vectors
                int elem_idx = s * vector_dim + j;
                input_hw[l][elem_idx] = val;
                input_ref[s][l][j] = val;
            }
        }
    }
    
    // Allocate output arrays - 8 separate arrays for 8 streams
    std::vector<std::vector<int>> idx_out_hw(num_streams, std::vector<int>(L));
    std::vector<std::vector<int>> idx_out_ref(num_streams, std::vector<int>(L));
    
    // Compute reference results for each stream
    std::cout << "Computing reference results..." << std::endl;
    for (int s = 0; s < num_streams; ++s) {
        for (int l = 0; l < L; ++l) {
            idx_out_ref[s][l] = find_closest_centroid(input_ref[s][l], centroids_ref[s]);
        }
    }
    
    // Run hardware implementation
    std::cout << "Running hardware implementation..." << std::endl;
    
    tapa::invoke(ccu_fp32_top, FLAGS_bitstream,
                L,
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(input_hw),
                tapa::read_only_mmap<tapa::vec_t<float, 16>>(centroids_hw),
                tapa::write_only_mmaps<int, 8>(idx_out_hw));
    
    // Verify results for each stream
    std::cout << "Verifying results..." << std::endl;
    int total_errors = 0;
    
    for (int s = 0; s < num_streams; ++s) {
        int stream_errors = 0;
        for (int l = 0; l < L; ++l) {
            if (idx_out_hw[s][l] != idx_out_ref[s][l]) {
                stream_errors++;
                total_errors++;
                if (total_errors <= 10) {  // Print first 10 errors
                    std::cout << "Error at stream " << s << ", index " << l 
                             << ": HW=" << idx_out_hw[s][l] 
                             << ", REF=" << idx_out_ref[s][l] << std::endl;
                    
                    // Print input vector and distances for debugging
                    std::cout << "  Input vector: [" << input_ref[s][l][0] << ", " << input_ref[s][l][1] << "]" << std::endl;
                    std::cout << "  HW chosen centroid " << idx_out_hw[s][l] << ": ["
                             << centroids_ref[s][idx_out_hw[s][l]][0] << ", " 
                             << centroids_ref[s][idx_out_hw[s][l]][1] << "] dist="
                             << chebyshev_distance(input_ref[s][l], centroids_ref[s][idx_out_hw[s][l]]) << std::endl;
                    std::cout << "  REF chosen centroid " << idx_out_ref[s][l] << ": ["
                             << centroids_ref[s][idx_out_ref[s][l]][0] << ", " 
                             << centroids_ref[s][idx_out_ref[s][l]][1] << "] dist="
                             << chebyshev_distance(input_ref[s][l], centroids_ref[s][idx_out_ref[s][l]]) << std::endl;
                }
            }
        }
        if (stream_errors > 0) {
            std::cout << "Stream " << s << ": " << stream_errors << " errors out of " << L << " vectors" << std::endl;
        }
    }
    
    if (total_errors == 0) {
        std::cout << "SUCCESS: All " << (L * num_streams) << " results match!" << std::endl;
    } else {
        std::cout << "FAILURE: " << total_errors << " out of " << (L * num_streams) << " results don't match!" << std::endl;
    }
    
    // Print some sample results from each stream
    std::cout << "\nSample results (first 5 from each stream):" << std::endl;
    for (int s = 0; s < num_streams; ++s) {
        std::cout << "Stream " << s << ":" << std::endl;
        for (int l = 0; l < std::min(5, L); ++l) {
            std::cout << "  Input " << l << ": [" << input_ref[s][l][0] << ", " << input_ref[s][l][1] 
                     << "] -> Centroid " << idx_out_hw[s][l] << " ["
                     << centroids_ref[s][idx_out_hw[s][l]][0] << ", " 
                     << centroids_ref[s][idx_out_hw[s][l]][1] << "]" << std::endl;
        }
    }
    
    return total_errors == 0 ? 0 : 1;
}
