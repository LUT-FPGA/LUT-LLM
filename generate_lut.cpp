#include <iostream>
#include <random>
#include <vector>
#include <bitset>
#include <fstream>
#include <limits>

// Function to convert float to FP16 bitstream
uint16_t floatToFP16(float value) {
    // Extract IEEE 754 single-precision (FP32) components
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint32_t sign = (bits >> 31) & 0x1;       // Extract sign bit
    int32_t exponent = ((bits >> 23) & 0xFF) - 127; // Extract exponent and subtract bias
    uint32_t mantissa = bits & 0x7FFFFF;      // Extract mantissa (23 bits)

    // Handle special cases
    if (exponent == 128) { // Inf or NaN
        return (sign << 15) | (0x1F << 10) | (mantissa ? 0x200 : 0);
    }
    if (exponent < -14) { // Subnormal or zero
        if (exponent < -24) {
            return (sign << 15); // Zero
        }
        // Subnormal number
        mantissa |= 0x800000; // Add implicit leading 1
        int shift = -exponent - 14;
        return (sign << 15) | (mantissa >> (13 + shift));
    }
    if (exponent > 15) { // Overflow to Inf
        return (sign << 15) | (0x1F << 10);
    }

    // Normalized number
    exponent += 15; // Add FP16 bias
    mantissa >>= 13; // Truncate to 10 bits
    return (sign << 15) | (exponent << 10) | mantissa;
}

// Function to generate random 16-bit floating point numbers
uint16_t generateRandomFP16() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dis(0, 1);
    float value = dis(gen);
    return floatToFP16(value);
}

int main() {
    const int inputBits = 9;
    const int outputBits = 8;
    const int numEntries = 1 << inputBits; // 2^10 entries

    std::vector<std::pair<std::bitset<inputBits>, uint16_t>> truthTable;

    for (int i = 0; i < numEntries; ++i) {
        std::bitset<inputBits> input(i);
        uint16_t output = generateRandomFP16();
        truthTable.push_back({input, output});
    }

    std::ofstream outFile("lut_mm.sv");
    if (!outFile) {
        std::cerr << "Error opening file for writing" << std::endl;
        return 1;
    }

    // declaration
    outFile << "module lut_mm (" << std::endl;
    outFile << "\tinput logic [" << inputBits-1 << ":0] data_in," << std::endl;
    outFile << "\toutput logic [" << outputBits-1 << ":0] data_out" << std::endl;
    outFile << ");" << std::endl << std::endl;
    outFile << "\talways_comb begin" << std::endl;
    outFile << "\t\tcase (data_in)" << std::endl;

    for (const auto& entry : truthTable) {
        outFile << "\t\t\t" << inputBits << "'b" << entry.first << ": data_out = " << outputBits << "'b" << std::bitset<outputBits>((entry.second & 0xFF00) >> 8) << ";" << std::endl;
    }

    outFile << "\t\tendcase" << std::endl;
    outFile << "\tend" << std::endl << std::endl;
    outFile << "endmodule" << std::endl;

    outFile.close();
    return 0;
}
