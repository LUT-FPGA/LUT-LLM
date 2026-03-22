#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <prefill_lat> <decode_lat>\n";
    return 1;
  }

  try {
    const double prefill_lat = std::stod(argv[1]);
    const double decode_lat = std::stod(argv[2]);
    const double reference_latency = 1.410;

    const double end_to_end_latency =
        (prefill_lat * 28.0 + (decode_lat * 28.0) * 256.0) / 250.0 / 1e6;
    const bool is_smaller_than_reference = end_to_end_latency < reference_latency;

    std::cout << std::fixed << std::setprecision(12)
              << "End-to-end latency: " << end_to_end_latency << " s\n";
    std::cout << "Reference latency: " << reference_latency << " s\n";
    std::cout << "Is end-to-end latency smaller than reference? "
              << (is_smaller_than_reference ? "Yes" : "No") << "\n";
  } catch (const std::exception&) {
    std::cerr << "Error: prefill_lat and decode_lat must be numeric values.\n";
    return 1;
  }

  return 0;
}
