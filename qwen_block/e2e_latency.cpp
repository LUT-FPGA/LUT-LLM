#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

// Reference latencies (seconds) keyed by {input_len, output_len}.
// Fill in the values below.
static const std::map<std::pair<int,int>, double> kReferenceLatency = {
    {{32,  16},  0.1153},
    {{32,  64},  0.3650},
    {{32,  256}, 1.3243},
    {{128, 16},  0.2519},
    {{128, 64},  0.4916},
    {{128, 256}, 1.4506},
};

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <prefill_lat> <decode_lat> <input_len> <output_len>\n"
              << "  input_len  : 32 or 128\n"
              << "  output_len : 16, 64, or 256\n";
    return 1;
  }

  try {
    const double prefill_lat = std::stod(argv[1]);
    const double decode_lat  = std::stod(argv[2]);
    const int    input_len   = std::stoi(argv[3]);
    const int    output_len  = std::stoi(argv[4]);

    if (input_len != 32 && input_len != 128) {
      std::cerr << "Error: input_len must be 32 or 128.\n";
      return 1;
    }
    if (output_len != 16 && output_len != 64 && output_len != 256) {
      std::cerr << "Error: output_len must be 16, 64, or 256.\n";
      return 1;
    }

    const double reference_latency =
        kReferenceLatency.at({input_len, output_len});

    const double end_to_end_latency =
        (prefill_lat * 28.0 + (decode_lat * 28.0) * output_len) / 250.0 / 1e6;
    const bool is_smaller_than_reference =
        end_to_end_latency < reference_latency;

    std::cout << std::fixed << std::setprecision(12)
              << "Input length:  " << input_len  << "\n"
              << "Output length: " << output_len << "\n"
              << "End-to-end latency: " << end_to_end_latency << " s\n"
              << "Reference latency:  " << reference_latency  << " s\n"
              << "Is end-to-end latency smaller than reference? "
              << (is_smaller_than_reference ? "Yes" : "No") << "\n";
  } catch (const std::invalid_argument&) {
    std::cerr << "Error: numeric arguments must be valid numbers.\n";
    return 1;
  } catch (const std::out_of_range&) {
    std::cerr << "Error: argument out of range.\n";
    return 1;
  }

  return 0;
}
