#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"
#include <aie_api/aie_adf.hpp>

using namespace adf;

/**
 * Single-Layer Perceptron graph for Xilinx AI Engine
 * Implements a simple classifier with one fully connected layer
 * and a step activation function
 */
class PerceptronGraph : public adf::graph {
private:
    kernel k;
public:
    input_plio in_features;
    input_plio in_weights;
    input_plio in_biases;
    output_plio out_result;

    PerceptronGraph() {
        // Create input/output interfaces
        in_features = input_plio::create(plio_64_bits, "data/input_features.txt");
        in_weights = input_plio::create(plio_64_bits, "data/weights.txt");
        in_biases = input_plio::create(plio_64_bits, "data/biases.txt");
        out_result = output_plio::create(plio_64_bits, "data/output.txt");
        
        // Create the perceptron kernel
        k = kernel::create(slp_forward);

        // Set runtime ratio
        runtime<ratio>(k) = 0.9;

        // Set dimensions for data ports
        dimensions(k.in[0]) = {INPUT_SIZE};           // Input features
        dimensions(k.in[1]) = {INPUT_SIZE * OUTPUT_SIZE}; // Weights
        dimensions(k.in[2]) = {OUTPUT_SIZE};          // Biases
        dimensions(k.out[0]) = {OUTPUT_SIZE};         // Output probabilities

        // Connect I/O ports to kernel
        connect(in_features.out[0], k.in[0]);
        connect(in_weights.out[0], k.in[1]);
        connect(in_biases.out[0], k.in[2]);
        connect(k.out[0], out_result.in[0]);

        // Set source file for kernel
        source(k) = "kernels/kernels.cpp";
    }
};
