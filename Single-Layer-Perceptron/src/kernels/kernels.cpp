#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include "../kernels.h"
#include "include.h"

using namespace adf;

/**
 * Single Layer Perceptron forward pass implementation for AIE
 * 
 * This implements a fully connected layer with weights, biases and 
 * a simple step activation function
 * 
 * @param input_features - Input buffer containing feature vectors
 * @param weights - Input buffer containing weight matrix (INPUT_SIZE x OUTPUT_SIZE)
 * @param biases - Input buffer containing bias values for each output
 * @param output - Output buffer for the classification results
 */
void slp_forward(
    input_buffer<float>& __restrict input_features,
    input_buffer<float>& __restrict weights,
    input_buffer<float>& __restrict biases,
    output_buffer<float>& __restrict output
) {
    // Get pointers to the buffers
    auto input_ptr = input_features.data();
    auto weight_ptr = weights.data();
    auto bias_ptr = biases.data();
    auto output_ptr = output.data();

    // AIE vector operations for optimized computation
    aie::vector<float, 16> input_vec;
    aie::vector<float, 16> weight_vec;
    
    // Load input features into vector register
    input_vec = aie::load_v<16>(input_ptr);
    
    // Compute output for each class
    for (int out_idx = 0; out_idx < OUTPUT_SIZE; out_idx++) {
        float result = bias_ptr[out_idx]; // Start with bias
        
        // Get the weights for this output
        for (int in_idx = 0; in_idx < INPUT_SIZE; in_idx += 16) {
            // Load weights for current input chunk
            weight_vec = aie::load_v<16>(&weight_ptr[out_idx * INPUT_SIZE + in_idx]);
            
            // Perform dot product between input and weights
            aie::accum<float, 16> acc = aie::mul(input_vec, weight_vec);
            
            // Accumulate the result
            result += aie::reduce_add(acc);
        }
        
        // Apply step activation function
        output_ptr[out_idx] = (result > ACTIVATION_THRESHOLD) ? 1.0f : 0.0f;
    }
}
