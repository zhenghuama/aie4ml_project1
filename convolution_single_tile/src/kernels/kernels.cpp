#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include "../kernels.h"
#include "include.h"

using namespace adf;

/**
 * 2D Convolution implementation for AIE
 * Performs a 3x3 convolution on input data
 * 
 * @param input - Input buffer containing the input image data
 * @param weights - Input buffer containing the 3x3 convolution kernel weights
 * @param output - Output buffer for the convolution result
 */
void conv2d_3x3(
    input_buffer<int16>& __restrict input,
    input_buffer<int16>& __restrict weights,
    output_buffer<int16>& __restrict output
) {
    // Get vector iterators for the input, weights, and output
    auto inPtr = input.data();
    auto wPtr = weights.data();
    auto outPtr = output.data();
    
    // Load the 3x3 convolution kernel weights into local memory
    int16 w[KERNEL_HEIGHT][KERNEL_WIDTH];
    for (int i = 0; i < KERNEL_HEIGHT; i++) {
        for (int j = 0; j < KERNEL_WIDTH; j++) {
            w[i][j] = wPtr[i * KERNEL_WIDTH + j];
        }
    }
    
    // Perform the convolution
    for (int y = 0; y < OUTPUT_HEIGHT; y++) {
        for (int x = 0; x < OUTPUT_WIDTH; x++) {
            int16 sum = 0;
            
            // 3x3 convolution kernel
            for (int ky = 0; ky < KERNEL_HEIGHT; ky++) {
                for (int kx = 0; kx < KERNEL_WIDTH; kx++) {
                    int16 pixel = inPtr[(y + ky) * INPUT_WIDTH + (x + kx)];
                    sum += pixel * w[ky][kx];
                }
            }
            
            // Write result to output
            outPtr[y * OUTPUT_WIDTH + x] = sum;
        }
    }
}
