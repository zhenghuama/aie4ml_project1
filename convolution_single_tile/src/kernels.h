#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

// Convolution function declaration
void conv2d_3x3(
    adf::input_buffer<int16>& input, 
    adf::input_buffer<int16>& weights, 
    adf::output_buffer<int16>& output
);

#endif
