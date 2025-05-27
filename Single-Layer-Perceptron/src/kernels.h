#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

// Single Layer Perceptron function declaration
void slp_forward(
    adf::input_buffer<float>& input_features,
    adf::input_buffer<float>& weights,
    adf::input_buffer<float>& biases,
    adf::output_buffer<float>& output
);

#endif
