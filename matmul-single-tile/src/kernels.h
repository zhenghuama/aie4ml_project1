#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

void matmul_4x4(
    adf::input_buffer<int16>& a, adf::input_buffer<int16>& b, adf::output_buffer<int16>& c
);

#endif
