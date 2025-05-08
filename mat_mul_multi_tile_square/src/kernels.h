#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

void matmul_4x16x4(
    adf::input_buffer<int16>& __restrict a,
    adf::input_buffer<int16>& __restrict b,
    adf::output_buffer<int16>& __restrict c);

#endif
