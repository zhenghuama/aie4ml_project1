#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>


void matmul_int32_scalar_transfer(
	input_window<int32_t>* in_A,
	input_window<int32_t>* in_B,
	output_window<int32_t>* out_C);

#endif
