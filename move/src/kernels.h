#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

void move(
    adf::input_buffer<int16>& mov_in, adf::output_buffer<int16>& mov_out
);

#endif
