#ifndef VEC_ADD_HPP
#define VEC_ADD_HPP

#include <adf.h>

void vec_add(
    adf::input_buffer<int32>& __restrict data1,
    adf::input_buffer<int32>& __restrict data2,
    adf::output_buffer<int32>& __restrict out
);

#endif

