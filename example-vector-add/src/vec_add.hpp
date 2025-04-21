#ifndef VEC_ADD_HPP
#define VEC_ADD_HPP

#include <adf.h>

void vec_add(
    adf::input_buffer<int32>& data1,  // Removed __restrict
    adf::input_buffer<int32>& data2,
    adf::output_buffer<int32>& out
);

#endif

