#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

void mmul_skinny(adf::input_buffer<int16>& a_buf, 
adf::input_buffer<int16>& b_buf, 
adf::output_buffer<int16>& c_buf, 
int a_block
);

void mmul_skinny_padding_aware(
    adf::input_buffer<int16>& a_buf,
    adf::input_buffer<int16>& b_buf,
    adf::output_buffer<int16>& c_buf
);

void add_tree_4(
    adf::input_buffer<int16>& in0, 
    adf::input_buffer<int16>& in1,  
    adf::input_buffer<int16>& in2,
    adf::input_buffer<int16>& in3,
    adf::output_buffer<int16>& out
);

void add_tree_3(
    adf::input_buffer<int16>& in0,
    adf::input_buffer<int16>& in1,
    adf::input_buffer<int16>& in2,
    adf::output_buffer<int16>& out
);

void add_tree_6(
    adf::input_buffer<int16>& in1,
    adf::input_buffer<int16>& in2,
    adf::input_buffer<int16>& in3,
    adf::input_buffer<int16>& in4,
    adf::input_buffer<int16>& in5,
    adf::input_buffer<int16>& in6,
    adf::output_buffer<int16>& out
);

#endif
