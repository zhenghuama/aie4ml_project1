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

void add_tree(
    adf::input_buffer<int16>& in0,  // North neighbor
    adf::input_buffer<int16>& in1,  // East neighbor  
    adf::input_buffer<int16>& in2,  // South neighbor
    adf::input_buffer<int16>& in3,  // West neighbor
    adf::output_buffer<int16>& out
);
#endif
