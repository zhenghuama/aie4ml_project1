#include <aie_api/aie.hpp>
#include <adf.h>
#include "kernels.h"
#include "include.h"


using MMUL = aie::mmul<4, 4, 4, int16, int16>; // 4x4x4 int16 config[1][3][5]

void matmul_4x4(
    adf::input_buffer<int16>& a, adf::input_buffer<int16>& b, adf::output_buffer<int16>& c
) {
    // Use vector iterators for 4x4 matrices (16 elements)
    auto a_iter = aie::begin_vector<16>(a);
    auto b_iter = aie::begin_vector<16>(b);
    auto c_iter = aie::begin_vector<16>(c);

    MMUL m;
    m.mul(*a_iter, *b_iter);  // Load via iterator dereference
    
    *c_iter = m.to_vector<int16>(); // Store via iterator assignment
}
