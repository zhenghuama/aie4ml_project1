#include <aie_api/aie.hpp>
#include <adf.h>
using namespace adf;
using MMUL = aie::mmul<4, 4, 4, int16, int16>;

void matmul_4x16x4(
    input_buffer<int16>& __restrict a,
    input_buffer<int16>& __restrict b,
    output_buffer<int16>& __restrict c,
    int a_block, int b_block)
{
    auto a_iter = aie::begin_vector<MMUL::size_A>(a) + a_block*64;
    auto b_iter = aie::begin_vector<MMUL::size_B>(b) + b_block*64;
    auto c_iter = aie::begin_vector<MMUL::size_C>(c);

    MMUL m;

    // First iteration: initialize accumulator
    m.mul(*a_iter++, *b_iter++);  // m.acc = A0 * B0

    // Subsequent iterations: multiply-accumulate
    for(int i = 1; i < 4; i++) {
        m.mac(*a_iter++, *b_iter++);  // m.acc += A_i * B_i
    }

    // Single store at end
    *c_iter = m.to_vector<int16>();
}
