#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include "include.h"
#include "kernels.h"

using namespace adf;

// Assume N*M is multiple of VEC for vector alignment
void add_tree_4::run(
    input_buffer<int16>& in0,  // North neighbor
    input_buffer<int16>& in1,  // East neighbor  
    input_buffer<int16>& in2,  // South neighbor
    input_buffer<int16>& in3,  // West neighbor
    output_buffer<int16>& out
) {
    // Vector iterators for VEC-element parallel processing
    auto in0_iter = aie::begin_vector<VEC>(in0);
    auto in1_iter = aie::begin_vector<VEC>(in1);
    auto in2_iter = aie::begin_vector<VEC>(in2);
    auto in3_iter = aie::begin_vector<VEC>(in3);
    auto out_iter = aie::begin_vector<VEC>(out);

    const int total_vectors = (N * M) / VEC;

    // Process all elements in vector chunks
    for(int i = 0; i < total_vectors; ++i) {
        aie::vector<int16, VEC> v0 = *in0_iter++;
        aie::vector<int16, VEC> v1 = *in1_iter++;
        aie::vector<int16, VEC> v2 = *in2_iter++;
        aie::vector<int16, VEC> v3 = *in3_iter++;

        // Vector addition with saturation
        aie::vector<int16, VEC> sum = aie::add(aie::add(v0, v1), 
                                             aie::add(v2, v3));
        
    	*out_iter++ = sum;
    }
}

