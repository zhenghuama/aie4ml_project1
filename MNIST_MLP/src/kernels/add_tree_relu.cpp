#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include "include.h"
#include "kernels.h"

using namespace adf;

// Assume N*M is multiple of VEC for vector alignment
void add_relu_4::run(
    input_buffer<int16>& in0,  
    input_buffer<int16>& in1,   
    input_buffer<int16>& in2, 
    input_buffer<int16>& in3,
    input_buffer<int16>& bias,
    output_buffer<int16>& out
) {
    // Vector iterators for VEC-element parallel processing
    auto in0_iter = aie::begin_vector<VEC>(in0);
    auto in1_iter = aie::begin_vector<VEC>(in1);
    auto in2_iter = aie::begin_vector<VEC>(in2);
    auto in3_iter = aie::begin_vector<VEC>(in3);
    auto out_iter = aie::begin_vector<VEC>(out);

    // Process all elements in vector chunks
    for(int row = 0; row < N; ++row) {
	auto bias_iter = aie::begin_vector<VEC>(bias);
	for (int col = 0; col < M/VEC; ++col) {
            aie::vector<int16, VEC> v0 = *in0_iter++;
            aie::vector<int16, VEC> v1 = *in1_iter++;
            aie::vector<int16, VEC> v2 = *in2_iter++;
            aie::vector<int16, VEC> v3 = *in3_iter++;
	    aie::vector<int16, VEC> v_bias = *bias_iter++;

            // Vector addition with saturation
            aie::vector<int16, VEC> sum = aie::add(aie::add(v0, v1), 
                                             aie::add(v2, v3));
	    sum = aie::add(v_bias, sum);
            aie::vector<int16, VEC> relu = aie::max(sum, aie::broadcast<int16, VEC>(0)); 
    	    *out_iter++ = relu;
	}
    }
}

void add_relu_6::run(
    input_buffer<int16>& in0,     
    input_buffer<int16>& in1, 
    input_buffer<int16>& in2, 
    input_buffer<int16>& in3,  
    input_buffer<int16>& in4,
    input_buffer<int16>& in5,
    input_buffer<int16>& bias,
    output_buffer<int16>& out
) {
    // Vector iterators for VEC-element parallel processing
    auto in0_iter = aie::begin_vector<VEC>(in0);
    auto in1_iter = aie::begin_vector<VEC>(in1);
    auto in2_iter = aie::begin_vector<VEC>(in2);
    auto in3_iter = aie::begin_vector<VEC>(in3);
    auto in4_iter = aie::begin_vector<VEC>(in4);
    auto in5_iter = aie::begin_vector<VEC>(in5);
    auto out_iter = aie::begin_vector<VEC>(out);


    // Process all elements in vector chunks
    for (int row = 0; row < N; ++row) {
        auto bias_iter = aie::begin_vector<VEC>(bias);
        for(int i = 0; i < M/VEC; ++i) {
            aie::vector<int16, VEC> v0 = *in0_iter++;
            aie::vector<int16, VEC> v1 = *in1_iter++;
            aie::vector<int16, VEC> v2 = *in2_iter++;
            aie::vector<int16, VEC> v3 = *in3_iter++;
            aie::vector<int16, VEC> v4 = *in4_iter++;
            aie::vector<int16, VEC> v5 = *in5_iter++;
	    aie::vector<int16, VEC> v_bias = *bias_iter++;

            // Vector addition with saturation
            aie::vector<int16, VEC> sum = aie::add(
	    aie::add(aie::add(v0, v1), aie::add(v2, v3)),
       	    aie::add(v4, v5));
	    
	    // Add bias with saturation
	    sum = aie::add(v_bias, sum);
	    
	    // Apply RELU
            aie::vector<int16, VEC> relu = aie::max(sum, aie::broadcast<int16, VEC>(0)); 
	
    	    *out_iter++ = relu;
	}
    }
}
