#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

const int M = 4; //rows of A (batch dimension)
const int K = 32; //length wise split (inner dimension)
const int N = 128; //columns of B (height of layer) 


void matmul_skinny(
  input_buffer<int16>& a, 
  input_buffer<int16>& b,
  output_buffer<int16>& c,
  int a_block, int b_block)
{
  auto a = aie::begin_vector<32>(in_a);
  auto b = aie::begin_vector<32>(in_b);
  auto c = aie::begin_vector<32>(out_c);

  aie::accum<acc48, 32> acc;

  for (int m = 0; m < M; ++m) {
    auto b = aie::begin_vector<32>(in_b);
    for (int n = 0; n < N; ++n) {
       auto a = aie::begin_vector<32>(in_a)+m*k/32;  //Establishes row of A that will be operated on
       for (int k = 0; k < K/32; ++k) {
	  a_vec = *a++; //increment length-wise splits in strides of 32 (must pad with zeros somehow)
	  b_vec = *b++; //increment section of the b column that must be operated on (must pad with zeros somehow)
	  if (k == 0) acc = aie::mul16(a_vec, b_vec);
	  else acc = aie::mac16(acc, a_vec, b_vec);
       }
    *c++ = acc.template to_vector<int16>();
    //instead of a vector I want to sum the accumulator element-wise to get a scalar then write output. Is this possible?
    }
  }



       
    
