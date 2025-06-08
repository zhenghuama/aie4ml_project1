#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include "include.h"
#include "kernels.h"

using namespace adf;

void mmul_skinny::run(
    input_buffer<int16>& a_buf,
    input_buffer<int16>& b_buf, 
    output_buffer<int16>& c_buf)
{
   aie::set_rounding(aie::rounding_mode::symmetric_zero);	   
   aie::set_saturation(aie::saturation_mode::saturate);

   const int K_Tile = K/T;

   auto a_iter = aie::begin_vector<VEC>(a_buf) + K_Tile/VEC*a_block;
   auto c_iter = aie::begin(c_buf);

   for (int n = 0; n < N; ++n) {
       auto b_iter = aie::begin_vector<VEC>(b_buf);
       for (int m = 0; m < M; ++m) {
           aie::accum<acc48, VEC> acc = aie::zeros<acc48, VEC>();
	   aie::vector<int16, VEC> a_vec;
	   aie::vector<int16, VEC> b_vec;
	   for (int k = 0; k < K_Tile/VEC; ++k) {
	       a_vec = *a_iter++;
	       b_vec = *b_iter++;
	        
	       if (k == 0) {
	           acc = aie::mul(a_vec, b_vec);
	       } else {
		   acc = aie::mac(acc, a_vec, b_vec);
	       }
	   }
	   a_iter -= K_Tile/VEC;

           aie::vector<int16, VEC> res_vec16 = acc.to_vector<int16>();

           int16 res = aie::reduce_add(res_vec16);
           *c_iter++ = res;
       }
       a_iter += K/VEC;
   }
}

