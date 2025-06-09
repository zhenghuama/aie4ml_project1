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
   // Enable hardware saturation for 48b→16b conversion
   aie::set_saturation(aie::saturation_mode::saturate);

   // Tiling strategy: K dimension split across T tiles
   const int K_Tile = K/T;
   
   // Offset A-pointer for this tile's portion of K
   auto a_iter = aie::begin_vector<VEC>(a_buf) + K_Tile/VEC*a_block;
   auto c_iter = aie::begin(c_buf);

   // Process N output rows (full output matrix)
   for (int n = 0; n < N; ++n) {
       auto b_iter = aie::begin_vector<VEC>(b_buf);
       // Process M output columns
       for (int m = 0; m < M; ++m) {
           aie::accum<acc48, VEC> acc = aie::zeros<acc48, VEC>();
	   aie::vector<int16, VEC> a_vec;
	   aie::vector<int16, VEC> b_vec;
	   
	   // Process K_Tile elements per output point (vectorized)
	   for (int k = 0; k < K_Tile/VEC; ++k) {
	       a_vec = *a_iter++;  // Vector load (A-matrix tile)
	       b_vec = *b_iter++;  // Vector load (B-matrix column)
	        
	       // First multiply → subsequent MAC operations
	       acc = (k == 0) ? aie::mul(a_vec, b_vec) 
	                      : aie::mac(acc, a_vec, b_vec);
	   }
	   a_iter -= K_Tile/VEC; // Rewind A-ptr for next column

	   // Convert 48b accumulator to 16b with saturation
	   aie::vector<int16, VEC> res_vec = acc.template to_vector<int16>();
           // Sum vector elements to scalar (output matrix point)
           int16_t res = aie::reduce_add(res_vec);
           *c_iter++ = res;
       }
       a_iter += K/VEC; // Advance to next A-matrix row block
   }
}

