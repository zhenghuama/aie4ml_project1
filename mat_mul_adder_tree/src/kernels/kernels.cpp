#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include "include.h"
#include "kernels.h"

int K_Tile = K/4;

using namespace adf;

void mmul_skinny(
    input_buffer<int16>& a_buf,
    input_buffer<int16>& b_buf, 
    output_buffer<int16>& c_buf,
    int a_block)
{
   auto a_iter = aie::begin_vector<32>(a_buf) + K_Tile/32*a_block;
   auto c_iter = aie::begin(c_buf);

   for (int n = 0; n < N; ++n) {
       auto b_iter = aie::begin_vector<32>(b_buf);
       for (int m = 0; m < M; ++m) {
           aie::accum<acc48, 32> acc = aie::zeros<acc48, 32>();
	   aie::vector<int16, 32> a_vec;
	   aie::vector<int16, 32> b_vec;
	   for (int k = 0; k < K_Tile/32; ++k) {
	       a_vec = *a_iter++;
	       b_vec = *b_iter++;
	        
	       if (k == 0) {
	           acc = aie::mul(a_vec, b_vec);
	       } else {
		   acc = aie::mac(acc, a_vec, b_vec);
	       }
	   }
	   a_iter -= K_Tile/32;
           aie::vector<int16, 32> res_vec = acc.template to_vector<int16>();
           int16_t res = aie::reduce_add(res_vec);
           *c_iter++ = res;  
       }
       a_iter += K/32;
   }
}
       
	   
// This kernels is supposed to work when the internal dimension (K) is not a multiple of 32. This kernel does not yet work. 
// I think it doesn't work because random access into a buffer, whether using the buffer directly or using an iterator, is not a thing. Stoopid.
void mmul_skinny_padding_aware(
    input_buffer<int16>& a_buf,
    input_buffer<int16>& b_buf,
    output_buffer<int16>& c_buf)
{
    auto a_iter = aie::begin(a_buf);
    auto b_iter = aie::begin(b_buf);
    auto c_iter = aie::begin(c_buf);
    
    constexpr int VEC = 32;
    const int K_full = K_Tile / VEC;
    const int K_rem = K_Tile % VEC;

    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            aie::accum<acc48, VEC> acc = aie::zeros<acc48, VEC>();

            // Full vector blocks
            for (int k = 0; k < K_full; ++k) {
                auto a_vec = aie::load_v<VEC>(&a_iter[n*K + k*VEC]);
                auto b_vec = aie::load_v<VEC>(&b_iter[m*K_Tile + k*VEC]);
                if (k == 0) {
		    aie::mul(a_vec, b_vec);
                } else {
                    aie::mac(acc, a_vec, b_vec);
	        }
            }

            // Remainder
            if (K_rem > 0) {
                alignas(aie::vector_decl_align) int16 a_tail[VEC] = {0};
                alignas(aie::vector_decl_align) int16 b_tail[VEC] = {0};
                for (int i = 0; i < K_rem; ++i) {
                    a_tail[i] = a_iter[n*K + K_full*VEC + i];
                    b_tail[i] = b_iter[m*K_Tile + K_full*VEC + i];
                }
                auto a_vec = aie::load_v<VEC>(a_tail);
                auto b_vec = aie::load_v<VEC>(b_tail);
                acc = aie::mac(acc, a_vec, b_vec);
            }

            aie::vector<int16, VEC> res_vec = acc.template to_vector<int16>();
            int16_t res = aie::reduce_add(res_vec);
            c_iter[n*K + m] = res; // Safe, bounds-checked write
        }
    }
}

