#include <aie_api/aie.hpp>
#include <adf.h>
#include "kernels.h"
#include "include.h"


void matmul_int32_accumulate(
    input_window<int32_t>* in_A,
    input_window<int32_t>* in_B,
    output_window<int32_t>* out_C) {
    
    // Vectorized approach for int32 matrix multiplication
    // Using AIE vector registers (v16int32 can hold 16 int32 values)
    v16int32 A_buf, B_buf;
    v16int32 acc = null_v16int32();  // initialize accumulator
    
    // Process the entire tile
    for (int i = 0; i < TILE_M; i++) {
        for (int j = 0; j < TILE_N; j += 16) {  // Process 16 columns at once with vector
            acc = null_v16int32();  // Reset accumulator for each output element
            
            // Compute dot product for K dimension
            for (int k = 0; k < TILE_K; k++) {
                // Load row of A (broadcast scalar to vector)
                int32_t a_val = window_readincr(in_A);
                A_buf = aie::broadcast<int32_t, 16>(a_val);
                
                // Load row of B into vector register
                B_buf = window_readincr_v16(in_B);
                
                // Vector MAC operation: acc += A_buf * B_buf
                acc = aie::mac(acc, A_buf, B_buf);
            }
            
            // Store result vector to output window
	    // Note that the transpose of the vector just calculated will be written to the output
	    // This is what we want for column major format
            window_writeincr_v16(out_C, acc);
        }
    }
}
