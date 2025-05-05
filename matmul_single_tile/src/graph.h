#include <adf.h>
#include "kernels.h"
#include "include.h"

using namespace adf;


/**
 * Single-tile matrix multiplication graph for Xilinx AI Engine using built-in mat_mul
 * Optimized for int32 data type
 * A: Input matrix of dimensions TILE_M x TILE_K
 * B: Input matrix of dimensions TILE_K x TILE_N
 * C: Output matrix of dimensions TILE_M x TILE_N
 */


class MatrixMultInt32 : public adf::graph {
	private:
		kernel k;
	public:
		input_plio in_A;
		input_plio in_B;
		output_plio out_C;

		MatrixMultInt32() {

			in_A = input_plio::create(plio_32_bits, "data/A_matrix.txt");
			in_B = input_plio::create(plio_32_bits, "data/B_matrix.txt");
			out_C = output_plio::create(plio_32_bits, "data/C_output.txt");
			kernel k = kernel::create(matmul_int32_scalar_transfer);


			runtime<ratio>(k) = 0.9;

	        	// Connect I/O ports to kernel
			connect<window<TILE_M * TILE_K * sizeof(int32_t)>>(in_A.out[0], k.in[0]);
			connect<window<TILE_K * TILE_N * sizeof(int32_t)>>(in_B.out[0], k.in[1]);
			connect<window<TILE_M * TILE_N * sizeof(int32_t)>>(k.out[0], out_C.in[0]);

			source(k) = "src/kernels/kernels.cpp";
		}
	
};
