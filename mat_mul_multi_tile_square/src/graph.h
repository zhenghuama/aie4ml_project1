#include <adf.h>
#include "kernels.h"
#include "include.h"

using namespace adf;

class MatMulGraph : public graph {
public:
    input_plio a_in;
    input_plio b_in;
    output_plio c_out[4][4];
    
    input_port a_block_param[4][4];
    input_port b_block_param[4][4];

    kernel mmul[4][4];

    MatMulGraph() {
	a_in = input_plio::create("A_Matrix", plio_128_bits, "data/A_matrix.txt");
        b_in = input_plio::create("B_Matrix", plio_128_bits, "data/B_matrix.txt");

        // Create 4x4 kernel grid
        for(int row=0; row<4; row++) {
            for(int col=0; col<4; col++) {
                mmul[row][col] = kernel::create(matmul_4x16x4);

                // Connect A row block (4x16 = 64 elements)
                connect(a_in.out[0], mmul[row][col].in[0]);
                dimensions(mmul[row][col].in[0]) = {256}; 

                // Connect B column block (16x4 = 64 elements)
                connect(b_in.out[0], mmul[row][col].in[1]);
                dimensions(mmul[row][col].in[1]) = {256};
							 
                // Connect parameter ports
                connect(a_block_param[row][col], mmul[row][col].in[2]);
                connect(b_block_param[row][col], mmul[row][col].in[3]);

		// Create PLIO with 32-bit interface for 4x4 int16 blocks
		c_out[row][col] = output_plio::create(
		  plio_128_bits,
		  "data/C_output_"+std::to_string(row)+"_"+std::to_string(col)+".txt"
		);

                connect(mmul[row][col].out[0], c_out[row][col].in[0]);

                // Set buffer dimensions (16 int16 elements = 4x4 matrix)
                dimensions(mmul[row][col].out[0]) = {16};

                // Map to physical tiles
                location<kernel>(mmul[row][col]) = tile(row, col);
		source(mmul[row][col]) = "src/kernels/kernels.cpp";
		runtime<ratio>(mmul[row][col]) = 1.0;
            }
        }
    }
};
