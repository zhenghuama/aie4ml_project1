#include <adf.h>
#include "kernels.h"
#include <aie_api/aie_adf.hpp>
#include "include.h"

using namespace adf;

class SingleTileTest : public adf::graph {
public:
        kernel mmul[4];
	kernel add;

	input_plio in_A;
	input_plio in_B[4];
	output_plio out_C;

	input_port a_block_param[4];

    SingleTileTest() {
        in_A = input_plio::create(plio_128_bits, "data/A_matrix.txt");
        out_C = output_plio::create(plio_128_bits, "data/C_output.txt");

	add = kernel::create(add_tree);
	source(add) = "src/kernels/add_tree.cpp";
	runtime<ratio>(add) = 1.0;
	dimensions(add.out[0]) = {N*M};
        connect(add.out[0], out_C.in[0]);

	for (int i = 0; i < 4; ++i) {
	    dimensions(add.in[i]) = {N*M};
            in_B[i] = input_plio::create(plio_128_bits, "data/B_"+std::to_string(i)+ ".txt");
	     
            mmul[i] = kernel::create(mmul_skinny);

            runtime<ratio>(mmul[i]) = 1.0;

            dimensions(mmul[i].in[0]) = {N*K};
            dimensions(mmul[i].in[1]) = {M*(K/4)};
            dimensions(mmul[i].out[0]) = {N*M};

            connect(in_A.out[0], mmul[i].in[0]);
            connect(in_B[i].out[0], mmul[i].in[1]);
            connect(mmul[i].out[0], add.in[i]);

	    // Connect parameter ports
	    connect(a_block_param[i], mmul[i].in[2]);

            source(mmul[i]) = "src/kernels/kernels.cpp";
	}
	location<kernel>(add) = tile(0, 1);
	location<kernel>(mmul[0]) = tile(0, 0);
	location<kernel>(mmul[1]) = tile(1, 1);
	location<kernel>(mmul[2]) = tile(0, 2);
	location<kernel>(mmul[3]) = tile(1, 0);
    }
};
	    
