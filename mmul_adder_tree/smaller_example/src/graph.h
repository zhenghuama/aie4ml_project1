#include <adf.h>
#include "kernels.h"
#include <aie_api/aie_adf.hpp>
#include "include.h"

using namespace adf;

class mmul_4x128x128 : public adf::graph {
private:
	const unsigned int K = 128;
	const unsigned int M = 128;
	const unsigned int T = 4;

public:
        kernel mmul[4];
	kernel add;

	input_plio in_A;
	input_plio in_B[4];

	output_plio out_C;

    mmul_128x128() {
        in_A = input_plio::create(plio_128_bits, "data/A_matrix.txt");
        out_C = output_plio::create(plio_128_bits, "data/C_output.txt");

	// Syntax change: the kernel is initialized through the wrapper class and called with the correct attributes
	add = kernel::create_object<add_tree_4>(M);
	source(add) = "src/kernels/add_tree.cpp";
	runtime<ratio>(add) = 1.0;

	dimensions(add.out[0]) = {N*M};
	dimensions(add.in[4]) = {M};

        connect(add.out[0], out_C.in[0]);

	for (unsigned int i = 0; i < N; ++i) {
	    dimensions(add.in[i]) = {N*M};
            in_B[i] = input_plio::create(plio_128_bits, "data/B_"+std::to_string(i)+ ".txt");

	    // Note how the necessary attributes are given to the kernel wrapper class
            mmul[i] = kernel::create_object<mmul_skinny>(K, M, T, i);

            runtime<ratio>(mmul[i]) = 1.0;

            dimensions(mmul[i].in[0]) = {N*K};
            dimensions(mmul[i].in[1]) = {M*(K/T)};
            dimensions(mmul[i].out[0]) = {N*M};

            connect(in_A.out[0], mmul[i].in[0]);
            connect(in_B[i].out[0], mmul[i].in[1]);
            connect(mmul[i].out[0], add.in[i]);

            source(mmul[i]) = "src/kernels/mmul.cpp";
	}
	location<kernel>(add) = tile(0, 1);
	location<kernel>(mmul[0]) = tile(0, 0);
	location<kernel>(mmul[1]) = tile(1, 1);
	location<kernel>(mmul[2]) = tile(0, 2);
	location<kernel>(mmul[3]) = tile(1, 0);
    }
};
