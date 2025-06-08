#include <adf.h>
#include "kernels.h"
#include <aie_api/aie_adf.hpp>
#include "include.h"

using namespace adf;

class layer_768x128: public adf::graph {

private:
	const unsigned int K = 768;
	const unsigned int M = 128;
	const unsigned int T = 24;

public:
        kernel mmul[24/6][24/4];
	kernel add[5];

	input_plio in_A;
	input_plio in_B[24/6][24/4];
	input_plio in_bias;
	output_plio out_C;

    layer_768x128(int layer_param) {
        in_A = input_plio::create(plio_128_bits, "data/A_matrix.txt");
        out_C = output_plio::create(plio_128_bits, "data/C_output.txt");
	in_bias = input_plio::create(plio_128_bits, "data/bias_"+std::to_string(layer_param)+".txt");
	add[4] = kernel::create_object<add_relu_4>(M);
	source(add[4]) = "src/kernels/add_tree_relu.cpp";
	connect(add[4].out[0], out_C.in[0]); 
	connect(in_bias.out[0], add[4].in[4]); 
	dimensions(add[4].out[0]) = {N*M};
	dimensions(add[4].in[4]) = {M};
	runtime<ratio>(add[4]) = 1.0;


	for (int t = 0; t < 4; ++t) {
	    add[t] = kernel::create_object<add_6>(M);
	    source(add[t]) = "src/kernels/add_tree.cpp";
	    runtime<ratio>(add[t]) = 1.0;
	    dimensions(add[t].out[0]) = {N*M};
	    dimensions(add[4].in[t]) = {N*M};
            connect(add[t].out[0], add[4].in[t]);

	    for (int i = 0; i < 6; ++i) {
	        dimensions(add[t].in[i]) = {N*M};
                in_B[t][i] = input_plio::create(plio_128_bits, "data/B_"+std::to_string(t*6+i)+ ".txt");
	     
                mmul[t][i] = kernel::create_object<mmul_skinny>(K, M, T, t*6+i);
                source(mmul[t][i]) = "src/kernels/kernels.cpp";

                runtime<ratio>(mmul[t][i]) = 1.0;

                dimensions(mmul[t][i].in[0]) = {N*K};
                dimensions(mmul[t][i].in[1]) = {M*(K/T)};
                dimensions(mmul[t][i].out[0]) = {N*M};

                connect(in_A.out[0], mmul[t][i].in[0]);
                connect(in_B[t][i].out[0], mmul[t][i].in[1]);
                connect(mmul[t][i].out[0], add[t].in[i]);
	    }
	    location<kernel>(add[t]) = tile(2*t+1, 2);
	    location<kernel>(mmul[t][0]) = tile(2*t, 0);
	    location<kernel>(mmul[t][1]) = tile(2*t, 1); 
	    location<kernel>(mmul[t][2]) = tile(2*t, 2);
	    location<kernel>(mmul[t][3]) = tile(2*t, 3); 
	    location<kernel>(mmul[t][4]) = tile(2*t, 4);
	    location<kernel>(mmul[t][5]) = tile(2*t, 5);
	}
	location<kernel>(add[4]) = tile(5, 0);
    }
};



class layer_128x128 : public adf::graph {
private:
	const unsigned int K = 128;
	const unsigned int M = 128;
	const unsigned int T = 4;

public:
        kernel mmul[4];
	kernel add;

	input_plio in_A;
	input_plio in_B[4];
	input_plio in_bias;
	output_plio out_C;

    layer_128x128(int layer_param) {
        in_A = input_plio::create(plio_128_bits, "data/A_matrix.txt");
	in_bias = input_plio::create(plio_128_bits, "data/bias_"+std::to_string(layer_param)+".txt");
        out_C = output_plio::create(plio_128_bits, "data/C_output.txt");

	add = kernel::create_object<add_relu_4>(M);
	source(add) = "src/kernels/add_tree_relu.cpp";
	runtime<ratio>(add) = 1.0;

	dimensions(add.out[0]) = {N*M};
	dimensions(add.in[4]) = {M};

        connect(add.out[0], out_C.in[0]);
	connect(in_bias.out[0], add.in[4]);

	for (unsigned int i = 0; i < N; ++i) {
	    dimensions(add.in[i]) = {N*M};
            in_B[i] = input_plio::create(plio_128_bits, "data/B_"+std::to_string(i)+ ".txt");

            mmul[i] = kernel::create_object<mmul_skinny>(K, M, T, i);

            runtime<ratio>(mmul[i]) = 1.0;

            dimensions(mmul[i].in[0]) = {N*K};
            dimensions(mmul[i].in[1]) = {M*(K/4)};
            dimensions(mmul[i].out[0]) = {N*M};

            connect(in_A.out[0], mmul[i].in[0]);
            connect(in_B[i].out[0], mmul[i].in[1]);
            connect(mmul[i].out[0], add.in[i]);

            source(mmul[i]) = "src/kernels/kernels.cpp";
	}
	location<kernel>(add) = tile(0, 1);
	location<kernel>(mmul[0]) = tile(0, 0);
	location<kernel>(mmul[1]) = tile(1, 1);
	location<kernel>(mmul[2]) = tile(0, 2);
	location<kernel>(mmul[3]) = tile(1, 0);
    }
};
