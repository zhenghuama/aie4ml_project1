#include <adf.h>
#include "vec_add.hpp"

using namespace adf;

class vec_add_graph : public graph {
public:
	/*
	 * Note that plio doesn't mean that we use pl but it is convention to call these pl
	 * to reflect internal data transfer abilities
	*/
    input_plio in1 = input_plio::create("in1", plio_32_bits, "data/input1.txt");
    input_plio in2 = input_plio::create("in2", plio_32_bits, "data/input2.txt");
    output_plio out = output_plio::create("out", plio_32_bits, "data/output.txt");

    kernel vadd;

    vec_add_graph() {
        //vector addition kernel
        vadd = kernel::create(vec_add);

        // Connecting io in the ADF graph model
        connect(in1.out[0], vadd.in[0]);
        connect(in2.out[0], vadd.in[1]);
        connect(vadd.out[0], out.in[0]);

        // Configure kernel
        source(vadd) = "vec_add.cpp";
    }
};

