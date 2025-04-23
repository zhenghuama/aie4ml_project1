#include <adf.h>
#include "vec_add.hpp"

using namespace adf;

class vec_add_graph : public graph {
public:
    input_plio in1, in2;
    output_plio out;
    kernel vadd;

    vec_add_graph() {
        // Create kernel with explicit template parameters
       // vadd = kernel::create(static_cast<void(*)(input_buffer<int32>&,input_buffer<int32>&,output_buffer<int32>&)>(vec_add));
        vadd = kernel::create(vec_add);
        
        // Configure kernel sources
        source(vadd) = "src/vec_add.cpp";
        
        // Set buffer dimensions
        dimensions(vadd.in[0]) = {1024};
        dimensions(vadd.in[1]) = {1024};
        dimensions(vadd.out[0]) = {1024};
        
        // Create I/O ports
        in1 = input_plio::create("Input1", plio_32_bits, "data/input1.txt");
        in2 = input_plio::create("Input2", plio_32_bits, "data/input2.txt");
        out = output_plio::create("Output", plio_32_bits, "data/output.txt");

        // Connect using simplified buffer syntax
        connect<>(in1.out[0], vadd.in[0]);
        connect<>(in2.out[0], vadd.in[1]);
        connect<>(vadd.out[0], out.in[0]);

	runtime<ratio>(vadd) = 1.0;
    }
};

vec_add_graph mygraph;

