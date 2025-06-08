#include <adf.h>
#include "kernels.h"
#include <aie_api/aie_adf.hpp>
#include "include.h"

using namespace adf;

// Matrix dimensions (T=24 likely matches AIE vector width)
const int T = 24;
const int K = 768;
const int M = 128;

class mmul_graph : public adf::graph {
public:
    // Kernel grid: 4x6 mmul units (T/6=4, T/4=6 → 24 total)
    kernel mmul[T/6][T/4];  
    // 4 adders for partial sums + 1 final adder
    kernel add[5];  

    input_plio in_A;
    input_plio in_B[T/6][T/4];  // 24 input B ports
    output_plio out_C;

    mmul_graph() {
        // Main input/output interfaces
        in_A = input_plio::create(plio_128_bits, "data/A_matrix.txt");
        out_C = output_plio::create(plio_128_bits, "data/C_output.txt");

        // Final adder stage (4-to-1 reduction)
        add[4] = kernel::create_object<add_tree_4>(M);
        source(add[4]) = "src/kernels/add_tree.cpp";
        connect(add[4].out[0], out_C.in[0]); 
        dimensions(add[4].out[0]) = {N*M};
        runtime<ratio>(add[4]) = 1.0;

        // 4 parallel reduction trees (6-to-1 each)
        for (int t = 0; t < 4; ++t) {
            add[t] = kernel::create_object<add_tree_6>(M);
            source(add[t]) = "src/kernels/add_tree.cpp";
            runtime<ratio>(add[t]) = 1.0;
            dimensions(add[t].out[0]) = {N*M};
            dimensions(add[4].in[t]) = {N*M};
            connect(add[t].out[0], add[4].in[t]);

            // 6 mmuls per reduction tree
            for (int i = 0; i < 6; ++i) {
                dimensions(add[t].in[i]) = {N*M};
                // 24 unique B matrix inputs (t*6+i gives 0-23)
                in_B[t][i] = input_plio::create(plio_128_bits, 
                    "data/B_"+std::to_string(t*6+i)+ ".txt");
             
                // Create mmul kernel with block-specific ID
                mmul[t][i] = kernel::create_object<mmul_skinny>(768, 128, 24, t*6+i);
                source(mmul[t][i]) = "src/kernels/kernels.cpp";
                runtime<ratio>(mmul[t][i]) = 1.0;

                // Buffer dimension annotations
                dimensions(mmul[t][i].in[0]) = {N*K};    // A matrix
                dimensions(mmul[t][i].in[1]) = {M*(K/T)}; // B tile
                dimensions(mmul[t][i].out[0]) = {N*M};   // Partial C

                // Data flow connections
                connect(in_A.out[0], mmul[t][i].in[0]);  // Broadcast A
                connect(in_B[t][i].out[0], mmul[t][i].in[1]); // Unique B per mmul
                connect(mmul[t][i].out[0], add[t].in[i]); // Partial → adder
            }

            // Tile mapping: 2 columns per reduction tree
            location<kernel>(add[t]) = tile(2*t+1, 2);
            // Spread mmuls across even columns (0-7)
            location<kernel>(mmul[t][0]) = tile(2*t, 0);
            location<kernel>(mmul[t][1]) = tile(2*t, 1); 
            location<kernel>(mmul[t][2]) = tile(2*t, 2);
            location<kernel>(mmul[t][3]) = tile(2*t, 3); 
            location<kernel>(mmul[t][4]) = tile(2*t, 4);
            location<kernel>(mmul[t][5]) = tile(2*t, 5);
        }
        // Final adder placed at tile (5,0)
        location<kernel>(add[4]) = tile(5, 0);
    }
};

