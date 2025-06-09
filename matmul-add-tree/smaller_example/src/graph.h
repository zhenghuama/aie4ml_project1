#include <adf.h>
#include "kernels.h"
#include <aie_api/aie_adf.hpp>
#include "include.h"

using namespace adf;

// Graph for 4-tile, 128x128 matrix multiplication
class mmul_4x128x128 : public adf::graph {
private:
    const unsigned int K = 128; // Inner dimension
    const unsigned int M = 128; // Output columns
    const unsigned int T = 4;   // Number of tiles

public:
    kernel mmul[4]; // 4 parallel matrix multiplication kernels
    kernel add;     // 4-input adder tree kernel

    input_plio in_A;      // Input for matrix A
    input_plio in_B[4];   // 4 inputs for matrix B tiles
    output_plio out_C;    // Output for matrix C

    mmul_128x128() {
        // Create input and output streams
        in_A = input_plio::create(plio_128_bits, "data/A_matrix.txt");
        out_C = output_plio::create(plio_128_bits, "data/C_output.txt");

        // Create 4-input adder tree kernel for final accumulation
        add = kernel::create_object<add_tree_4>(M);
        source(add) = "src/kernels/add_tree.cpp";
        runtime<ratio>(add) = 1.0;

        dimensions(add.out[0]) = {N*M};   // Output is full matrix
        dimensions(add.in[4]) = {M};      // Each input is a column block

        connect(add.out[0], out_C.in[0]); // Connect adder output to final output

        // Instantiate and connect 4 mmul kernels, each handling a tile
        for (unsigned int i = 0; i < N; ++i) {
            dimensions(add.in[i]) = {N*M};
            in_B[i] = input_plio::create(plio_128_bits, "data/B_"+std::to_string(i)+ ".txt");

            // Each mmul kernel gets its tile/block ID as 'i'
            mmul[i] = kernel::create_object<mmul_skinny>(K, M, T, i);
            runtime<ratio>(mmul[i]) = 1.0;

            // Set input/output buffer shapes for each kernel
            dimensions(mmul[i].in[0]) = {N*K};         // A: full row block
            dimensions(mmul[i].in[1]) = {M*(K/T)};     // B: tile (partitioned K)
            dimensions(mmul[i].out[0]) = {N*M};        // Output: full matrix block

            // Connect data streams
            connect(in_A.out[0], mmul[i].in[0]);       // Broadcast A to all mmuls
            connect(in_B[i].out[0], mmul[i].in[1]);    // Unique B for each tile
            connect(mmul[i].out[0], add.in[i]);        // Each mmul feeds one adder input

            source(mmul[i]) = "src/kernels/mmul.cpp";
        }

        // Map kernels to hardware tiles for parallel execution
        location<kernel>(add) = tile(0, 1);
        location<kernel>(mmul[0]) = tile(0, 0);
        location<kernel>(mmul[1]) = tile(1, 1);
        location<kernel>(mmul[2]) = tile(0, 2);
        location<kernel>(mmul[3]) = tile(1, 0);
    }
};

