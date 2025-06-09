#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

using namespace adf;

// Matrix multiplication kernel for tiled architectures.
// T: number of tiles, N: rows of output, M: columns of output
class mmul_skinny {
    int K;       // Inner dimension (shared between A and B)
    int N;       // Rows of output matrix
    int M;       // Columns of output matrix
    int a_block; // Block/tile ID for parallel execution
public:
    mmul_skinny(int K_val, int N_val, int M_val, int id) 
        : K(K_val), N(N_val), M(M_val), a_block(id) {}

    void run(adf::input_buffer<int16>& a_buf,
             adf::input_buffer<int16>& b_buf,
             adf::output_buffer<int16>& c_buf);

    // Registers this kernel and its parameters with the ADF framework
    static void registerKernelClass() {
        REGISTER_FUNCTION(mmul_skinny::run);
        REGISTER_PARAMETER(K);
        REGISTER_PARAMETER(N);
        REGISTER_PARAMETER(M);
        REGISTER_PARAMETER(a_block);
    }
};

// 4-input adder tree for vector reductions (output size: N x M)
class add_tree_4 {
    int N; // Rows of output
    int M; // Columns of output
public:
    add_tree_4(int N_val, int M_val) : N(N_val), M(M_val) {} 
    void run(
        adf::input_buffer<int16>& in0, 
        adf::input_buffer<int16>& in1,  
        adf::input_buffer<int16>& in2,
        adf::input_buffer<int16>& in3,
        adf::output_buffer<int16>& out);
    static void registerKernelClass() {
        REGISTER_FUNCTION(add_tree_4::run);
        REGISTER_PARAMETER(N);
        REGISTER_PARAMETER(M);
    }
};

// 6-input adder tree for wider reductions (output size: N x M)
class add_tree_6 {
    int N;
    int M;
public:
    add_tree_6(int N_val, int M_val) : N(N_val), M(M_val) {} 

    void run(
        adf::input_buffer<int16>& in1,
        adf::input_buffer<int16>& in2,
        adf::input_buffer<int16>& in3,
        adf::input_buffer<int16>& in4,
        adf::input_buffer<int16>& in5,
        adf::input_buffer<int16>& in6,
        adf::output_buffer<int16>& out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(add_tree_6::run);
        REGISTER_PARAMETER(N);
        REGISTER_PARAMETER(M);
    }
};

#endif

