#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

using namespace adf;

// Matrix multiplication kernel class wrapper for "skinny" matrices (one dimension much larger)
class mmul_skinny {
    int K;       // Inner dimension (columns of A / rows of B)
    int M;       // Rows of A and C
    int T;       // Columns of B and C
    int a_block; // Block ID for parallel processing
public:
    mmul_skinny(int K_val, int M_val, int T_val, int id) 
        : K(K_val), M(M_val), T(T_val), a_block(id) {}

    void run(adf::input_buffer<int16>& a_buf,
             adf::input_buffer<int16>& b_buf,
             adf::output_buffer<int16>& c_buf);

    // Registers kernel function and parameters with the ADF runtime
    static void registerKernelClass() {
        REGISTER_FUNCTION(mmul_skinny::run);
        REGISTER_PARAMETER(K);
        REGISTER_PARAMETER(M);
        REGISTER_PARAMETER(T);
        REGISTER_PARAMETER(a_block);
    }
};

// 4-input addition tree kernel (element-wise sum of 4 vectors)
class add_tree_4 {
    int M; // Vector length
public:
    add_tree_4(int M_val) : M(M_val) {}

    void run(
        adf::input_buffer<int16>& in0, 
        adf::input_buffer<int16>& in1,  
        adf::input_buffer<int16>& in2,
        adf::input_buffer<int16>& in3,
        adf::output_buffer<int16>& out);

    static void registerKernelClass() {
        REGISTER_FUNCTION(add_tree_4::run);
        REGISTER_PARAMETER(M);
    }
};

#endif
