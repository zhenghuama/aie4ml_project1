#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

using namespace adf;

class mmul_skinny {
    int K;
    int M;
    int T;
    int a_block;
public:
    mmul_skinny(int K_val, int M_val, int T_val, int id) :  K(K_val), M(M_val), T(T_val), a_block(id) {}

    void run(adf::input_buffer<int16>& a_buf,
             adf::input_buffer<int16>& b_buf,
             adf::output_buffer<int16>& c_buf);

    static void registerKernelClass() {
        REGISTER_FUNCTION(mmul_skinny::run);
	REGISTER_PARAMETER(K);
	REGISTER_PARAMETER(M);
	REGISTER_PARAMETER(T);
        REGISTER_PARAMETER(a_block);
    }
};

class add_tree_4 {
    int M;
public:
    
    add_4(int M_val) : M(M_val) {} 
    void run(
    adf::input_buffer<int16>& in0, 
    adf::input_buffer<int16>& in1,  
    adf::input_buffer<int16>& in2,
    adf::input_buffer<int16>& in3,
    adf::output_buffer<int16>& out);
    static void registerKernelClass() {
        REGISTER_FUNCTION(add_4::run);
        REGISTER_PARAMETER(M);
    }
};

#endif
