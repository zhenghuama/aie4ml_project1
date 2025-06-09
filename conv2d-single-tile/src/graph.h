#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"
#include <aie_api/aie_adf.hpp>

using namespace adf;

/**
 * Single-tile 2D convolution graph for Xilinx AI Engine
 * Optimized for int16 data type
 * Performs a 3x3 convolution on an 8x8 input image
 */
class Convolution2DGraph : public adf::graph {
private:
    kernel k;
public:
    input_plio in_data;
    input_plio in_weights;
    output_plio out_result;

    Convolution2DGraph() {
        // Create input/output interfaces
        in_data = input_plio::create(plio_64_bits, "data/input_data.txt");
        in_weights = input_plio::create(plio_64_bits, "data/weights.txt");
        out_result = output_plio::create(plio_64_bits, "data/output.txt");
        
        // Create the convolution kernel
        k = kernel::create(conv2d_3x3);

        // Set runtime ratio
        runtime<ratio>(k) = 0.9;

        // Set dimensions for data ports
        dimensions(k.in[0]) = {INPUT_HEIGHT * INPUT_WIDTH};
        dimensions(k.in[1]) = {KERNEL_HEIGHT * KERNEL_WIDTH};
        dimensions(k.out[0]) = {OUTPUT_HEIGHT * OUTPUT_WIDTH};

        // Connect I/O ports to kernel
        connect(in_data.out[0], k.in[0]);
        connect(in_weights.out[0], k.in[1]);
        connect(k.out[0], out_result.in[0]);

        // Set source file for kernel
        source(k) = "kernels/kernels.cpp";
    }
};
