#include <adf.h>
#include "graph.h"

using namespace adf;

Convolution2DGraph convolution_graph;

int main() {
    convolution_graph.init();
    
    // Configure data window sizes
    adf::config_request req;
    // Input data: 8x8 image of int16 elements
    req.add_buffer(0, INPUT_HEIGHT * INPUT_WIDTH * sizeof(int16));
    // Weights: 3x3 kernel of int16 elements
    req.add_buffer(1, KERNEL_HEIGHT * KERNEL_WIDTH * sizeof(int16));
    // Output: 6x6 result of int16 elements
    req.add_buffer(2, OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(int16));
    convolution_graph.update(req);

    // Execute the program (Start streaming and kernel execution)
    convolution_graph.run(1);
    convolution_graph.end();
    
    return 0;
}
