#include <adf.h>
#include "graph.h"

using namespace adf;

Convolution2DGraph convolution_graph;

int main() {
    convolution_graph.init();

    // Execute the program (Start streaming and kernel execution)
    convolution_graph.run(1);
    convolution_graph.end();
    
    return 0;
}
