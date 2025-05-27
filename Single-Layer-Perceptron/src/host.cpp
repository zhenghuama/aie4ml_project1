#include <adf.h>
#include "graph.h"

using namespace adf;

PerceptronGraph perceptron_graph;

int main() {
    perceptron_graph.init();
    
    // Configure data window sizes
    adf::config_request req;
    // Input features: 16 float elements
    req.add_buffer(0, INPUT_SIZE * sizeof(float));
    // Weights: 16x4 matrix of float elements
    req.add_buffer(1, INPUT_SIZE * OUTPUT_SIZE * sizeof(float));
    // Biases: 4 float elements
    req.add_buffer(2, OUTPUT_SIZE * sizeof(float));
    // Output: 4 float elements
    req.add_buffer(3, OUTPUT_SIZE * sizeof(float));
    perceptron_graph.update(req);

    // Execute the program (Start streaming and kernel execution)
    perceptron_graph.run(1);
    perceptron_graph.end();
    
    return 0;
}
