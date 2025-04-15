#include <adf.h>
#include "graph.hpp"

using namespace adf;

vec_add_graph mygraph;

int main() {
    mygraph.init();
    
    //Configure data window sizes (1024 elements)
	/* Buffers are allocated to the local memory of a tile
	 * keep in mind that this memory is shared with only the
	 * adjacent tiles
	*/
    adf::config_request req;
    req.add_buffer(0, 1024 * sizeof(int32));
    req.add_buffer(1, 1024 * sizeof(int32));
    req.add_buffer(2, 1024 * sizeof(int32));
    mygraph.update(req);

    mygraph.run(1); // Execute the program (Start streaming and kernal execution)
    mygraph.end();
    return 0;
}

