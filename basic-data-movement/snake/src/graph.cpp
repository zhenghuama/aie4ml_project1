#include <adf.h>
#include "kernels.h"
#include "graph.h"
#include "kernels/include.h"

using namespace adf;

AIEGraph snake_graph;

int main(void) {
	snake_graph.init();
	snake_graph.run(1);
	snake_graph.end();
	return 0;
}
