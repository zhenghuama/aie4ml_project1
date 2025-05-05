#include <adf.h>
#include "kernels.h"
#include "graph.h"
#include "include.h"

using namespace adf;

MatrixMultInt32 mmul_graph;

int main(void) {
	mmul_graph.init();
	mmul_graph.run();
	mmul_graph.end();
	return 0;
}
