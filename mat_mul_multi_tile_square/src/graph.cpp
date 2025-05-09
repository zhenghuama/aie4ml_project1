#include <adf.h>
#include "kernels.h"
#include "graph.h"
#include "include.h"

using namespace adf;

MatMulGraph mmul_graph;

int main(void) {

	mmul_graph.init();
	for(int row=0; row<4; row++) {
	    for(int col=0; col<4; col++) {
		mmul_graph.update(mmul_graph.a_block_param[row][col], row);
		mmul_graph.update(mmul_graph.b_block_param[row][col], col);
	    }
	}

	mmul_graph.run(1);
	mmul_graph.end();
	return 0;
}

