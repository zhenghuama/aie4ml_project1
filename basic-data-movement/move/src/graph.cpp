#include <adf.h>
#include "kernels.h"
#include "graph.h"
#include "kernels/include.h"

using namespace adf;

AIEGraph move_by_1;

int main(void) {
	move_by_1.init();
	move_by_1.run(1);
	move_by_1.end();
	return 0;
}
