#include <adf.h>
#include "graph.h"

using namespace adf;

SingleTileTest mmul_graph;

int main(void) {
   mmul_graph.init();
   mmul_graph.run(1);
   mmul_graph.end();
   return 0;
}
