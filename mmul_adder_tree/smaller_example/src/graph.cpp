#include <adf.h>
#include "graph.h"
#include "include.h"

using namespace adf;

mmul_4x128x128 mmul_graph;

int main(void) {
   mmul_graph.init();
   mmul_graph.run(1);
   mmul_graph.end();
   return 0;
}
