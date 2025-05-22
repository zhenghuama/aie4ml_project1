#include <adf.h>
#include "graph.h"

using namespace adf;

SingleTileTest mmul_graph;

int main(void) {
   mmul_graph.init();
   // Update block parameters
   for (int i = 0; i < 4; ++i) {
       mmul_graph.update(mmul_graph.a_block_param[i], i);
   }
   mmul_graph.run(1);
   mmul_graph.end();
   return 0;
}
