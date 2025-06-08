#include <adf.h>
#include "layers.h"
#include "MLP.h"
#include "include.h"

using namespace adf;

MLP mlp_graph;

int main(void) {
   mlp_graph.init();
   // Update block parameters
   for (int i = 0; i < T; ++i) {
       mlp_graph.update(mlp_graph.layer1.a_block_param[i], i*(K/T/32));
   }
   mlp_graph.run(1);
   mlp_graph.end();
   return 0;
}
