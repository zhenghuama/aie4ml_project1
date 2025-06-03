#include <adf.h>
#include "layers.h"

using namespace adf;

layer_768x128 relu_graph(0);

int main(void) {
   relu_graph.init();
   relu_graph.run(1);
   relu_graph.end();
   return 0;
}
