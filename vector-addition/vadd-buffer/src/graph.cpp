
#include <adf.h>
#include "kernels.h"
#include "graph.h"

using namespace adf;

vecAddGraph v_graph;

int main(void) {
  v_graph.init();
  v_graph.run(1);
  v_graph.end();
  return 0;
}
