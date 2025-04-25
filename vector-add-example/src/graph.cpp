
#include <adf.h>
#include "vec_add.h"
#include "graph.h"

using namespace adf;

simpleGraph mygraph;

int main(void) {
  mygraph.init();
  mygraph.run(4);
  mygraph.end();
  return 0;
}
