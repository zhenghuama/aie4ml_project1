#include <adf.h>
#include "graph.h"

using namespace adf;

mmul_graph g;

int main(void) {
   g.init();
   g.run(1);
   g.end();
   return 0;
}
