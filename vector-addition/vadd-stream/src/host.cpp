
#include "graph.hpp"

simpleGraph vadd_graph;

int main(int argc, char** argv) {
  vadd_graph.init();
  vadd_graph.run(256);
  vadd_graph.end();

  return 0;
}
