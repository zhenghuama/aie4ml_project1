#include <adf.h>
#include "vec_add.hpp"
#include "graph.hpp"

using namespace adf;

vec_add_graph::vec_add_graph() {
  in1 = input_plio::create("in1", plio_32_bits, "data/input1.txt");
  in2 = input_plio::create("in2", plio_32_bits, "data/input2.txt");
  out = output_plio::create("out", plio_32_bits, "data/output.txt");
  
  vadd = kernel::create(vec_add);
  
  // FIX: Add explicit adf::window<size> with namespace
  connect<adf::window<1024>>(in1.out[0], vadd.in[0]);
  connect<adf::window<1024>>(in2.out[0], vadd.in[1]);
  connect<adf::window<1024>>(vadd.out[0], out.in[0]);
  
  source(vadd) = "vec_add.cpp";
}

// Single instance (rename to match host.cpp)
vec_add_graph mygraph;  // Changed from graph_ob to mygraph

