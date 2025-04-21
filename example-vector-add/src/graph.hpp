#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <adf.h>
#include "vec_add.hpp"

class vec_add_graph : public adf::graph {
public:
  adf::input_plio in1, in2;
  adf::output_plio out;
  adf::kernel vadd;
  
  vec_add_graph();  // Constructor declaration
};

#endif

