
#include <adf.h>
#include "adf/new_frontend/types.h"
#include "vec_add.h"

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel vadd;
  kernel second;
public:
  input_plio  in1;
  input_plio in2;
  output_plio out;
  simpleGraph(){
    
    in1  = input_plio::create(plio_32_bits, "data/input1.txt");
    in2 = input_plio::create(plio_32_bits, "data/input2.txt");
    out = output_plio::create(plio_32_bits, "data/output.txt");

    vadd = adf::kernel::create(vec_add);

    adf::connect<>(in1.out[0], vadd.in[0]);
    adf::connect<>(in2.out[0], vadd.in[1]);
    adf::connect<>(vadd.out[0], out.in[0]);

    dimensions(vadd.in[0]) = {1028};
    dimensions(vadd.in[1]) = {1028};
    dimensions(vadd.out[0]) = {1028};

    source(vadd) = "kernels/vec_add.cc";

    runtime<ratio>(vadd) = 1.0;

  }
};
