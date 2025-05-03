
#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"

using namespace adf;

class vecAddGraph : public adf::graph {
private:
  kernel vadd;
public:
  input_plio  in1;
  input_plio in2;
  output_plio out;
  vecAddGraph(){
    
    in1  = input_plio::create(plio_32_bits, "data/input1.txt");
    in2  = input_plio::create(plio_32_bits, "data/input2.txt");

    out = output_plio::create(plio_32_bits, "data/output.txt");

    vadd = kernel::create(vec_add);

    adf::connect(in1.out[0], vadd.in[0]);
    connect(in2.out[0], vadd.in[1]);
    connect(vadd.out[0], out.in[0]);
    dimensions(vadd.in[0]) = { NUM_SAMPLES };
    dimensions(vadd.in[1]) = { NUM_SAMPLES };
    dimensions(vadd.out[0]) = { NUM_SAMPLES };

    source(vadd) = "kernels/kernels.cc";

    runtime<ratio>(vadd) = 1.0;


  }
};
