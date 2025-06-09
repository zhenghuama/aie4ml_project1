#include <adf.h>
#include  "kernels.hpp"

using namespace adf;

class simpleGraph : public graph {
  private:
    kernel vadd;

  public: 
    input_plio p_in0;
    input_plio p_in1;
    output_plio p_out0;

    simpleGraph() {
      vadd = kernel::create(aie_vadd_stream);
      source(vadd) = "kernels/vadd_stream.cc";

      p_in0 = input_plio::create("data_in0", plio_32_bits, "data/input0.txt");
      p_in1 = input_plio::create("data_in1", plio_32_bits, "data/input1.txt");
      p_out0 = output_plio::create("data_out0", plio_32_bits, "data/output.txt");


      // connect ports and kernel

      connect<stream>(p_in0.out[0], vadd.in[0]);
      connect<stream>(p_in1.out[0], vadd.in[1]);
      connect<stream>(vadd.out[0], p_out0.in[0]);
      
      // kernel runtime ratio
      runtime<ratio>(vadd) = 1;
    };
};
