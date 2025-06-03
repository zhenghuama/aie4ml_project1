#include <adf.h>
#include "kernels.h"
#include <aie_api/aie_adf.hpp>
#include "include.h"
#include "layers.h"


using namespace adf;

class MLP: public adf::graph {
public:
    input_plio in_A;
    output_plio out_C;

    layer_768x128 layer1;
    layer_128x128 layer2;
    layer_128x128 layer3;
    layer_128x10 classifier;

    MLP()
        : layer1(0), layer2(1), layer3(2)
    {
        in_A = input_plio::create(plio_128_bits, "data/images.txt");

	connect(in_A.out[0], layer1.in_A);
	connect(layer1.out_C, layer2.in_A);
	connect(layer2.out_C, layer3.in_A);
	connect(layer3.out_C, out_C.in[0]);
    }
}




