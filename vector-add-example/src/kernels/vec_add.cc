/* A simple kernel
 */
#include <adf.h>
#include "adf/io_buffer/io_buffer_types.h"
#include "vec_add.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

void simple(adf::input_buffer<int32> & in1, adf::input_buffer<int32> & in2, adf::output_buffer<int32> & out) {
    auto inIter1 = aie::begin_vector<16>(in1);
    auto inIter2 = aie::begin_vector<16>(in2);
    auto outIter = aie::begin_vector<16>(out);
  for (unsigned i=0; i< in1.size()/16; i++) {
        aie::vector<int32, 16> vec1 = *inIter1;
        aie::vector<int32, 16> vec2 = *inIter2;
        aie::vector<int32, 16> res = aie::add(vec1, vec2);
        *outIter = res;
		
		//Increment indices
		inIter1++;
		inIter2++;
		outIter++;
  }
}
