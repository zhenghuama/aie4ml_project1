

#include <adf.h>
#include<aie_api/aie_adf.hpp>


void aie_vadd_stream(input_stream_int32 *data_in0, input_stream_int32 *data_in1, output_stream_int32 *data_out0) {
  aie::vector<int32, 4> a = readincr_v<4>(data_in0);
  aie::vector<int32, 4> b = readincr_v<4>(data_in1);
  aie::vector<int32, 4> result = aie::add(a,b);
  writeincr_v<4>(data_out0, result);
}
