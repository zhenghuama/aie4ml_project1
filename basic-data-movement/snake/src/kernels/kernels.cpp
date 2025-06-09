#include <aie_api/aie.hpp>
#include <adf.h>
#include "include.h"


void move(
    adf::input_buffer<int16>& mov_in, adf::output_buffer<int16>& mov_out
) {
    auto inIter1=aie::begin(mov_in);
    auto outIter=aie::begin(mov_out);
    //for(int i=0;i<4;i++) {
    *outIter++=(*inIter1);
    //}
}
