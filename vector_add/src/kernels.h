
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include <adf.h>

  void simple(
      adf::input_buffer<int32> & data1, 
      adf::input_buffer<int32> & data2,
      adf::output_buffer<int32> & out);

#endif
