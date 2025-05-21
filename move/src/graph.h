#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"
#include <aie_api/aie_adf.hpp>

using namespace adf;

class AIEGraph : public adf::graph {
	private:
		kernel kernel_move;
	public:
		input_plio plio_in;
		output_plio plio_out;

		AIEGraph() {

			plio_in = input_plio::create(plio_64_bits, "data/input.txt");
			plio_out = output_plio::create(plio_64_bits, "data/output.txt");
			kernel kernel_move = kernel::create(move);


			runtime<ratio>(kernel_move) = 0.9;

			dimensions(kernel_move.in[0]) = {16};
			dimensions(kernel_move.out[0]) = {16};

			connect(plio_in.out[0], kernel_move.in[0]);
            connect(kernel_move.out[0], plio_out.in[0]);

			source(kernel_move) = "src/kernels/kernels.cpp";

            // Constraints //
      
            // Single Buffering is an inefficient but easier to keep buffers in one place (Don't do this for speed. You want the ping-pong)
            single_buffer(kernel_move.in[0]);
            single_buffer(kernel_move.out[0]);

            // Buffer the location of the memories:
            location<buffer>(kernel_move.in[0]) = bank(0, 7, 0);
            location<buffer>(kernel_move.out[0]) = bank(1, 7, 0);

            location<kernel>(kernel_move) = tile(0, 7);
		}
	
};
