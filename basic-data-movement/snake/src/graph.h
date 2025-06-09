#include <adf.h>
#include "kernels.h"
#include "kernels/include.h"
#include <aie_api/aie_adf.hpp>

using namespace adf;


#define N 400

class AIEGraph : public adf::graph {
	private:
		//int N = PLATFORM_WIDTH * PLATFORM_HEIGHT;
		kernel kernel_move[N];
	public:
		input_plio plio_in;
		output_plio plio_out;

		AIEGraph() {

      plio_in = input_plio::create(plio_64_bits, "data/input.txt");
      plio_out = output_plio::create(plio_64_bits, "data/output.txt");







      for (int i = 0; i < N; i++) {
        kernel_move[i] = kernel::create(move);
        runtime<ratio>(kernel_move[i])    = 0.9;
        dimensions(kernel_move[i].in[0])  = {16};
        dimensions(kernel_move[i].out[0]) = {16};
        source(kernel_move[i])            = "src/kernels/kernels.cpp";
      }






      // connection of kernels
      for (int i = 0; i < N; i++) {

        // First kernel (use plio)
        if (i == 0){
          connect(plio_in.out[0], kernel_move[i].in[0]);
        }

        // Connect to next kernel if it exists
        if (i < N-1) {
          connect(kernel_move[i].out[0], kernel_move[i+1].in[0]);
        }
        
        // Last kernel (use plio)
        if (i == N-1) {
          connect(kernel_move[i].out[0], plio_out.in[0]);
        } 
      }


      // Constraints //

      // Single Buffering is an inefficient but easier to keep buffers in one place (Don't do this for speed. You want the ping-pong)
      for (int i = 0; i < N; i++) {
          single_buffer(kernel_move[i].in[0]);
          single_buffer(kernel_move[i].out[0]);
      }

      // Placing the buffers and kernels where they are wanted
      // We want a snaking pattern starting from the top left, 
      // so the for loop indexing starts j at the height
      for (int n = 0; n < N; n++) {
        // y coordinate of the tile
        int j = PLATFORM_HEIGHT - 1 - n/PLATFORM_WIDTH;
        // If an EAST/WEST Direction tile then change the x coordinate
        // x coordinate of the tile
        int i = (j%2)?
          (n%PLATFORM_WIDTH)
          :(PLATFORM_WIDTH-1-(n%PLATFORM_WIDTH));
      //}
      //
      // for (int j = PLATFORM_HEIGHT-1; j >= 7; j--) {
      //   for (int i = 0; i < 50; i++) {
          //int x = PLATFORM_WIDTH - i;
          //int n = i + (PLATFORM_HEIGHT - j - 1);

          location<kernel>(kernel_move[n]) = tile(i, j);
          location<stack> (kernel_move[n]) = address(i, j, 24576);
          if (j%2) { // EAST Direction Tile
            if (i == PLATFORM_WIDTH - 1) { // Eastern most tile
              location<buffer>(kernel_move[n].in[0])  = address(i, j, 0);
              // Check if there exists a row below it
              location<buffer>(kernel_move[n].out[0]) = address(i, (j)?j-1:j, (j)?0:8192);
            } else { // just write to the next one in a line
              location<buffer>(kernel_move[n].in[0]) = address(i, j, 0);
              location<buffer>(kernel_move[n].out[0]) = address(i+1, j, 0);
            }
          } else { // WEST Direction Tile
            if (i == 0) { // Western most tile
              location<buffer>(kernel_move[n].in[0])  = address(i,   j, 0);
              // Check if there exists a row below it. Write to second memory bank if it exists.
              // This should be an unecessary ternary since there is always an even number of rows
              location<buffer>(kernel_move[n].out[0]) = address(i, (j)?j-1:j, (j)?0:8192);
            } else { // just write to the next one in a line
              location<buffer>(kernel_move[n].in[0])  = address(i,   j, 0);
              location<buffer>(kernel_move[n].out[0]) = address(i-1, j, 0);
            }
        }
      }
		}
};
