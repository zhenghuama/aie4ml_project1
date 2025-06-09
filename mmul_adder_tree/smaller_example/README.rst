Multi-Tile Matrix Multiplication with Adder-Tree
====================================================
In the Simple Multi-Tile Matrix Multiplication Example (16x16x16), the entire inner dimension (K) of the matrix multiplication is kept intact and not split across tiles. Here, K represents the inner dimension in an NxKxM multiplication, where A is MxK and B is KxN.

Keeping the inner dimension whole allows each kernel to perform matrix multiplication independently, without needing to sum results between kernels. This is efficient for small matrices, but for large K, splitting the inner dimension and accumulating results between kernels helps manage memory and compute resources.

There are several ways to combine outputs between kernels. One approach is to use the cascade stream to pass data directly between accumulator registers of neighboring kernels. In this example, matrix multiplication tiles are placed next to a central addition kernel (``add_tree.cpp``), which reads output buffers from its neighbors and accumulates the partial results.

.. figure:: ../images/adder-tree.png
   :alt: Adder-tree tiling scheme
   :width: 600px
   :align: center

This adder-tree method is demonstrated with an int16 4x128x128 matrix multiplication spread across 4 tiles. The matrix multiplication kernel (``mmul_skinny.cpp``) uses ``aie::mac`` and ``aie::mul`` intrinsics for flexibility. The K dimension must be a multiple of VEC. VEC is the vectorization factor, and VEC = 16 in our case because of length limits on the default 48-bit accumulator register for int16.

File structure:

::

  .
  ├── graph.h # Declaration of the graph
  ├── graph.cpp  # Initializes, runs, and ends the graph.
  ├── kernels
  │   └── mmul.cpp # matmul kernel implementation
  │   └── add_tree.cpp # addition kernel implementation
  └── kernels.h # declarations of kernel

Kernel Code
*************
The multiplication kernel is called ``mmul_skinny`` because it operates on 4x32x128 dimensions, where the A matrix is skinny with a large K.

    The global variables N = 4, M = 128, K = 128 set the multiplication size. The variable K_Tile represents the portion of the inner dimension handled by each of the 4 tiles, so K_Tile = K/4.

When multiplying two int16 vectors with ``aie::mac``, results are stored in a 48-bit accumulator. These are cast back to int16 vectors using ``to_vector<int16>()``, which automatically saturates the data to prevent overflow.

This approach uses kernel classes with attributes for configuration. The graph code instantiates and runs these kernel classes, passing the appropriate parameters. The ``a_block`` parameter in ``mmul_skinny`` acts like a thread ID, determining which part of the A matrix each kernel processes. The B matrix is split into 4x128 blocks, so no ``b_block`` parameter is needed. While previous designs used scalar port streaming for this, the class-based approach sets these parameters at compile time.


Graph Code
***************
 Each kernel is carefully mapped to a specific adjacent tile to ensure direct read/write buffer access between the addition tile and the multiplication tiles. Without direct buffering between adjacent tiles, bandwidth may be lowered when data is forced to be streamed through the 32 bit AXI4 interface. 

 Note the change in syntax when calling the kernels. The kernels are called with the correct values for the class wrapper parameters. K = 128, M = 128, T = 4, and a_block is set to the tile ID (0-3).


AIE Grid View
****************
Through software simulation, the kernel layout is visualized. Note how output buffers of the mmul kernels are read directly into the adder tree kernel, bypassing the AXI4 stream.

.. image:: image/4x128x128_array.svg
   :alt: Adder-tree Matmul Grid Layout 
   :width: 600px
   :align: center
