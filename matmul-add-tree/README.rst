Multi-Tile Matrix Multiplication with Adder-Tree
====================================================

2 examples of using an addition tree for multi-tile matrix multiplication.

smaller_example: A = 4x128, B = 128x128, C = 4x128. A*B = C distributed across 5 tiles (4 multiplication, 1 addition)

larger_example: A = 4x784, B = 784x128, C = 4x128. A*B = C distributed across 29 tiles (24 multiplication, 5 addition)
