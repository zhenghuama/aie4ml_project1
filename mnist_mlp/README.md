MNIST MLP Example

Target Dimensions are 768 -> 128 -> 128 -> 128 -> 10

So far we have just the skeleton along with a relu_test graph which tests the first layer. Note the data director holds tests matrices for the test graph. There is no input/output images or weights for the entire MLP included yet. This is becuase I (Gram) have run into an issue with converting the int48 accumulator to an int16 vector while properly clamping the overflowed integers into the max/min range of int16. Right now, it just converts those overflowed integers to zero.

This error can be found in the mmul_skinny kernel at the bottom of the function. It is only relevant when some multiplication is brought outside of the int16 range, but I believe it effects the behavior of the nueral network.
