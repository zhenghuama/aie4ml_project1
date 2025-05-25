#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H

// Convolution parameters
static const int INPUT_HEIGHT = 8;
static const int INPUT_WIDTH = 8;
static const int KERNEL_HEIGHT = 3;
static const int KERNEL_WIDTH = 3;
static const int OUTPUT_HEIGHT = INPUT_HEIGHT - KERNEL_HEIGHT + 1;
static const int OUTPUT_WIDTH = INPUT_WIDTH - KERNEL_WIDTH + 1;

#endif
