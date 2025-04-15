Basic "Hello World!" example for AI Engine programming. This implements a single kernel allocated to a single tile for vector addition of 2 int32 vectors of size 1028.
The graph, kernal, and source code are located in src. 

The data files hold example vectors A and B to be read and processed as a simple testing example. The output.txt holds A+B after the computation.

File Structure:
```bash
├── Makefile
├── data
│   ├── input1.txt
│   ├── input2.txt
│   └── output.txt
└── src
    ├── graph.cpp
    ├── host.cpp
    └── vec_add.cpp
```
