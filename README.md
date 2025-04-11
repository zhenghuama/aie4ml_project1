# AIE4ML Project 1

This is the repo containing development and documentation of AIE4ML project 1, [Project description](https://docs.google.com/presentation/d/1TGhn2yvmKFxDjmy0UJLweu5KbBvQz4DxouOcfj35NIc/edit#slide=id.g34ae8a4d457_0_272).

### Repo structure
`./dense_example` is an example of a dense+relu layer with 16 bit quantization on 1 AIE tile. You can run software or hardware emulation.

Each workload should be kept in separate folders, e.g., `./vector_add`, `./matmul`, `./matmul_tiling`, etc.

Make sure to maintain a proper `.gitignore` file, so that you do not accidetally include all the logs, waveforms etc. in each commit.