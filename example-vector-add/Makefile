ROOTFS ?= /home/z.ma/Downloads/xilinx-versal-common-v2024.1/rootfs.ext4
PLATFORM_REPO_PATHS := /tools/Xilinx/Vitis/2024.1/base_platforms
BASE_PLATFORM ?= ${PLATFORM_REPO_PATHS}/xilinx_vck190_base_202410_1/xilinx_vck190_base_202410_1.xpfm

compile:
	aiecompiler  --include "./src" --platform ${BASE_PLATFORM}  --workdir=Work   src/graph.cpp
