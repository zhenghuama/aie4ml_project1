PLATFORM_REPO_PATHS := /tools/Xilinx/Vitis/2024.1/base_platforms
BASE_PLATFORM := ${PLATFORM_REPO_PATHS}/xilinx_vck190_base_202410_1/xilinx_vck190_base_202410_1.xpfm

compile:
	aiecompiler  --include "./src" --platform ${BASE_PLATFORM}  --workdir=Work   src/graph.cpp

clean:
	rm -rf Work
	rm -rf aie_out
	rm -f *.o
	rm -f *.xpe
	rm -f *.log
	rm -rf logs
	rm -f *.csv
	rm -f *.db
	rm -f *.jou
	echo "Clean done"
