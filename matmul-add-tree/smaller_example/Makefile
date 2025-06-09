
PLATFORM_REPO_PATHS := /tools/Xilinx/Vitis/2024.1/base_platforms
BASE_PLATFORM ?= ${PLATFORM_REPO_PATHS}/xilinx_vck190_base_202410_1/xilinx_vck190_base_202410_1.xpfm

SRC_DIR := src
AIE_OUT_DIR := Work

AIE_FLAGS := -target=hw
AIE_FLAGS += --platform $(BASE_PLATFORM)
AIE_FLAGS += --include="$(SRC_DIR)"             
AIE_FLAGS += --include="$(SRC_DIR)/kernels"      
AIE_FLAGS += --xlopt=0

AIE_TEST_SOURCES := $(SRC_DIR)/relu_test.cpp $(SRC_DIR)/kernels 

compile_test:
	aiecompiler $(AIE_FLAGS) $(AIE_TEST_SOURCES) -workdir=$(AIE_OUT_DIR)

sim_test:
	#cd data && g++ generate_golden_int16.cpp && ./a.out
	#cd ..
	aiesimulator --pkg-dir=$(AIE_OUT_DIR) --dump-vcd=foo --profile
	grep -v '^T' aiesimulator_output/data/C_output.txt > data/C_output.txt

###########################################################################

clean:
	rm -rf .AIE_SIM_CMD_LINE_OPTIONS ISS_RPC_SERVER_PORT *.json pl_sample_counts .wsdata Work logs	
	rm -rf *.log *.csv *.a .Xil *.vcd *.db aiesimulator_output pl_sample_count*
	> data/C_output.txt
	rm data/a.out
