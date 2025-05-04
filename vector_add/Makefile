#Simple Makefile for vector addition program

PLATFORM_REPO_PATHS := /tools/Xilinx/Vitis/2024.1/base_platforms
BASE_PLATFORM ?= ${PLATFORM_REPO_PATHS}/xilinx_vck190_base_202410_1/xilinx_vck190_base_202410_1.xpfm

SRC_DIR := src
AIE_OUT_DIR := Work

AIE_FLAGS := -target=hw
AIE_FLAGS += --platform $(BASE_PLATFORM)
AIE_FLAGS += --include="$(SRC_DIR)"             
AIE_FLAGS += --include="$(SRC_DIR)/kernels"      
AIE_FLAGS += --xlopt=0

AIE_SOURCES := $(SRC_DIR)/graph.cpp $(SRC_DIR)/kernels/kernels.cc

compile:
	aiecompiler $(AIE_FLAGS) $(AIE_SOURCES) -workdir=$(AIE_OUT_DIR)

sim:
	aiesimulator --pkg-dir=$(AIE_OUT_DIR) --dump-vcd=foo
	grep -v '^T' aiesimulator_output/data/output.txt > data/output.txt
	@echo "##################### Comparing simulated output to correct output ####################"
	@if diff -w data/output.txt data/correct_output.txt; then \
		echo "Success - AIE output matches correct output."; \
	else \
		echo "Failure - AIE output does not match correct output."; \
	fi
	vitis_analyzer aiesimulator_output/default.aierun_summary


###########################################################################

clean:
	rm -rf .AIE_SIM_CMD_LINE_OPTIONS ISS_RPC_SERVER_PORT *.json pl_sample_counts .wsdata Work logs	
	rm -rf *.log *.csv *.a .Xil *.vcd *.db aiesimulator_output
	> data/output.txt
