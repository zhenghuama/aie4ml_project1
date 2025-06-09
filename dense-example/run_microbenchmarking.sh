#!/bin/bash
set -e

# Define arrays with given values
K_vals=(4 16 32 64 16 32 256 16 128 32 64 16 64 128)
N_vals=(16 16 16 16 64 32 16 256 32 128 64 512 128 64)

# Iterate through the arrays and call make
for i in "${!K_vals[@]}"; do
    M=4
    K=${K_vals[$i]}
    N=${N_vals[$i]}
    m=2
    k=4
    n=8

    echo "Running make with: i=$i SUBDIR=r M=$M K=$K N=$N m=$m k=$k n=$n"

    make clean && make i=$i M=$M K=$K N=$N m=$m k=$k n=$n run_sim

    diff -w data/matC0_sim.txt data/matC0.txt > /dev/null || { 
        echo -e "\n\nError: Output does not match\n\n" >&2
        exit 1
    }

done
