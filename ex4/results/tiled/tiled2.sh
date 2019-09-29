#!/bin/bash
#
## run_dmm.sh -- Run DMM in GPU systems.
##
## This is an example script for submitting to the Torque system your
## experiments. You can freely change the script as you like. Please respect the
## `walltime' attribute.
##
## Please remember to compile your code with `make DEBUG=0' before
## submitting. If you plan to use this script, we recommend you to enable only
## the GPU kernels to avoid unnecessary executions of the serial and OpenMP
## version of the code wasting your time. Use similar scripts with just the
## required executions for these versions.
##
## Copyright (C) 2019, Computing Systems Laboratory (CSLab)
##

#PBS -o tiled_best.out
#PBS -e tiled_best.err
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=24:cuda

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

gpu_kernels="2"
problem_sizes="256 512 1024 2048"
block_sizes="32"
gpu_prog="./cuda/dmm_main"

## Change this to the directory of your executable!
cd /home/parallel/parlab26/ex4
echo "Benchmark started on $(date) in $(hostname)"
for i in $gpu_kernels; do
    for m in $problem_sizes; do
	for n in $problem_sizes; do
	    for k in $problem_sizes; do
		for b in $block_sizes; do
		    GPU_KERNEL=$i GPU_BLOCK_SIZEX=$b GPU_BLOCK_SIZEY=$b $gpu_prog $m $n $k
		done
	    done
	done
    done
done
echo "Benchmark ended on $(date) in $(hostname)"
