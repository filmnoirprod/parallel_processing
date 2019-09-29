#!/bin/bash


## Give the Job a descriptive name
#PBS -N fw-serial 

## Output and error files
#PBS -o sertest.out
#PBS -e fwserialtest.err

## Limit memory, runtime etc.
#PBS -l walltime=00:05:00


## Start
## Run the job (use full paths to make sure we execute the correct things
## Just replace the path with your local path to openmp file

module load openmp
openmp_exe=/home/parallel/parlab26/ex1b/part1/fw

for t in 64
do
	echo " "
	echo "#threads = ${t} "
	echo " "
	echo "------------------------------------------"
	echo " "
	# Execute OpenMP executable
	for N in 4096
	do
		export OMP_NUM_THREADS=$t
		$openmp_exe $N   

	done
	echo " "
	echo "------------------------------------------"
done 
