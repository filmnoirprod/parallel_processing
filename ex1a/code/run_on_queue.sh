#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_gol

## Output and error files
#PBS -o run_gol2.out
#PBS -e run_gol2.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:20:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd /home/parallel/parlab26/ex1a
export OMP_NUM_THREADS=1
echo "#threads = 1"
echo " "
./game_of_life 64 1000
./game_of_life 1024 1000
./game_of_life 4096 1000
echo " "
echo "-----------------------------------------------"
echo " "
export OMP_NUM_THREADS=2
echo "#threads = 2"
echo " "
./game_of_life 64 1000
./game_of_life 1024 1000
./game_of_life 4096 1000
echo " "
echo "-----------------------------------------------"
echo " "
export OMP_NUM_THREADS=4
echo "#threads = 4"
echo " "
./game_of_life 64 1000
./game_of_life 1024 1000
./game_of_life 4096 1000
echo " "
echo "-----------------------------------------------"
echo " "
export OMP_NUM_THREADS=6
echo "#threads = 6"
echo " "
./game_of_life 64 1000
./game_of_life 1024 1000
./game_of_life 4096 1000
echo " "
echo "-----------------------------------------------"
echo " "
export OMP_NUM_THREADS=8
echo "#threads = 8"
echo " "
./game_of_life 64 1000
./game_of_life 1024 1000
./game_of_life 4096 1000
echo " "
echo "-----------------------------------------------"
echo " "
