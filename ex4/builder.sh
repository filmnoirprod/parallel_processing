#!/bin/bash

## Output and error files
#PBS -o builder.out
#PBS -e builder.err

cd /home/parallel/parlab26/ex4
make DEBUG=0 all
./runner.sh
