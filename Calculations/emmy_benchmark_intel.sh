#!/bin/bash
#   allocate 1 nodes with 40 CPU per node for 6 hours:
#PBS -l nodes=1:ppn=40,walltime=6:00:00
#   job name
#PBS -N dealii_large_strain
#   stdout and stderr files:
#PBS -o /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/bench.txt -e /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/bench_error.txt
#   first non-empty non-comment line ends PBS options

# submit with: qsub <name>.sh
cd /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/Calculations/
module load intel64/18.0up03

source run.sh
