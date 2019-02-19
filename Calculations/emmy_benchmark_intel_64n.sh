#!/bin/bash
#   allocate 64 nodes with 40 CPU per node for 3 hours:
#PBS -l nodes=64:ppn=40,walltime=3:00:00
#   job name
#PBS -N dealii_mf_cm64
#   stdout and stderr files:
#PBS -o /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/bench64.txt -e /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/bench_error64.txt
#   first non-empty non-comment line ends PBS options

# submit with: qsub <name>.sh
cd /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/Calculations/
module load intel64/18.0up03

mpirun -np 1280 /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/main /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/Calculations/__holes_3d_p2q3r5_MF_CG_gmg_tensor4_64node.prm 2>&1 | tee __holes_3d_p2q3r5_MF_CG_gmg_tensor4_64node.toutput
