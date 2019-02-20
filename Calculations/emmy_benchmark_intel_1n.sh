#!/bin/bash
#   allocate 1 nodes with 40 CPU per node for 3 hours:
#PBS -l nodes=1:ppn=40:noturbo,walltime=3:00:00
#   job name
#PBS -N dealii_mf_cm1
#   stdout and stderr files:
#PBS -o /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/bench8.txt -e /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/bench_error8.txt
#   first non-empty non-comment line ends PBS options

# submit with: qsub <name>.sh
cd /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/Calculations/
module load intel64/18.0up03

mpirun -np 20 /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/main /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/Calculations/__holes_3d_p2q3r3_MF_CG_gmg_tensor4_1node.prm 2>&1 | tee __holes_3d_p2q3r3_MF_CG_gmg_tensor4_1node.toutput
mv __holes_3d_p2q3r3_MF_CG_gmg_tensor4_1node.toutput Emmy_RRZE_scaling/holes_3d_p2q3r3_MF_CG_gmg_tensor4/holes_3d_p2q3r3_MF_CG_gmg_tensor4_1node.toutput
