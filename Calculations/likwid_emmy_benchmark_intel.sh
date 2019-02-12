#!/bin/bash
#   allocate 1 nodes (with likwid) with 40 CPU per node for 3 hour and fix CPU clock:
#PBS -l nodes=1:ppn=40:likwid:f2.2,walltime=3:00:00
#   job name
#PBS -N dealii_mf_cm_likwid
#   stdout and stderr files:
#PBS -o /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/likwid_bench.txt -e /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/likwid_bench_error.txt
#   first non-empty non-comment line ends PBS options

# submit with: qsub <name>.sh
cd /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/Calculations/
module load intel64/18.0up03 likwid/4.2.1

source likwid_run.sh
