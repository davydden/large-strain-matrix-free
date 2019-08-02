#PBS -l nodes=1:ppn=40
#PBS -l walltime=45:00:00
#PBS -N csl_benchmark
#PBS -q gold
#PBS -j oe

echo -n "this script is running on: "
hostname -f
date

echo ""
echo "### PBS_NODEFILE (${PBS_NODEFILE}) ###"
cat ${PBS_NODEFILE}
echo ""
echo ""

#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"

# since openmpi is compiled with PBS(Torque) support there is no need to
# specify the number of processes or a hostfile to mpirun.
cd /home/kronbichler/sw/denis/large-strain-matrix-free/Calculations

module load gcc/9 mpi/openmpi-4.0.1

python pre_process.py --likwid --dir=CSL_Munich --prefix=/home/kronbichler/sw/denis/large-strain-matrix-free/build_avx512/ --calc=/home/kronbichler/sw/denis/large-strain-matrix-free/Calculations/

source likwid_run.sh 
