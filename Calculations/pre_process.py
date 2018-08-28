# prepare input files and bash script for calculations
import re
import os
import argparse

# define command line arguments
parser = argparse.ArgumentParser(description='Prepare input files.')
parser.add_argument('base_prm', metavar='base_prm', default='holes.prm', nargs='?',
                    help='Base parameter file to use')
args = parser.parse_args()

# parameters (list of tuples):

# FE degree, quadrature and global refinement
poly_quad_ref = [
    (1,2,7),
    (2,3,6),
    (3,4,5),
    (4,5,5),
    (5,6,5),
    (6,7,4),
    (7,8,4),
    (8,9,4)
]

# Solvers (type, preconditioner and caching)
solvers = [
    ('MF_CG', 'gmg', 'scalar'),
    ('MF_CG', 'gmg', 'tensor2'),
    ('MF_CG', 'gmg', 'tensor4'),
]

# MPI run command
mpicmd = 'mpirun -np 20 /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/main  /home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/Calculations/{0}.prm 2>&1 | tee {0}.toutput\nmv {0}.prm {0}/{0}.prm\n\n'


#
# from here on the actual preprocessing:
#
parameter_file = """
include {0}

subsection Finite element system
  set Polynomial degree = {1}
  set Quadrature order  = {2}
end

subsection Geometry
  set Global refinement  = {3}
end

subsection Linear solver
  set Preconditioner type        = {4}
  set Solver type                = {5}
  set MF caching                 = {6}
end

subsection Misc
  set Output folder = {7}
end
"""

base_prm = args.base_prm
base_name = args.base_prm.split('.')[0]

print 'base parameter file: {0}'.format(base_prm)

filenames = []

# write parameter file
for pqr in poly_quad_ref:
    for s in solvers:
        name = base_name + '_p{0}q{1}r{2}_{3}_{4}_{5}'.format(pqr[0],pqr[1],pqr[2],s[0],s[1],s[2])
        fname = name + '.prm'
        print '{0}'.format(fname)
        fout = open(fname, 'w')
        fout.write(parameter_file.format(
            base_prm,
            pqr[0],
            pqr[1],
            pqr[2],
            s[0],
            s[1],
            s[2],
            name
        ))
        filenames.append(name)

# write bash script
fout = open ('run.sh', 'w')
for f in filenames:
    fout.write(mpicmd.format(f))
