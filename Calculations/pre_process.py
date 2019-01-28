# prepare input files and bash script for calculations
import re
import os
import argparse

# define command line arguments
parser = argparse.ArgumentParser(
    description='Prepare input files.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--base_prm', metavar='base_prm', default='holes.prm',
                    help='Base parameter file to use')
parser.add_argument('--dir', metavar='dir', default='Emmy_RRZE',
                    help='Subdirectory to store calculations')
parser.add_argument('--prefix', metavar='prefix', default='/home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build/',
                    help='Build directory with executable `main`')
parser.add_argument('--calc', metavar='calc', default='/home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/Calculations/',
                    help='Directory with calculations where .prm files will be generated and `base_prm` is located')
parser.add_argument('--mpirun', metavar='mpirun', default='mpirun -np 20',
                    help='mpi run command with cores')
parser.add_argument('--likwid', help='Prepare LIKWID run', action="store_true")
parser.add_argument('--breakdown', help='LIKWID run with breakdown of costs', action="store_true")
args = parser.parse_args()

# parameters (list of tuples):

# FE degree, quadrature, global refinement, dim
poly_quad_ref_dim = [
    (1,2,7,2),
    (2,3,6,2),
    (3,4,5,2),
    (4,5,5,2),
    (5,6,5,2),
    (6,7,4,2),
    (7,8,4,2),
    (8,9,4,2),
    (1,2,4,3),
    (2,3,3,3),
    (3,4,2,3),
    (4,5,2,3),
]

poly_quad_ref_dim_likwid = [
    (2,3,6,2),
    (4,5,5,2),
    (6,7,4,2),
    (2,3,3,3),
    (4,5,2,3),
]

poly_quad_ref_dim_likwid_breakdown = [
    (2,3,1,2),
    (4,5,1,2),
    (6,7,1,2),
    (2,3,0,3),
    (4,5,0,3),
]

# Solvers (type, preconditioner and caching)
solvers = [
    ('MF_CG', 'gmg', 'scalar'),
    ('MF_CG', 'gmg', 'tensor2'),
    ('MF_CG', 'gmg', 'tensor4'),
    ('CG',    'amg', 'scalar'),
]

solvers_likwid = [
    ('MF_CG', 'gmg', 'scalar'),
    ('MF_CG', 'gmg', 'tensor2'),
    ('MF_CG', 'gmg', 'tensor4'),
    ('CG',    'amg', 'scalar'),
]

solvers_likwid_breakdown = [
    ('MF_CG', 'gmg', 'tensor4')
]


# MPI run command (override if use LIKWID)
mpirun_args = args.mpirun
if args.likwid:
  mpirun_args = 'likwid-mpirun -np 10 -nperdomain S:10 -g MEM_DP -m'

mpicmd = mpirun_args + ' ' + args.prefix + 'main ' + args.calc + '{0}.prm 2>&1 | tee {0}.toutput\nmv {0}.toutput {1}{0}/{0}.toutput\n\n'

# if run likwid, do a smaller subset of test cases:
if args.likwid:
  solvers = solvers_likwid
  poly_quad_ref_dim = poly_quad_ref_dim_likwid
  if args.breakdown:
    solvers = solvers_likwid_breakdown
    poly_quad_ref_dim = poly_quad_ref_dim_likwid_breakdown

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
  set Dimension          = {8}
end

subsection Linear solver
  set Solver type                = {4}
  set Preconditioner type        = {5}
  set MF caching                 = {6}
  set Preconditioner AMG aggregation threshold = 1e-4
  set MF Chebyshev coarse accurate eigenvalues = false
  set MF Chebyshev coarse = true
end

subsection Misc
  set Output folder = {7}
  set Output points = {9}
  set Output solution = {13}
end

subsection Boundary conditions
  set Dirichlet IDs and expressions = {10}
  set Dirichlet IDs and component mask = {11}
  set Neumann IDs and expressions = {12}
end

subsection Time
  set End time  = {14}
end
"""

output_points  = {
  2: '0,0',
  3: '0,0,0.5e-3'
}

dirichlet_id   = {
  2: '1:0,0',
  3: '1:0,0,0;2:0,0,0'
}

dirichlet_mask = {
  2: '1:true,true',
  3: '1:true,true,true;2:false,false,true'
}

neumann_bc     = {
  2: '11:12.5e3*t,0',
  3: '11:12.5e3*t,0,12.5e3*t'
}

base_prm = args.base_prm
base_name = args.base_prm.split('.')[0]
out_dir = args.dir

# add prefix to the base name
if args.likwid:
  base_name = 'likwid_' + base_name
  out_dir = 'LIKWID_' + out_dir
  if args.breakdown:
    out_dir = out_dir + '_breakdown'

if out_dir and not out_dir.endswith('/'):
    out_dir = out_dir + '/'

print 'base parameter file: {0}'.format(base_prm)
print 'output directory:    {0}'.format(out_dir)
print 'mpi comand:\n{0}'.format(mpicmd)

filenames = []

# write parameter file
for pqrd in poly_quad_ref_dim:
    for s in solvers:
        name = base_name + '_{6}d_p{0}q{1}r{2}_{3}_{4}_{5}'.format(pqrd[0],pqrd[1],pqrd[2],s[0],s[1],s[2],pqrd[3])
        fname = name + '.prm'
        print '{0}'.format(fname)
        fout = open(fname, 'w')
        fout.write(parameter_file.format(
            base_prm,
            pqrd[0],
            pqrd[1],
            pqrd[2],
            s[0],
            s[1],
            s[2],
            out_dir + name,
            pqrd[3],
            output_points[pqrd[3]],
            dirichlet_id[pqrd[3]],
            dirichlet_mask[pqrd[3]],
            neumann_bc[pqrd[3]],
            'false',
            0.2 if args.likwid else 1.0  # make sure we do 1 fake solution step
        ))
        filenames.append(name)

# two more runs just to illustrate the deformed mesh
if not args.likwid:
  for pqrd in [(2,3,2,2),(2,3,2,3)]:
      for s in [('MF_CG', 'gmg', 'tensor4')]:
          name = '__' + base_name + '_{6}d_p{0}q{1}r{2}_{3}_{4}_{5}'.format(pqrd[0],pqrd[1],pqrd[2],s[0],s[1],s[2],pqrd[3])
          fname = name + '.prm'
          print '{0}'.format(fname)
          fout = open(fname, 'w')
          fout.write(parameter_file.format(
              base_prm,
              pqrd[0],
              pqrd[1],
              pqrd[2],
              s[0],
              s[1],
              s[2],
              name,
              pqrd[3],
              output_points[pqrd[3]],
              dirichlet_id[pqrd[3]],
              dirichlet_mask[pqrd[3]],
              neumann_bc[pqrd[3]],
              'true',
              1.0
          ))


# write bash script
shell_script = 'run.sh'
if args.likwid:
  shell_script = 'likwid_' + shell_script

fout = open (shell_script, 'w')
for f in filenames:
    fout.write(mpicmd.format(f, out_dir))
