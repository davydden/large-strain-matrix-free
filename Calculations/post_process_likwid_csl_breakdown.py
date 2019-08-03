#!/usr/bin/python

import re
import os
import argparse
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
import fileinput
import math
from matplotlib.ticker import MaxNLocator

# own functions
from utilities import *

# define command line arguments
parser = argparse.ArgumentParser(
    description='Post-Process LIKWID output for breakdown analysis and plot figures.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prefix', metavar='prefix', default='LIKWID_CSL_Munich',
                    help='A folder to look for benchmark results')
parser.add_argument('--dim', metavar='dim', default=2, type=int,
                    help='Dimension (2 or 3)')
parser.add_argument('--nmpi', metavar='nmpi', default=20, type=int,
                    help='Number of MPI cores (used in averaging)')

args = parser.parse_args()

n_mpi_proc = args.nmpi
prefix = args.prefix if args.prefix.startswith('/') else os.path.join(os.getcwd(), args.prefix)
files = collection_toutput_files(prefix)
pattern = get_regex_pattern()

print 'Gather data from {0}'.format(prefix)
print 'found {0} files'.format(len(files))

stack_labels = ['Zero vector', 'MPI', 'Quadrature loop', 'Read/Write', 'Sum factorization']
stack_colors = ['b','g','c','m','r']
n_stack = len(stack_labels)

# hard-code polynomial degree to be used in stack plot
ind = [2,4,6] if args.dim == 2 else [2, 4]

# prepare data here
bar_data = [[ np.nan for i in range(len(ind))] for i in range(n_stack)]

for f in files:
    # guess parameters from the file name:
    fname = os.path.basename(f)
    strings = fname.split('_')
    dim = int(re.findall(pattern,strings[2])[0])
    p   = int(re.findall(pattern,strings[3])[0])
    q   = int(re.findall(pattern,strings[3])[1])

    if dim != args.dim:
        continue

    post_process = False
    if '_tensor4' in fname and 'tensor4_ns' not in fname:
        for i in ind:
            if i == p:
                post_process = True

    if not post_process:
        continue

    print 'dim={0} p={1} q={2} file={3}'.format(dim,p,q,fname)

    result = parse_likwid_file(f, last_line='LIKWID_MARKER_CLOSE', debug_output=False)

    #
    # Indirect measurements using stack plot:
    #

    # 1. get AVERAGE fractions of RW, SF and QD parts inside the CellLoop over MPI cores
    # CellLoop == RW + SF + QD
    cell_fraction_sf = 0
    cell_fraction_qd = 0
    cell_fraction_rw = 0
    for col in range(n_mpi_proc):
        time_rw_sf_qd = float(result['vmult_MF_cell']     ['Metric']['Runtime (RDTSC) [s]'][col]) # T = RW + SF + QD
        time_rw       = float(result['vmult_MF_cell_RW']  ['Metric']['Runtime (RDTSC) [s]'][col]) # T = RW
        time_rw_sf    = float(result['vmult_MF_cell_RWSF']['Metric']['Runtime (RDTSC) [s]'][col]) # T = RW + SF

        time_qd   = time_rw_sf_qd - time_rw_sf
        time_sf   = time_rw_sf    - time_rw

        cell_fraction_sf = time_sf/time_rw_sf_qd + cell_fraction_sf
        cell_fraction_qd = time_qd/time_rw_sf_qd + cell_fraction_qd
        cell_fraction_rw = time_rw/time_rw_sf_qd + cell_fraction_rw

    # average over all MPI processes:
    cell_fraction_sf = cell_fraction_sf/n_mpi_proc
    cell_fraction_qd = cell_fraction_qd/n_mpi_proc
    cell_fraction_rw = cell_fraction_rw/n_mpi_proc

    # 2. Now look at timers including imbalance and MPI
    col = 3   # Avg
    # col = 2 # Max  // FIXME: take max?
    time_cell    = float(result['vmult_MF_cell']['Metric_Sum']['Runtime unhalted [s] STAT'][col]) # T = RW + SF + QD
    time_zero    = float(result['vmult_MF_zero']['Metric_Sum']['Runtime unhalted [s] STAT'][col]) # T =                Zero
    time_mpi     = float(result['vmult_MF_mpi'] ['Metric_Sum']['Runtime unhalted [s] STAT'][col]) # T =                       MPI
    time_vmult   = float(result['vmult_MF']     ['Metric_Sum']['Runtime unhalted [s] STAT'][col]) # T = RW + SF + QD + Zero + MPI

    time_sf      = time_cell * cell_fraction_sf
    time_rw      = time_cell * cell_fraction_rw
    time_qd      = time_cell * cell_fraction_qd

    time_ref     = time_cell + time_zero + time_mpi

    fraction_zero = time_zero / time_ref
    fraction_mpi  = time_mpi  / time_ref
    fraction_sf   = time_sf   / time_ref
    fraction_rw   = time_rw   / time_ref
    fraction_qd   = time_qd   / time_ref

    def frac2per(frac):
        return int(math.floor(frac*100))

    print 'Breakdown (Avg) timing from separate measurements:'
    print '      |{0:>20} | {1:>4} | {2:>9}'.format('Time', 'Frac', 'Frac Cell')
    print 'Total: {0:>20}'.format(time_vmult)
    print 'Sum:   {0:>20}   {1:>4}'.format(time_ref, 100)
    print 'SF:    {0:>20}   {1:>4}   {2:>9}'.format(time_sf, frac2per(fraction_sf), frac2per(cell_fraction_sf))
    print 'RW:    {0:>20}   {1:>4}   {2:>9}'.format(time_rw, frac2per(fraction_rw), frac2per(cell_fraction_rw))
    print 'QD:    {0:>20}   {1:>4}   {2:>9}'.format(time_qd, frac2per(fraction_qd), frac2per(cell_fraction_qd))
    print 'Zero:  {0:>20}   {1:>4}'.format(time_zero, frac2per(fraction_zero))
    print 'MPI:   {0:>20}   {1:>4}'.format(time_mpi, frac2per(fraction_mpi))
    print ''

    # Now put stack bar data:
    for idx, s in enumerate(ind):
        if s == p:
            # order consistent with stack_labels
            bar_data[0][idx] = fraction_zero
            bar_data[1][idx] = fraction_mpi
            bar_data[2][idx] = fraction_qd
            bar_data[3][idx] = fraction_rw
            bar_data[4][idx] = fraction_sf


#
# Plot stack bar:
#
print '============ Plot stack bar ============'
print 'Label             2               4'
for d, t in zip(bar_data, stack_labels):
  print t.ljust(18) + '{0}'.format(d[0]).ljust(16) + '{0}'.format(d[1]).ljust(16)
print ''


plt.clf()
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ind_str = ['{0}'.format(s) for s in ind]
ind_plt = range(len(ind))

# setup "bottom" for bar data
bar_data_bottom = [[ 0 for i in range(len(ind))] for j in range(n_stack)]

for j in range(len(ind)):      # for all degrees
  for i in range(1, n_stack):  # for all stack starting from second
    for k in range(i):
      bar_data_bottom[i][j] = bar_data_bottom[i][j] + bar_data[k][j]

print 'Bottom for stack:'
print 'Label             2               4'
for d, t in zip(bar_data_bottom, stack_labels):
  print t.ljust(18) + '{0}'.format(d[0]).ljust(16) + '{0}'.format(d[1]).ljust(16)
print ''

width = 0.5
bars = [i for i in range(n_stack)]
for i in range(n_stack):
  if i==0:
    b = plt.bar(ind_plt, bar_data[i], width, color=stack_colors[i], align='center')
  else:
    b = plt.bar(ind_plt, bar_data[i], width, color=stack_colors[i], align='center', bottom=bar_data_bottom[i])

  bars[i] = b[0]

plt.legend(bars, stack_labels, loc='upper left')

plt.ylabel('fraction of wall time')
plt.xticks(ind_plt, ind_str)
plt.xlabel('polynomial degree')


# Save figure to file:
fig_path = os.path.join(os.getcwd(), '../doc/{0}_breakdown_stackedbar_{1}d.pdf'.format(args.prefix,args.dim))
plt.tight_layout()
plt.savefig(fig_path, format='pdf')

