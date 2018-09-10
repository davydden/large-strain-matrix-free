import re
import os
import argparse
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
import fileinput

def remove_creation_date(file_name):
    for line in fileinput.input(file_name, inplace=True):
        if not 'CreationDate' in line:
            print line,


# define command line arguments
parser = argparse.ArgumentParser(
    description='Post-Process timing/memory info and plot figures.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('prefix', metavar='prefix', default='Emmy_RRZE', nargs='?',
                    help='A folder to look for benchmark results')
args = parser.parse_args()

prefix = args.prefix if args.prefix.startswith('/') else os.path.join(os.getcwd(), args.prefix)

files = [os.path.join(prefix, k,'timings.txt') for k in os.listdir(prefix) if os.path.isfile(os.path.join(prefix, k,'timings.txt'))]

print 'Gather data from {0}'.format(prefix)
print 'found {0} files'.format(len(files))

pattern = r'[+\-]?(?:[0-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'

# sections in timer output:
sections = [
    'vmult (MF)',
    'vmult (Trilinos)',
    'Linear solver',
    'Assemble linear system',
    'Coarse solve level 0'
]

start_line = 'Total wallclock time elapsed since start'

mf2d_data_scalar = []
mf2d_data_tensor2 = []
mf2d_data_tensor4 = []
mb2d_data = []

mf3d_data_scalar = []
mf3d_data_tensor2 = []
mf3d_data_tensor4 = []
mb3d_data = []


for f in files:
    fin = open(f, 'r')
    ready = False

    # timing = ['-' for i in range(2*len(sections))]
    timing = [np.nan for i in range(len(sections))]

    # reset CG iterations in case AMG did not have enough memory
    cg_iterations = np.nan
    for line in fin:
        if 'dim   =' in line:
            dim = int(re.findall(pattern,line)[0])

        elif 'p     =' in line:
            p = int(re.findall(pattern,line)[0])

        elif 'q     =' in line:
            q = int(re.findall(pattern,line)[0])

        elif 'cells =' in line:
            cells = int(re.findall(pattern,line)[0])

        elif 'dofs  =' in line:
            dofs = int(re.findall(pattern,line)[0])

        elif 'Trilinos memory =' in line:
            tr_memory = float(re.findall(pattern,line)[0])

        elif 'MF cache memory =' in line:
            mf_memory = float(re.findall(pattern,line)[0])

        elif 'Average CG iter =' in line:
            cg_iterations = int(re.findall(pattern,line)[0])

        if start_line in line:
            print 'p={0} q={1} cells={2} dofs={3} tr_memory={4} mf_memory={5} file={6}'.format(p, q, cells, dofs, tr_memory, mf_memory, f)
            ready = True

        if ready:
            for idx, s in enumerate(sections):
                if s in line:
                    nums = re.findall(pattern, "".join(line.rsplit(s)))
                    # do time of a single vmult and the reset -- total time
                    n = int(nums[0]) if 'vmult (' in s else int(1)  # how many times
                    t = float(nums[1]) # total time
                    print '  {0} {1} {2} {3}'.format(s,idx,n,t)
                    timing[idx] = t / n

    # finish processing the file, put the data
    tp = tuple((p, dofs, tr_memory, mf_memory, timing, cg_iterations))
    if 'MF_CG' in f:
        if '_scalar' in f:
            if dim == 2:
                mf2d_data_scalar.append(tp)
            else:
                mf3d_data_scalar.append(tp)
        elif '_tensor2' in f:
            if dim == 2:
                mf2d_data_tensor2.append(tp)
            else:
                mf3d_data_tensor2.append(tp)
        elif '_tensor4' in f:
            if dim == 2:
                mf2d_data_tensor4.append(tp)
            else:
                mf3d_data_tensor4.append(tp)
    else:
        if dim == 2:
            mb2d_data.append(tp)
        else:
            mb3d_data.append(tp)

# now we have lists of tuples ready
# first, sort by degree:
mf2d_data_scalar.sort(key=lambda tup: tup[0])
mf2d_data_tensor2.sort(key=lambda tup: tup[0])
mf2d_data_tensor4.sort(key=lambda tup: tup[0])
mb2d_data.sort(key=lambda tup: tup[0])

mf3d_data_scalar.sort(key=lambda tup: tup[0])
mf3d_data_tensor2.sort(key=lambda tup: tup[0])
mf3d_data_tensor4.sort(key=lambda tup: tup[0])
mb3d_data.sort(key=lambda tup: tup[0])

# now get the data for printing
deg2d = [tup[0] for tup in mf2d_data_scalar]
deg3d = [tup[0] for tup in mf3d_data_scalar]

# vmult time per dof
time2d_tr = [tup[4][1]/tup[1] for tup in mf2d_data_scalar]
time2d_sc = [tup[4][0]/tup[1] for tup in mf2d_data_scalar]
time2d_t2 = [tup[4][0]/tup[1] for tup in mf2d_data_tensor2]
time2d_t4 = [tup[4][0]/tup[1] for tup in mf2d_data_tensor4]

time3d_tr = [tup[4][1]/tup[1] for tup in mf3d_data_scalar]
time3d_sc = [tup[4][0]/tup[1] for tup in mf3d_data_scalar]
time3d_t2 = [tup[4][0]/tup[1] for tup in mf3d_data_tensor2]
time3d_t4 = [tup[4][0]/tup[1] for tup in mf3d_data_tensor4]

# solver time per dof
solver2d_tr = [tup[4][2]/tup[1] for tup in mb2d_data]
solver2d_sc = [tup[4][2]/tup[1] for tup in mf2d_data_scalar]
solver2d_t2 = [tup[4][2]/tup[1] for tup in mf2d_data_tensor2]
solver2d_t4 = [tup[4][2]/tup[1] for tup in mf2d_data_tensor4]
solver2d_t4_coarse = [tup[4][4]/tup[1] for tup in mf2d_data_tensor4]

solver3d_tr = [tup[4][2]/tup[1] for tup in mb3d_data]
solver3d_sc = [tup[4][2]/tup[1] for tup in mf3d_data_scalar]
solver3d_t2 = [tup[4][2]/tup[1] for tup in mf3d_data_tensor2]
solver3d_t4 = [tup[4][2]/tup[1] for tup in mf3d_data_tensor4]
solver3d_t4_coarse = [tup[4][4]/tup[1] for tup in mf3d_data_tensor4]

assembly2d_tr = [tup[4][3]/tup[1] for tup in mb2d_data]
assembly3d_tr = [tup[4][3]/tup[1] for tup in mb3d_data]

# CG iterations
cg2d_tr = [tup[5] for tup in mb2d_data]
cg2d_t4 = [tup[5] for tup in mf2d_data_tensor4]

cg3d_tr = [tup[5] for tup in mb3d_data]
cg3d_t4 = [tup[5] for tup in mf3d_data_tensor4]

# Mb per dof
mem2d_tr = [tup[2]/tup[1] for tup in mf2d_data_scalar]
mem2d_sc = [tup[3]/tup[1] for tup in mf2d_data_scalar]
mem2d_t2 = [tup[3]/tup[1] for tup in mf2d_data_tensor2]
mem2d_t4 = [tup[3]/tup[1] for tup in mf2d_data_tensor4]

mem3d_tr = [tup[2]/tup[1] for tup in mf3d_data_scalar]
mem3d_sc = [tup[3]/tup[1] for tup in mf3d_data_scalar]
mem3d_t2 = [tup[3]/tup[1] for tup in mf3d_data_tensor2]
mem3d_t4 = [tup[3]/tup[1] for tup in mf3d_data_tensor4]

# file location
fig_prefix = os.path.join(os.getcwd(), '../doc/' + os.path.basename(os.path.normpath(prefix)) + '_')

plt.plot(deg2d,time2d_tr, 'rs--', label='Trilinos')
plt.plot(deg2d,time2d_sc, 'bo--', label='MF scalar')
plt.plot(deg2d,time2d_t2, 'g^--', label='MF tensor2')
plt.plot(deg2d,time2d_t4, 'cv--', label='MF tensor4')
plt.xlabel('degree')
plt.ylabel('vmult wall time (s) / DoF')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'timing2d.eps', format='eps')
remove_creation_date(fig_prefix + 'timing2d.eps')

# clear
plt.clf()

plt.plot(deg3d,time3d_tr, 'rs--', label='Trilinos')
plt.plot(deg3d,time3d_sc, 'bo--', label='MF scalar')
plt.plot(deg3d,time3d_t2, 'g^--', label='MF tensor2')
plt.plot(deg3d,time3d_t4, 'cv--', label='MF tensor4')
plt.xlabel('degree')
plt.ylabel('vmult wall time (s) / DoF')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'timing3d.eps', format='eps')
remove_creation_date(fig_prefix + 'timing3d.eps')

# clear
plt.clf()

plt.plot(deg2d,mem2d_tr, 'rs--', label='Trilinos')
plt.plot(deg2d,mem2d_sc, 'bo--', label='MF scalar')
plt.plot(deg2d,mem2d_t2, 'g^--', label='MF tensor2')
plt.plot(deg2d,mem2d_t4, 'cv--', label='MF tensor4')
plt.xlabel('degree')
plt.ylabel('memory (Mb) / DoF')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'memory2d.eps', format='eps')
remove_creation_date(fig_prefix + 'memory2d.eps')

# clear
plt.clf()

plt.plot(deg3d,mem3d_tr, 'rs--', label='Trilinos')
plt.plot(deg3d,mem3d_sc, 'bo--', label='MF scalar')
plt.plot(deg3d,mem3d_t2, 'g^--', label='MF tensor2')
plt.plot(deg3d,mem3d_t4, 'cv--', label='MF tensor4')
plt.xlabel('degree')
plt.ylabel('memory (Mb) / DoF')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'memory3d.eps', format='eps')
remove_creation_date(fig_prefix + 'memory3d.eps')

# clear
plt.clf()

plt.plot(deg2d,solver2d_tr, 'rs--', label='Trilinos Solver')
# plt.plot(deg2d,solver2d_sc, 'bo--', label='MF scalar')
# plt.plot(deg2d,solver2d_t2, 'g^--', label='MF tensor2')
plt.plot(deg2d,solver2d_t4, 'cv--', label='MF Solver')  # tensor4')
plt.plot(deg2d,solver2d_t4_coarse, 'g^--', label='MF Coarse Solver')
plt.xlabel('degree')
plt.ylabel('wall time (s) / DoF')
plt.plot(deg2d,assembly2d_tr, 'mp--', label='Trilinos Assembly')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'solver2d.eps', format='eps')
remove_creation_date(fig_prefix + 'solver2d.eps')

# clear
plt.clf()

plt.plot(deg3d,solver3d_tr, 'rs--', label='Trilinos Solver')
# plt.plot(deg3d,solver3d_sc, 'bo--', label='MF scalar')
# plt.plot(deg3d,solver3d_t2, 'g^--', label='MF tensor2')
plt.plot(deg3d,solver3d_t4, 'cv--', label='MF Solver')
plt.plot(deg3d,solver3d_t4_coarse, 'g^--', label='MF Coarse Solver')
plt.xlabel('degree')
plt.ylabel('wall time (s) / DoF')
plt.plot(deg3d,assembly3d_tr, 'mp--', label='Trilinos Assembly')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'solver3d.eps', format='eps')
remove_creation_date(fig_prefix + 'solver3d.eps')

# clear
plt.clf()

plt.plot(deg2d,cg2d_tr, 'rs--', label='Trilinos')
plt.plot(deg2d,cg2d_t4, 'cv--', label='MF')  # tensor4')
plt.xlabel('degree')
plt.ylabel('average number of CG iterations')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'cg2d.eps', format='eps')
remove_creation_date(fig_prefix + 'cg2d.eps')

# clear
plt.clf()

plt.plot(deg3d,cg3d_tr, 'rs--', label='Trilinos')
plt.plot(deg3d,cg3d_t4, 'cv--', label='MF')  # tensor4')
plt.xlabel('degree')
plt.ylabel('average number of CG iterations')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'cg3d.eps', format='eps')
remove_creation_date(fig_prefix + 'cg3d.eps')
