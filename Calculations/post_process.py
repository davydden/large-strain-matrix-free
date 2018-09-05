import re
import os
import argparse
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np

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
    'Assemble linear system'
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

        if start_line in line:
            print 'p={0} q={1} cells={2} dofs={3} tr_memory={4} mf_memory={5} file={6}'.format(p, q, cells, dofs, tr_memory, mf_memory, f)
            ready = True

        if ready:
            for idx, s in enumerate(sections):
                if s in line:
                    nums = re.findall(pattern, line)
                    timing[idx] = float(nums[1]) / int(nums[0]) # total / how many times

    # finish processing the file, put the data
    tp = tuple((p, dofs, tr_memory, mf_memory, timing))
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

# time per dof
time2d_tr = [tup[4][1]/tup[1] for tup in mf2d_data_scalar]
time2d_sc = [tup[4][0]/tup[1] for tup in mf2d_data_scalar]
time2d_t2 = [tup[4][0]/tup[1] for tup in mf2d_data_tensor2]
time2d_t4 = [tup[4][0]/tup[1] for tup in mf2d_data_tensor4]

time3d_tr = [tup[4][1]/tup[1] for tup in mf3d_data_scalar]
time3d_sc = [tup[4][0]/tup[1] for tup in mf3d_data_scalar]
time3d_t2 = [tup[4][0]/tup[1] for tup in mf3d_data_tensor2]
time3d_t4 = [tup[4][0]/tup[1] for tup in mf3d_data_tensor4]

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
plt.ylabel('wall time (s) / DoF')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'timing2d.eps', format='eps')

# clear
plt.clf()

plt.plot(deg3d,time3d_tr, 'rs--', label='Trilinos')
plt.plot(deg3d,time3d_sc, 'bo--', label='MF scalar')
plt.plot(deg3d,time3d_t2, 'g^--', label='MF tensor2')
plt.plot(deg3d,time3d_t4, 'cv--', label='MF tensor4')
plt.xlabel('degree')
plt.ylabel('wall time (s) / DoF')
leg = plt.legend(loc='best', ncol=1)
plt.savefig(fig_prefix + 'timing3d.eps', format='eps')

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
