import re
import os
import argparse
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np

# define command line arguments
parser = argparse.ArgumentParser(description='Post-Process timing/memory info and plot figures.')
parser.add_argument('prefix', metavar='prefix', default='Emmy_RRZE', nargs='?',
                    help='A folder to look for data')
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
]

start_line = 'Total wallclock time elapsed since start'

data_scalar = []
data_tensor2 = []
data_tensor4 = []

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
    if '_scalar' in f:
        data_scalar.append(tp)
    elif '_tensor2' in f:
        data_tensor2.append(tp)
    elif '_tensor4' in f:
        data_tensor4.append(tp)


# now we have lists of tuples ready
# first, sort by degree:
data_scalar.sort(key=lambda tup: tup[0])
data_tensor2.sort(key=lambda tup: tup[0])
data_tensor4.sort(key=lambda tup: tup[0])


# now get the data for printing
deg = [tup[0] for tup in data_scalar]

# time per dof
time_tr = [tup[4][1]/tup[1] for tup in data_scalar]
time_sc = [tup[4][0]/tup[1] for tup in data_scalar]
time_t2 = [tup[4][0]/tup[1] for tup in data_tensor2]
time_t4 = [tup[4][0]/tup[1] for tup in data_tensor4]

# Mb per dof
mem_tr = [tup[2]/tup[1] for tup in data_scalar]
mem_sc = [tup[3]/tup[1] for tup in data_scalar]
mem_t2 = [tup[3]/tup[1] for tup in data_tensor2]
mem_t4 = [tup[3]/tup[1] for tup in data_tensor4]

plt.plot(deg,time_tr, 'rs--', label='Trilinos')
plt.plot(deg,time_sc, 'bo--', label='MF scalar')
plt.plot(deg,time_t2, 'g^--', label='MF tensor2')
plt.plot(deg,time_t4, 'cv--', label='MF tensor4')
plt.xlabel('degree')
plt.ylabel('wall time (s) / DoF')
leg = plt.legend(loc='best', ncol=1)
plt.savefig('timing.eps', format='eps')


# clear
plt.clf()

plt.plot(deg,mem_tr, 'rs--', label='Trilinos')
plt.plot(deg,mem_sc, 'bo--', label='MF scalar')
plt.plot(deg,mem_t2, 'g^--', label='MF tensor2')
plt.plot(deg,mem_t4, 'cv--', label='MF tensor4')
plt.xlabel('degree')
plt.ylabel('memory (Mb) / DoF')
leg = plt.legend(loc='best', ncol=1)
plt.savefig('memory.eps', format='eps')
