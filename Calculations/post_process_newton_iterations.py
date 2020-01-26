import re
import os
import argparse
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
import fileinput
import matplotlib.ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

# define command line arguments
parser = argparse.ArgumentParser(
    description='Post-Process average number of newton iterations.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('prefix', metavar='prefix', default='IWR_newest_patched', nargs='?',
                    help='A folder to look for benchmark results')
args = parser.parse_args()

prefix = args.prefix if args.prefix.startswith('/') else os.path.join(os.getcwd(), args.prefix)

files = [os.path.join(prefix, k,'output') for k in os.listdir(prefix) if os.path.isfile(os.path.join(prefix, k,'output'))]

print 'Gather data from {0}'.format(prefix)
print 'found {0} files'.format(len(files))

pattern = r'[+\-]?(?:[0-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'

newton2d = []
newton3d = []

for f in files:
    # Newton iterations should be same everywhere, take one flavour
    if 'gmg_tensor4' not in f:
        continue
    if 'gmg_tensor4_ns' in f:
        continue

    # guess parameters from directory name
    dirname = f.split('/')[-2]
    strings = dirname.split('_')
    dim = int(re.findall(pattern,strings[1])[0])
    p   = int(re.findall(pattern,strings[2])[0])
    q   = int(re.findall(pattern,strings[2])[1])

    print 'dim={0} p={1} q={2} dir={3}'.format(dim,p,q,dirname)

    # count newton iterations
    fin = open(f, 'r')

    newtonIterations = 0
    counter = 0
    for line in fin:
        # Converged in 4 Newton iterations
        if 'Converged in' in line:
            thisIt = int(re.findall(pattern,line)[0])
            counter = counter + 1
            newtonIterations = newtonIterations + thisIt

    averageNewtonIterations = float(newtonIterations)/counter
    # print(averageNewtonIterations)

    tp = tuple((p, q, averageNewtonIterations))
    if dim == 2:
        newton2d.append(tp)
    else:
        newton3d.append(tp)

# now we have lists of tuples ready
# first, sort by degree:
newton2d.sort(key=lambda tup: tup[0])
newton3d.sort(key=lambda tup: tup[0])

print('---')
for data in [(newton2d, '2'), (newton3d, '3')]:
    nIt = [tup[2] for tup in data[0]]
    res = sum(nIt) / len(nIt)
    print('dim={0}  average Newton iterations = {1}'.format(data[1],res))
