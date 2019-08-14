import re
import os
import argparse
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
import fileinput
from matplotlib.ticker import MaxNLocator

def remove_creation_date(file_name):
    for line in fileinput.input(file_name, inplace=True):
        if not 'CreationDate' in line:
            print line,


# define command line arguments
parser = argparse.ArgumentParser(
    description='Post-Process timing/memory info and plot figures.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prefix', metavar='prefix', default='LIKWID_CSL_Munich',
                    help='A folder to look for benchmark results')
parser.add_argument('--dim', metavar='dim', default=2, type=int,
                    help='Dimension (2 or 3)')
parser.add_argument('--clockspeed', metavar='clockspeed', default=2.2, type=float,
                    help='CPU clock speed')

args = parser.parse_args()

prefix = args.prefix if args.prefix.startswith('/') else os.path.join(os.getcwd(), args.prefix)

files = []
for root, dirs, files_ in os.walk(prefix):
    for f in files_:
        if f.endswith(".toutput"):
            files.append(os.path.join(root, f))

print 'Gather data from {0}'.format(prefix)
print 'found {0} files'.format(len(files))

pattern = r'[+\-]?(?:[0-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'

# Sections to pick up from
#|                 Metric                 |     Sum    |    Min    |     Max    |    Avg    |  %ile 25  |  %ile 50  |  %ile 75  |
# table:
sections = [
    'DP MFLOP/s STAT',
    'Memory bandwidth [MBytes/s] STAT',
    'Operational intensity STAT',
    'Runtime unhalted [s] STAT',
    'Runtime (RDTSC) [s] STAT', # spoke with Georg Haager, this shall be closest to the walltime.
    'Clock [MHz] STAT'
]

# NODE data:
clock_speed = 2.0  # GHz
# memory bandwidth
# likwid-bench -t load_avx -w S0:1GB:20:1:2
B=120 # 120 GB/s single socket

print 'Peak performance calculations:'
P = clock_speed
print '  Clock speed:  {0}'.format(clock_speed)
P = P * 4                             # 2xFMA per cycle
print '  w FMA:        {0}'.format(P)
P = P * 8                             # vectorization over 8 doubles = 512 bits (AVX), VECTORIZATION_LEVEL=3
print '  w FMA w SIMD: {0}'.format(P)
P = P * 20                            # 20 cores
print '  peak:         {0}'.format(P)

table_names = [
    'Event',
    'Metric'
]

likwid_data = []
regions = []

for f in files:
    # guess parameters from the file name:
    fname = os.path.basename(f)
    strings = fname.split('_')
    dim = int(re.findall(pattern,strings[2])[0])
    p   = int(re.findall(pattern,strings[3])[0])
    q   = int(re.findall(pattern,strings[3])[1])
    regions = [
        'vmult_MF' if 'MF_CG' in f else 'vmult_Trilinos'
    ]

    # skip tensor4_ns for plotting
    if '_tensor4_ns' in fname:
        continue

    label = ''
    color = ''  # use colors consistent with post_process.py
    if 'MF_CG' in fname:
        if '_scalar' in fname:
            label = 'MF scalar'
            color = 'b'
        elif '_tensor2' in fname:
            label = 'MF tensor2'
            color = 'g'
        elif '_tensor4_ns' in fname:
            label = 'MF tensor4 P'
            color = 'm'
        elif '_tensor4' in fname:
            label = 'MF tensor4'
            color = 'c'
    else:
        label = 'Trilinos'
        color = 'r'

    print 'dim={0} p={1} q={2} region={3} label={4} color={5} file={6}'.format(dim,p,q,regions[0],label,color,fname)

    fin = open(f, 'r')

    # store data for each requested section and region here:
    timing = [ [ np.nan for i in range(len(sections))] for j in range(len(regions))]

    found_region = False
    found_table = False
    r_idx = np.nan
    n_cells = np.nan
    for line in fin:
        if 'Number of active cells' in line:
            n_cells = int(re.findall(pattern,line)[0])

        # Check if we found one of the regions:
        if 'Region:' in line:
            found_region = False
            this_line_region = line.split()[1]
            for idx, s in enumerate(regions):
                # we could have regions starting from the same part,
                # so do exact comparison here
                if s == this_line_region:
                    found_region = True
                    r_idx = idx

        # Reset the table if found any table:
        for t in table_names:
            if t in line:
                found_table = False

        # Now check if the table is actually what we need
        if found_region and ('Metric' in line) and ('Sum' in line):
            found_table = True
            print '-- Region {0} {1}'.format(regions[r_idx], r_idx)

        # Only if we are inside the table of interest and region of interest, try to match the line:
        if found_table and found_region:
            columns = [s.strip() for s in line.split('|')]
            if len(columns) > 1:
                for idx, s in enumerate(sections):
                    if s == columns[1]:
                        if ('Runtime' in s) or ('Clock' in s):
                            # Take "Max" (third number)
                            val = float(columns[4])
                        else:
                            # Take "Sum" (first number)
                            val = float(columns[2])
                        print '   {0} {1}'.format(s,val)
                        # make sure we run with clockspeed we use for Roofline:
                        #if 'Clock' in s:
                            # allow 0.1% variation
                            # assert abs(val - clock_speed * 1000) < clock_speed

                        # we should get here only once for each region:
                        assert np.isnan(timing[r_idx][idx])
                        timing[r_idx][idx] = val

    # finish processing the file, put the data
    for r_idx in range(len(regions)):
        t = timing[r_idx]
        tp = tuple((dim,p,q,label,color,t,r_idx,n_cells))
        likwid_data.append(tp)

# now we have lists of tuples ready
# first, sort by label:
likwid_data.sort(key=lambda tup: (tup[3], tup[1]))

#####################
#       PLOT        #
#####################
fig = plt.figure()
ax = plt.subplot()

fs = 10
params = {'legend.fontsize': 9,
          'font.size': fs}
plt.rcParams.update(params)

plt.xscale('log', basex=2)
plt.yscale('log', basey=2)

ax.set_ylim([2**2,20*2**7])
ax.set_xlim([2**(-4),2**5])

xtics = [2**i for i in range(-3,6)]
ytics = [2**i for i in range(2,12)]
ax.set_xticks(xtics)
ax.set_yticks(ytics)

# Roofline model
# p = min (P, b I)
# where I is measured intensity (Flops/byte)
def Roofline(I,P,B):
    return np.array([min(P,B*i) for i in I])

x = np.linspace(2**(-5), 2**6+10, num=2000)
base = np.array([x[0] for i in x])

roofline_style = 'b-'
peak = Roofline(x,P,B)
# see https://github.com/matplotlib/matplotlib/issues/8623#issuecomment-304892552
ax.plot(x,peak, roofline_style, label='_nolegend_')
ax.plot(x,Roofline(x,P,90), roofline_style, label='_nolegend_')

# various ceilings (w/o FMA, w/o FMA and vectorization):
for p_ in [P/2, P/2/8]:
    ax.plot(x,Roofline(x,p_,B), roofline_style, label='_nolegend_')

ax.fill_between(x, base, peak, where=peak>base, interpolate=True, zorder=1, color='aqua', alpha=0.1)

ax.grid(True, which="both",color='grey', linestyle=':', zorder=5)


# map degrees to point labels:
degree_to_points = {
    2:'s',
    4:'o',
    6:'^',
    8:'v'
}

mf_perf = []
tr_perf = []

# Now plot each measured point:
for d in likwid_data:
  if d[0] == args.dim:
    f = d[5][0]
    b = d[5][1]
    x = [f/b]
    y = [f/1000]  # MFLOP->GFLOP
    style = d[4] + degree_to_points[d[1]]
    label = '{0} (p={1})'.format(d[3],d[1])
    ax.plot(x,y, style, label=label)
    if 'MF' in label:
      mf_perf.append(y[0])
    else:
      tr_perf.append(y[0])

ax.set_xlabel('intensity (Flop/byte)')
ax.set_ylabel('performance (GFlop/s)')
ax.set_aspect('equal', adjustable=None) #'datalim')
ax.yaxis.set_major_formatter(mp.ticker.FuncFormatter(lambda x, pos: '{0}'.format(int(round(x))) ))
ax.xaxis.set_major_formatter(mp.ticker.FuncFormatter(lambda x, pos: '1/{0}'.format(int(round(1./x))) if x < 1.0 else '{0}'.format(int(round(x))) ))

ax.text(2**(-4)+0.01, 37, 'B=120 GB/s', rotation=45, fontsize=fs)
ax.text(2**(-4)+0.01, 12, 'B=90 GB/s', rotation=45, fontsize=fs)

x_pos = 9
ax.text(x_pos,1400,'Peak DP', fontsize=fs)
ax.text(x_pos,450,'w/o FMA', fontsize=fs)
ax.text(x_pos,58, 'w/o SIMD', fontsize=fs)

leg = ax.legend(loc='upper left', ncol=1, labelspacing=0.1)

# file location
fig_prefix = os.path.join(os.getcwd(), '../doc/' + os.path.basename(os.path.normpath(prefix)) + '_')

name = 'roofline_{0}d.pdf'

fig_file = fig_prefix + name.format(args.dim)

print 'Saving figure in: {0}'.format(fig_file)

plt.savefig(fig_file, format='pdf', bbox_inches = 'tight')

# Finally report average performance for all MF and Trilinos runs:
print 'Average performance:'
print '  MF:       {0}'.format(np.mean(mf_perf))
print '  Trilinos: {0}'.format(np.mean(tr_perf))
