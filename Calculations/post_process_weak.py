import argparse
import numpy as np

# own functions
from utilities import *

# define command line arguments
parser = argparse.ArgumentParser(
    description='Post-Process timing/memory info and create a table.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prefix', metavar='prefix', default='CSL_Munich_scaling',
                    help='A folder prefix to look for benchmark results')
parser.add_argument('--dim', metavar='dim', default=3, type=int,
                    help='Dimension (2 or 3)')

args = parser.parse_args()

prefix = args.prefix if args.prefix.startswith('/') else os.path.join(os.getcwd(), args.prefix)

pattern = get_regex_pattern()

# Go through all the suffixes and parse the data
files = collect_timing_files(prefix)

print 'Gather data from {0}'.format(prefix)
print 'found {0} files'.format(len(files))

table_data = []
# no go through the files:
for f in files:
    dirname = os.path.dirname(f).split('/')[-1]
    strings = dirname.split('_')
    dim  = int(re.findall(pattern,strings[1])[0])
    gref = int(re.findall(pattern,strings[2])[2])

    if dim != args.dim:
        continue

    parsed_tup = parse_timing_file(f)
    table = parsed_tup[4]
    tot_cg = parsed_tup[6]
    cores = parsed_tup[7]
    dofs = parsed_tup[1]

    vmult = table['vmult (MF)']
    cg = table['Linear solver']

    t = float(vmult[1])/int(vmult[0])
    tcg = float(cg[1])/int(tot_cg)

    print 'Dofs={0} cores={1} gref={4} vmult={2} cg_it={3}'.format(dofs,cores,t, tcg, gref)

    table_data.append(tuple((cores,dofs,gref,t,tcg)))

table_data.sort(key=lambda tup: tup[0])

# # Finally write out a latex multicolumn table
file_name = os.path.join(os.getcwd(), '../doc/weak_scaling_{0}d.tex'.format(args.dim))
print 'Saving table in ' + file_name

with open(file_name, 'w') as f:
    # start with the header
#     f.write("""\
# \\begin{table}
# \centering
# \\begin{tabular}{|c|c|c|c|c|}
# \hline
# cores  & DoFs & $N_{gref}$ & vmult [s] & CG iteration [s] \\\\
# \hline
# """)

    # now print the gathered data:
    for row in table_data:
        line = '{0} & {1:,} & {2} '.format(int(row[0]), int(row[1]), row[2])
        for i in range(3, len(row)):
            line = line + '& \pgfmathprintnumber{' + '{0}'.format(row[i]) + '} '
        line = line + '\\\\\n'
        f.write(line)

#     # now footer:
#     f.write("""\
# \hline
# \end{tabular}
# \caption{Weak scaling of Algorithm \\ref{alg:mf_tensor4} in 3D for quadratic polynomial basis.}
# \label{tab:weak_3d}
# \end{table}
# """)
