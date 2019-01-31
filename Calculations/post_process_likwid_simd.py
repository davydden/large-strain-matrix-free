import argparse
import numpy as np

# own functions
from utilities import *

# define command line arguments
parser = argparse.ArgumentParser(
    description='Post-Process timing/memory info and create a table.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prefix', metavar='prefix', default='LIKWID_Emmy_RRZE',
                    help='A folder prefix to look for benchmark results')
parser.add_argument('--dim', metavar='dim', default=2, type=int,
                    help='Dimension (2 or 3)')

args = parser.parse_args()


# serial | MPI | SIMD | MPI+SIMD
suffixes = [
    '_1proc_novec',
    '_novec',
    '_1proc',
    ''
]

prefix = args.prefix if args.prefix.startswith('/') else os.path.join(os.getcwd(), args.prefix)

# We will generate a table similar to Table 1 in Kronbichler 2012
table_data = []

# Go through all the suffixes and parse the data
for idx, s in enumerate(suffixes):
    this_prefix = prefix + s
    files = collection_toutput_files(prefix + s)

    print 'Gather data from {0}'.format(this_prefix)
    print 'found {0} files'.format(len(files))

    # no go through the files:
    for f in files:
        # guess parameters from the file name:
        fname = os.path.basename(f)
        strings = fname.split('_')
        dim = int(re.findall(get_regex_pattern(),strings[2])[0])
        p   = int(re.findall(get_regex_pattern(),strings[3])[0])
        q   = int(re.findall(get_regex_pattern(),strings[3])[1])

        # we are only interested in tensor2 (Algorithm 3) for the currently chosen dimension
        if not (dim == args.dim and '_tensor4' in fname):
            continue

        print 'dim={0} p={1} q={2} file={3}'.format(dim,p,q,fname)

        # these files contain a single region
        result = parse_likwid_file(f,'LIKWID_MARKER_CLOSE')['vmult_MF']

        # depending on the run, the LIKWID output may not have Sum table
        if 'Metric_Sum' not in result:
            data = result['Metric']
            flops = data['MFLOP/s'][0]
            runtime = data['Runtime (RDTSC) [s]'][0]
        else:
            data = result['Metric_Sum']
            flops = data['MFLOP/s STAT'][0]                # take Sum
            runtime = data['Runtime (RDTSC) [s] STAT'][2]  # take Max

        print '  {0}'.format(flops)
        print '  {0}'.format(runtime)



