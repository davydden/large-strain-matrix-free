import os
import fileinput
import re
import numpy as np

def remove_creation_date(file_name):
    '''remove creationg data from .eps file in place'''
    for line in fileinput.input(file_name, inplace=True):
        if not 'CreationDate' in line:
            print line,

def collection_toutput_files(prefix):
    '''starting from prefix, collection full path to all files that
    end with .toutput'''
    files = []
    for root, dirs, files_ in os.walk(prefix):
        for f in files_:
            if f.endswith(".toutput"):
                files.append(os.path.join(root, f))

    return files

def collect_timing_files(prefix):
    '''inside prefix get all files prefix/some_dir/timings.txt'''
    return [os.path.join(prefix, k,'timings.txt') for k in os.listdir(prefix) if os.path.isfile(os.path.join(prefix, k,'timings.txt'))]

def get_regex_pattern():
    return r'[+\-]?(?:[0-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'


def parse_timing_file(f):
    '''Parse timing.txt file

    Return:
        a tuple consiting of
        p, dofs, tr_memory, mf_memory, timing, cg_iterations, cores
    '''
    fin = open(f, 'r')
    ready = False
    pattern = get_regex_pattern()

    dividers = '+---------------------------------'
    inside_table = False

    timing = {}
    # reset CG iterations in case AMG did not have enough memory
    cg_iterations = np.nan
    tot_cg_iterations = np.nan
    for line in fin:
        if 'running with' in line:
            cores = int(re.findall(pattern,line)[0])

        elif 'dim   =' in line:
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
        elif 'Total CG iter = ' in line:
            tot_cg_iterations = int(re.findall(pattern,line)[0])

        if ready:
            if dividers in line:
                inside_table = not inside_table
            elif inside_table:
                # main parsring logic is here

                # get the columns, disregard empty first and last
                columns = [s.strip() for s in line.split('|')][1:-1]

                key = columns[0]
                val = []

                for v in columns[1:]:
                    val.append(re.findall(pattern, v)[0])

                timing[key] = val

        if 'Section' in line and 'wall time' in line:
            # print 'dim={0} p={1} q={2} cells={3} dofs={4} tr_memory={5} mf_memory={6} cg_it={7} file={8}'.format(dim, p, q, cells, dofs, tr_memory, mf_memory, cg_iterations, f)
            ready = True

    # return data as a tuple
    return tuple((p, dofs, tr_memory, mf_memory, timing, cg_iterations, tot_cg_iterations, cores))


def parse_likwid_file(filename, last_line = '', debug_output = False):
    '''Parse LIKWID part in terminal output of a benchmark

    Arguments:
    filename -- the path to the file
    last_line -- the last line of non-LIKWID output. Starting from the line after this
        the parser will be active.

    Return:
    parsed data stored as:
        dictionary (region)
        dictionary (table name i.e. Event, Event_Sum, Metric, Metric_Sum or alike
        dictionary of rows. That is, one can get the row as
        result['vmult']['Metric_Sum']['MFLOP/s']
    '''
    result = {}
    fin = open(filename, 'r')

    row_separator = '---------'

    no_last_line = last_line == ""

    found_start = False
    region = ''
    separator_counter = 0
    table_name = ''
    table_ind = 0

    for line in fin:
        line = line.strip()

        # skip empty lines
        if line == "":
            continue

        # if we are not provided with last_line, start when we
        # see first Region: in line:
        if (no_last_line and not found_start):
            if 'Region:' in line:
                found_start = True

        if found_start:
            #
            # Main logic to parse
            #

            # Check if we found one of the regions:
            if 'Region:' in line:
                region = line[8:]
                if debug_output:
                    print '-- Region: {0}'.format(region)
                # regions should be unique
                assert region not in result
                result[region] = {}
                separator_counter = 0
                table_name = ''
                continue

            elif 'Group:' in line:
                # FIXME: read in groups as well?
                continue

            # At this point we SHOULD have only 3 options: we are on one of the separators,
            # inside the header or inside the core of the table.
            elif row_separator in line:
                separator_counter = separator_counter + 1
                # reset the counter if we are at the end of the current table
                if separator_counter == 3:
                    separator_counter = 0
                continue

            # If we are not inside the table, there must be some
            # dummy output in the terminal, skip it
            if separator_counter == 0 and len(line) > 0:
                if debug_output:
                    print '-- Skip line: {0}'.format(line)
                continue

            # Otherwise if we are in LIKWID part, we should have some region always around
            assert region != ''

            # get the columns, disregard empty first and last
            columns = [s.strip() for s in line.split('|')][1:-1]
            if separator_counter == 1:
                # should be reading the header, get the name:
                table_name = columns[0]
                if 'Sum' in line and 'Min' in line and 'Max' in line and 'Avg' in line:
                    table_name = table_name + '_Sum'

                # table names for a given region should be unique
                assert table_name not in result[region]
                result[region][table_name] = {}
                if debug_output:
                    print '   Name  : {0}'.format(table_name)
            else:
                # we should have table name around already
                assert table_name != ''
                # we should have non-empty list of columns
                assert len(columns) > 1
                # index should be either 0 or 1
                assert table_ind == 0 or table_ind == 1
                key = columns[0]
                val = columns[1:]

                # finally put the data
                result[region][table_name][key] = val
                if debug_output:
                    print '      {0}'.format(key)
        else:
            # If we have not found LIKWID part yet
            if not no_last_line and last_line in line:
                if debug_output:
                    print '-- Start parsing from line: {0}'.format(line)
                found_start = True

    return result

