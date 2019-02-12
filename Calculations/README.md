
# Summary of runs on Emmy RRZE

## general benchmark
```
python pre_process.py --likwid --dir=Emmy_RRZE
qsub emmy_benchmark_intel.sh
```

## LIKWID runs
configure code with `-DWITH_LIKWID=TRUE -DLIKWID_DIR=/apps/likwid/4.2.1/`:

### standard (SIMD + MPI)
```
python pre_process.py --likwid
qsub likwid_emmy_benchmark_intel.sh
```

### SIMD only
```
python pre_process.py --likwid --single
qsub likwid_emmy_benchmark_intel.sh
```

### fully serial
```
python pre_process.py --likwid --single --dir=Emmy_RRZE_novec --prefix=/home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build_novec/
qsub likwid_emmy_benchmark_intel.sh
```

### MPI only
```
python pre_process.py --likwid --dir=Emmy_RRZE_novec --prefix=/home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build_novec/
qsub likwid_emmy_benchmark_intel.sh
```

### breakdown
additionally configure with `-DWITH_BREAKDOWN=TRUE`
```
python pre_process.py --likwid --breakdown --prefix=/home/woody/iwtm/iwtm108/deal.ii-mf-elasticity/_build_breakdown/
qsub likwid_emmy_benchmark_intel.sh
```