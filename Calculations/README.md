
# Summary of runs on Emmy RRZE

## general benchmark
```
python pre_process.py --dir=Emmy_RRZE
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

# Post-processing / plotting

## general benchmark
```
python post_process.py
```

## LIKWID

### Roofline
```
python post_process_likwid.py --dim=2
python post_process_likwid.py --dim=3
python post_process_likwid.py --dim=2 --breakdown
python post_process_likwid.py --dim=3 --breakdown
```

### Speedup (SIMD/MPI)

```
python post_process_likwid_simd.py --dim=2
python post_process_likwid_simd.py --dim=3
```
