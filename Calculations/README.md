
# Summary of runs on Emmy RRZE

`pre_process` script generates `run.sh` script which is sourced
from scripts that submit jobs (e.g. `likwid_emmy_benchmark_intel.sh`).
Exceptions are `emmy_benchmark_intel_*` which have hard-coded paths and parameters
to do the run.
Unless executed on RRZE with DD's account, pre-process scripts should get custom
`--prefix` and `--calc` parameters and `cd` in _submit_ scripts should be modified.

## general benchmark
```
python pre_process.py --dir=Emmy_RRZE
qsub emmy_benchmark_intel.sh
```

## weak scaling
```
qsub emmy_benchmark_intel_1n.sh
qsub emmy_benchmark_intel_8n.sh
qsub emmy_benchmark_intel_64n.sh
```

## LIKWID runs
configure code with `-DWITH_LIKWID=TRUE -DLIKWID_DIR=/apps/likwid/4.2.1/`.

### standard (SIMD + MPI)
```
python pre_process.py --likwid
qsub likwid_emmy_benchmark_intel.sh
```
This is used both for Roofline plot and indirect breakdown measurements.

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

# Post-processing / plotting

Current plots are done using: Python 2.7 + Matplotlib 2.2.4 + numpy-1.16.4 + macOS 10.14.6

## general benchmark
```
python post_process.py CSL_Munich
python post_process.py IWR_newest_patched
```

## weak scaling
```
python post_process_weak.py --dim=2
python post_process_weak.py --dim=3
```

## LIKWID

### Roofline
```
python post_process_likwid_csl.py --dim=2
python post_process_likwid_csl.py --dim=3
python post_process_likwid_csl_breakdown.py --dim=2
python post_process_likwid_csl_breakdown.py --dim=3
```

### Speedup (SIMD/MPI)
```
python post_process_likwid_simd.py --dim=2 --alg=tensor2
python post_process_likwid_simd.py --dim=2 --alg=tensor4
python post_process_likwid_simd.py --dim=3 --alg=tensor2
python post_process_likwid_simd.py --dim=3 --alg=tensor4
```
