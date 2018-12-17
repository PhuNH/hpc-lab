#!/bin/bash
export OMP_NUM_THREADS=10
srun likwid-perfctr -C N:0-9 -g FLOPS_DP -m ./dtrmv > dtrmv_10
export OMP_NUM_THREADS=16
srun likwid-perfctr -C N:0-15 -g FLOPS_DP -m ./dtrmv > dtrmv_16
export OMP_NUM_THREADS=32
srun likwid-perfctr -C N:0-31 -g FLOPS_DP -m ./dtrmv > dtrmv_32
export OMP_NUM_THREADS=50
srun likwid-perfctr -C N:0-49 -g FLOPS_DP -m ./dtrmv > dtrmv_50
export OMP_NUM_THREADS=64
srun likwid-perfctr -C N:0-63 -g FLOPS_DP -m ./dtrmv > dtrmv_64
export OMP_NUM_THREADS=100
srun likwid-perfctr -C N:0-99 -g FLOPS_DP -m ./dtrmv > dtrmv_100
export OMP_NUM_THREADS=128
srun likwid-perfctr -C N:0-127 -g FLOPS_DP -m ./dtrmv > dtrmv_128
export OMP_NUM_THREADS=200
srun likwid-perfctr -C N:0-199 -g FLOPS_DP -m ./dtrmv > dtrmv_200
export OMP_NUM_THREADS=256
srun likwid-perfctr -C N:0-255 -g FLOPS_DP -m ./dtrmv > dtrmv_256
