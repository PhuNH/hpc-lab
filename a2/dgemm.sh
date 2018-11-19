#!/bin/bash
#SBATCH -o /home/hpc/t1221/t1221ao/hpc-lab/a2/4/dgemm.%j.%N.out
#SBATCH -D /home/hpc/t1221/t1221ao/hpc-lab/a2/4/
#SBATCH -J dgemm
#SBATCH --clusters=mpp3
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --hint=memory_bound
#SBATCH --mail-type=end
#SBATCH --mail-user=phu.nguyen@tum.de
#SBATCH --export=NONE

#cat /proc/meminfo
source /etc/profile
source /etc/profile.d/modules.sh
module unload intel/17.0
module load intel/18.0

export OMP_NUM_THREADS=256
./dgemm 4096 1 32 1
