#!/bin/bash
#SBATCH -o /home/hpc/t1221/t1221ao/host_name.%j.%N.out
#SBATCH -D /home/hpc/t1221/t1221ao/
#SBATCH -J host_name
#SBATCH --clusters=mpp3
#SBATCH --get-user-env
#SBATCH --nodes=2
#SBATCH --mail-type=end
#SBATCH --mail-user=phu.nguyen@tum.de
#SBATCH --export=NONE
source /etc/profile.d/modules.sh
srun ./host_name
