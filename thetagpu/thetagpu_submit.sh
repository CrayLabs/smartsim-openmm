#!/bin/bash
#COBALT -t 120
#COBALT -n 8
#COBALT -q full-node 
#COBALT -A datascience


. /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh
conda activate openmm-gpu37

export LD_LIBRARY_PATH=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/lib:/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/ucx-1.9.0rc7/lib:$LD_LIBRARY_PATH
export PATH=/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/bin/:/lus/theta-fs0/software/thetagpu/openmpi-4.0.5/ucx-1.9.0rc7/bin:$PATH

cd /lus/theta-fs0/projects/datascience/arigazzi/smartsim-dev/smartsim-openmm
python smartsim_md_thetagpu.py
