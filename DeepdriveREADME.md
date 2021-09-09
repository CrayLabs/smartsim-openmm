# DeepDriveMD


## Availability

| System      | HPC Resource    | Status | Available date from |
| ----------- | --------------- | ------ | ------------------- |
| adrp        | longhorn        | Ready  | May 12th, 2020 |
| plpro       | longhorn        | -      | |
| adrp        | lassen          | Ready  | May 28th, 2020 |
| plpro       | lassen          | -      | |
| adrp        | summit          | Ready  | |
| plpro       | summit          | Ready* | May 12th, 2020 |

*plpro is waiting for new implementation of CVAE

## Installation

### Tensorflow/Keras on ORNL Summit

```
(python3)
. "/sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh"
conda create -n deepdrivemd python=3.6 -y
conda activate deepdrivemd
conda install tensorflow-gpu keras scikit-learn swig numpy cython scipy matplotlib pytables h5py -y
pip install MDAnalysis MDAnalysisTests parmed
pip install radical.entk radical.pilot radical.saga radical.utils --upgrade
```

### OpenMM

by source code (for linux ppc64le e.g. Summit)
https://gist.github.com/lee212/4bbfe520c8003fbb91929731b8ea8a1e

by conda (for linux 64 e.g. PSC Bridges)
```
module load anaconda3
module load cuda/9.2
source /opt/packages/anaconda/anaconda3-5.2.0/etc/profile.d/conda.sh
conda install -c omnia-dev/label/cuda92 openmm
```

### Latest stack info

The following versions are verified on Summit

```
$ radical-stack

  python               : 3.6.10
  pythonpath           : /sw/summit/xalt/1.2.1/site:/sw/summit/xalt/1.2.1/libexec
  virtualenv           : deepdrivemd

  radical.entk         : 1.5.8
  radical.gtod         : 1.5.0
  radical.pilot        : 1.5.8
  radical.saga         : 1.5.8
  radical.utils        : 1.5.8

```

### Environment variables

#### RMQ (Mandatory)

```
export RMQ_HOSTNAME=two.radical-project.org; export RMQ_PORT=33239
```

#### Profiling (Optional)

Profiling produces `*.json` and `*.prof` for additional info, if the following variables defined.

```
export RADICAL_PILOT_PROFILE=TRUE; 
export RADICAL_ENTK_PROFILE=TRUE 
```

## Run

It will require writable space before running the main script. Output files are stored in sub-directories. Locate the main repo at GPFS i.e., $MEMBERWORK/{{PROJECTID}}/ and run the script on Summit. $HOME becomes readable only system when a job is running.

```
$ python molecules.py
```

## Prerequisites


Bash environment variable for EnTK
- RMQ_HOSTNAME/RMQ_PORT (mandatory)
- export RADICAL_LOG_TGT=radical.log; export RADICAL_LOG_LVL=DEBUG (optional for extra log messages)
- export RADICAL_PILOT_PROFILE=TRUE; export RADICAL_ENTK_PROFILE=TRUE (optional for profiling)


## Clean up

Intermediate files/folders are generated over the workflow execution. These need to be deleted before the next run.
Currently, manual deletion is required like:
```
rm -rf CVAE_exps/cvae_runs_*
rm -rf MD_exps/VHP_exp/omm_runs_*
rm -rf MD_exps/fs-pep/omm_runs_*
rm -rf Outlier_search/outlier_pdbs/*
rm -rf Outlier_search/eps_record.json
rm -rf Outlier_search/restart_points.json
```

