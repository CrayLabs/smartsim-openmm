# SmartSim OpenMM

This repo contains a SmartSim version of the DeepDriveMD workflow. 

This branch's version is equivalent to the standard file-based
SmartSimMD pipeline.

## Installation

We suggest to create a fresh conda environment

```bash
conda create --name openmm python=3.7
```

and then to install packages in the following way:

```bash
pip install tensorflow==2.4.2 numpy==1.19.2 cython smartsim sklearn
conda install -c conda-forge openmm
pip install MDAnalysis MDAnalysisTests parmed --force
conda install git-lfs pytables
```

## System-dependent settings

The code contained in this repo is written to be run on a Cray XC-50 system running Slurm as a workload manager. Launcher and system constraints (like the flag used to access GPUs), have to
be adapted for other systems.

## Running the pipeline

From the repo root directory, run

```bash
python smartsim_md.py
```