import simtk.unit as u
import sys, os, shutil 
import argparse 

from MD_utils.openmm_simulation import openmm_simulate_amber_fs_pep 

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--pdb_file", dest="f", help="pdb file")
parser.add_argument("-p", "--topol", dest='p', help="topology file")
parser.add_argument("-c", help="check point file to restart simulation")
parser.add_argument("-l", "--length", default=10, help="how long (ns) the system will be simulated")
parser.add_argument("-g", "--gpu", default=0, help="id of gpu to use for the simulation")
args = parser.parse_args() 

if args.f: 
    pdb_file = os.path.abspath(args.f) 
else: 
    raise IOError("No pdb file assigned...") 

if args.p: 
    top_file = os.path.abspath(args.p) 
else: 
    top_file = None 

if args.c: 
    check_point = os.path.abspath(args.c) 
else: 
    check_point = None 
# pdb_file = os.path.abspath('./pdb/100-fs-peptide-400K.pdb')
# ref_pdb_file = os.path.abspath('./pdb/fs-peptide.pdb')

gpu_index = 0 # os.environ["CUDA_VISIBLE_DEVICES"]

# check_point = None
openmm_simulate_amber_fs_pep(pdb_file,
                             check_point = check_point,
                             GPU_index=gpu_index,
                             output_traj="output.dcd",
                             output_log="output.log",
                             output_cm='output_cm.h5',
                             report_time=50*u.picoseconds,
                             sim_time=float(args.length)*u.nanoseconds)


