import simtk.unit as u
import sys, os, shutil 
import time
import argparse 
from smartredis import Client, Dataset
from smartredis.error import RedisReplyError

from MD_utils_fspep.openmm_simulation import openmm_simulate_amber_fs_pep 

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--pdb_file", dest="f", help="pdb file", type=str, default=None)
parser.add_argument("-p", "--topol", dest='p', help="topology file", type=str, default=None)
parser.add_argument("-c", help="check point file to restart simulation", type=str, default=None)
parser.add_argument("-l", "--length", default=10, help="how long (ns) the system will be simulated", type=str)
parser.add_argument("-g", "--gpu", default=0, help="id of gpu to use for the simulation", type=str)
parser.add_argument("--output_path", default=".", type=str)


# check_point = None
stop = False
client = Client(None, bool(int(os.getenv("SS_CLUSTER", False))))
client.use_tensor_ensemble_prefix(False)
keyout = os.getenv("SSKEYOUT")
if keyout is None:
    raise IOError('Could not read SSKEYOUT for Openmm simulation')
else:
    input_dataset_key =  keyout + "_input"

iteration = 0

while not stop:
    if not client.key_exists(input_dataset_key):
        time.sleep(5)
    else:
        try:
            input = client.get_dataset(input_dataset_key)
        except RedisReplyError:
            time.sleep(5)
            continue
        input_args = input.get_meta_strings('args')
        if 'STOP' in input_args:
            stop = True
            continue
        else:
            args = parser.parse_args(input_args)
            print(args)
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

            gpu_index = args.gpu
            
            output_path = args.output_path
            os.makedirs(output_path)

            openmm_simulate_amber_fs_pep(pdb_file,
                                        check_point = check_point,
                                        GPU_index=gpu_index,
                                        output_traj=os.path.join(output_path, "output.dcd"),
                                        output_log=os.path.join(output_path, "output.log"),
                                        report_time=50*u.picoseconds,
                                        sim_time=float(args.length)*u.nanoseconds)
            client.delete_dataset(input_dataset_key)
            print(f"Completed iteration #{iteration}", flush=True)
            iteration += 1
