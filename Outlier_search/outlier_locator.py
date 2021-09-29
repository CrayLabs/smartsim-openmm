import os, random, json, shutil, time
from smartsim_utils import get_text_stream
import argparse 
import numpy as np 
from glob import glob
import MDAnalysis as mda
from numpy.lib.arraysetops import unique
from utils import outliers_from_latent, write_pdb_frame_to_db
from utils import find_frame, write_pdb_frame, make_dir_p 
from  MDAnalysis.analysis.rms import RMSD
from MDAnalysis.lib.util import NamedStream

from smartredis import Client, Dataset
from smartredis.error import RedisReplyError
import smartredis

DEBUG = 0

max_outliers = 150  # original: 150
base_eps = 0.2  # original: 0.2

# Set to True to keep .dcd and .chk as real files
# Set to False to store .dcd and .chk on DB (experimental)
binary_files = False

# Print info about used PDBs
print_used_pdb_info = True

# Inputs 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--md", help="Input: MD simulation directory")
parser.add_argument("-p", "--pdb", help="Input: pdb file") 
parser.add_argument("-r", "--ref", default=None, help="Input: Reference pdb for RMSD")
parser.add_argument("--gpus_per_node", default=1, type=int)
parser.add_argument("--len_initial", type=int, default=10)
parser.add_argument("--len_iter", type=int, default=10)

args = parser.parse_args()
LEN_initial = args.len_initial
LEN_iter = args.len_iter

# Pdb file for MDAnalysis 
pdb_file = os.path.abspath(args.pdb)
ref_pdb_file = os.path.abspath(args.ref)

# Separate incoming ml from md workers
incoming_entities = os.getenv("SSKEYIN")
md_workers = []
ml_workers = []
md_timestamps = {}
md_iters = {}
for key in incoming_entities.split(":"):
    if key.startswith("openmm"):
        md_workers.append(key)
        md_timestamps[key] = 0.0
        md_iters[key] = 0
    if key.startswith("cvae"):
        ml_workers.append(key)

num_md_workers = len(md_workers)
num_ml_workers = len(ml_workers)


client = Client(None, bool(int(os.getenv("SS_CLUSTER", False))))
client.use_tensor_ensemble_prefix(False)

exp_path = os.path.dirname(os.getcwd())
base_path = os.path.dirname(exp_path)

md_updated = True

eps_record = {} 

used_files = Dataset('used_files')
used_files.add_meta_string('pdbs', pdb_file)
used_files.add_meta_string('checkpoints', '_.chk')  # Fake, just to initialize field
client.put_dataset(used_files)


def update_MD_exe_args(md_workers, outlier_idx):
        
        initial_MD = True

        if client.key_exists('outliers'):
            outliers = client.get_dataset('outliers')
            try:
                outlier_list = outliers.get_meta_strings('points')
                num_outliers = len(outlier_list)
            except:
                outlier_list = []
                num_outliers = 0
        else:
            num_outliers = 0
        
        initial_MD = num_outliers == 0

        
        for (i, omm) in enumerate(md_workers):
            if not initial_MD and outlier_idx < num_outliers:
                outlier = outlier_list[outlier_idx]

            input_dataset_key = omm + "_input"
            if client.key_exists(input_dataset_key):
                continue
            
            input_dataset = Dataset(input_dataset_key)

            exe_args = []
            md_iters[omm] += 1
            exe_args.extend(["--output_path",
                            os.path.join(exp_path,"omm_out",f"omm_runs_{i:02d}_{md_iters[omm]:06d}"),
                            "-g", str(i%args.gpus_per_node)])

            # pick initial point of simulation 
            if initial_MD or outlier_idx >= len(outlier_list): 
                exe_args.extend(['--pdb_file', f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb'])

            elif outlier.endswith('pdb'): 
                exe_args.extend(['--pdb_file', outlier])

                used_files = client.get_dataset('used_files')
                used_files.add_meta_string('pdbs', outlier)
                client.put_dataset(used_files)

                outlier_idx += 1

            elif outlier.endswith('chk'): 
                exe_args.extend(['--pdb_file',
                                f'{base_path}/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb',
                                '-c', outlier] )

                used_files = client.get_dataset('used_files')
                used_files.add_meta_string('checkpoints', os.path.basename(outlier))
                client.put_dataset(used_files)

                outlier_idx += 1

            # how long to run the simulation 
            if initial_MD: 
                exe_args.extend(['--length', str(LEN_initial)])
            else: 
                exe_args.extend(['--length', str(LEN_iter)])

            for exe_arg in exe_args:
                input_dataset.add_meta_string("args", exe_arg)
            
            client.put_dataset(input_dataset)
            if DEBUG:
                print("Updated " + input_dataset_key, flush=True)


# Index of outlier in outliers_list
# to avoid re-using the same outlier
# if the list was not updated
# reset to 0 every time outliers_list
# is updated
outlier_idx = 0

while True:

    # Check if any MD worker has put new data on DB
    for md_worker in md_workers:
        md_dataset = client.get_dataset(md_worker)
        timestamp = float(md_dataset.get_meta_strings('timestamps')[-1])
        if timestamp > md_timestamps[md_worker]:
            md_updated = True
            md_timestamps[md_worker] = timestamp

    if not md_updated:
        update_MD_exe_args(md_workers, outlier_idx)
        time.sleep(30)
        continue
    

    # Find best ML model
    best_worker_id = None
    best_loss = None
    best_prefix = None
    model_dim = None
    for ml_worker in ml_workers:
        if not client.key_exists(ml_worker):
            continue
        dataset = client.get_dataset(ml_worker)
        prefixes = dataset.get_meta_strings("prefixes")
        latent_dims = dataset.get_meta_scalars("latent_dims").astype(np.int64)

        # Experimental feature, use only last five model generations
        if len(latent_dims) > 5:
            prefixes=prefixes[-5:]
            latent_dims = latent_dims[-5:]
        for (prefix, latent_dim) in zip(prefixes, latent_dims):
            # We don't update prefix lists, thus we must check if model still exists
            if not client.key_exists(prefix+"_loss"):
                continue
            loss = client.get_tensor(prefix+"_loss")[-1]
            if best_loss is None or best_loss > loss:
                best_worker_id = ml_worker
                best_loss = loss
                best_prefix = prefix
                model_dim = latent_dim

    if best_worker_id is None:
        print("Error: no ID found")
        continue
    else:
        print(f"Using model {best_prefix} with loss {best_loss}, hyper_dim: {model_dim}")
        
    # Outlier search 
    outlier_list = [] 

    # Find the trajectories and contact maps 
    
    cm_predict = np.empty(shape=(0, model_dim), dtype=np.float32)

    # Get latent space wrt best model for all MD frames
    for idx, md_worker in enumerate(md_workers):
        md_worker_prefix = "{"+md_worker+"}."
        latent_name = md_worker_prefix + "latent"
        if client.tensor_exists(latent_name):
            client.delete_tensor(latent_name)
            client.delete_tensor(latent_name+"_mean")
            client.delete_tensor(latent_name+"_var")
        if not client.tensor_exists(md_worker_prefix + "preproc"):
            continue
        client.run_model(best_prefix+"_encoder", [md_worker_prefix + "preproc"],
                        [latent_name+"_mean", latent_name+"_var", latent_name])
        loc_predict = client.get_tensor("{"+md_worker+"}.latent_mean")
        omm_dataset = client.get_dataset(md_worker)
        loc_lengths = omm_dataset.get_meta_scalars("cm_lengths").astype(np.int64)
        loc_paths = omm_dataset.get_meta_strings("paths")
        # Consistency check
        min_length = min(len(loc_lengths), len(loc_paths))
        loc_lengths = loc_lengths[:min_length]
        loc_paths = loc_paths[:min_length]
        total_traj_length = np.sum(np.asarray(loc_lengths))
        loc_predict = loc_predict[:total_traj_length,:]
        if idx == 0:
            train_data_length = loc_lengths
            cm_paths = loc_paths
            cm_predict = loc_predict
        else:
            train_data_length = np.concatenate((train_data_length, loc_lengths), axis=0)
            cm_paths = np.concatenate((cm_paths, loc_paths), axis=0)
            cm_predict = np.vstack((cm_predict, loc_predict))

    traj_file_list = [os.path.join(path, 'output.dcd') for path in cm_paths]
    checkpnt_list = [os.path.join(path, 'checkpnt.chk') for path in cm_paths]
    if DEBUG:
        print(traj_file_list)
    traj_dict = dict(zip(traj_file_list, train_data_length))


    # initialize eps if empty 
    if str(best_prefix) in eps_record.keys():
        eps = eps_record[str(best_prefix)]
    else:
        eps = base_eps

    # Search the right eps for DBSCAN 
    while True: 
        outliers = np.squeeze(outliers_from_latent(cm_predict, eps=eps)) 
        n_outlier = len(outliers) 
        print('dimension = {0}, eps = {1:.2f}, number of outliers found: {2}'.format(
            model_dim, eps, n_outlier))
        if n_outlier > max_outliers: 
            eps = eps + 0.05 
        else: 
            eps_record[best_prefix] = eps 
            outlier_list.append(outliers) 
            break 

    ## Unique outliers 
    outlier_list_uni, outlier_count = np.unique(np.hstack(outlier_list), return_counts=True) 

    if DEBUG: 
        print (outlier_list_uni)
        

    # Write the outliers using MDAnalysis
    outliers_pdb_path = os.path.abspath('./outlier_pdbs') 

    if binary_files:
        make_dir_p(outliers_pdb_path) 
    
    if DEBUG:
        print ('Writing outliers in %s' % outliers_pdb_path)

    new_outliers_list = [] 
    for outlier in outlier_list_uni: 
        traj_file, num_frame = find_frame(traj_dict, outlier)  
        outlier_pdb_file = os.path.join(outliers_pdb_path, '{}_{:06d}.pdb'.format(os.path.basename(os.path.dirname(traj_file)), num_frame)) 
        # Only write new pdbs to reduce redundancy. 
        if not os.path.exists(outlier_pdb_file): 
            if DEBUG:
                print ('Found a new outlier# {} at frame {} of {}'.format(outlier,
                    num_frame, traj_file))
            # outlier_pdb = write_pdb_frame(traj_file, pdb_file, num_frame, outlier_pdb_file)  
            write_pdb_frame_to_db(traj_file, pdb_file, num_frame, outlier_pdb_file, client)
            if DEBUG:
                print ('     Written as {}'.format(outlier_pdb_file))
        new_outliers_list.append(outlier_pdb_file) 

    # Clean up outdated outliers
    # outliers_list = []
    # try:
    #     outliers = client.get_dataset('outliers').get_meta_strings('points')
    #     outliers_list = [outlier for outlier in outliers if outlier.endswith('.pdb')]
    # except RedisReplyError:
    #     print("No outlier from previous runs was found")

    #outliers_list = glob(os.path.join(outliers_pdb_path, 'omm_runs*.pdb')) 
    # for outlier in outliers_list: 
    #     if outlier not in new_outliers_list: 
    #         print (f'Old outlier {os.path.basename(outlier)} is now connected to a cluster and removing it' \
    #         + ' from the outlier list ')
    #         client.rename_dataset(outlier, os.path.join(os.path.dirname(outlier), '_'+os.path.basename(outlier)))
    #         # os.rename(outlier, os.path.join(os.path.dirname(outlier), '_'+os.path.basename(outlier))) 


    # Set up input configurations for next batch of MD simulations 
    ## Restarts from pdb
    # used_pdbs = glob(os.path.join(args.md, 'omm_runs_*/omm_runs_*.pdb'))
    # used_pdbs_basenames = [os.path.basename(used_pdb) for used_pdb in used_pdbs ]
    used_files = client.get_dataset('used_files')

    used_pdbs = used_files.get_meta_strings('pdbs')
    # outliers_list = glob(os.path.join(outliers_pdb_path, 'omm_runs*.pdb'))
    restart_pdbs = [outlier for outlier in new_outliers_list if outlier not in used_pdbs] 

    ## Restarts from NEW check point
    outliers_list = new_outliers_list
    used_checkpnts = used_files.get_meta_strings('checkpoints')
    restart_checkpnts = [] 
    for checkpnt in checkpnt_list: 
        checkpnt_filepath = os.path.join(outliers_pdb_path, os.path.basename(os.path.dirname(checkpnt) + '.chk'))
        if not os.path.basename(checkpnt_filepath) in used_checkpnts and (not binary_files or not os.path.exists(checkpnt_filepath)): 
            if binary_files:
                shutil.copy2(checkpnt, checkpnt_filepath)
            else:
                client.copy_dataset(checkpnt, checkpnt_filepath)
            if DEBUG:
                print ([os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outliers_list])
            if any(os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outliers_list):  
                restart_checkpnts.append(checkpnt_filepath)

    if DEBUG: 
        print ("CHK", restart_checkpnts)
        print ("PDB", restart_pdbs)

    # rank the restart_pdbs according to their RMSD to local state 

    if restart_pdbs:
        if ref_pdb_file: 
            restart_pdb_streams = [NamedStream(get_text_stream(filename, client), filename) for filename in restart_pdbs]
            outlier_traj = mda.Universe(restart_pdb_streams[0], restart_pdb_streams) 
            ref_traj = mda.Universe(ref_pdb_file) 
            R = RMSD(outlier_traj, ref_traj, select='protein and name CA') 
            R.run()    
            rmsd = R.rmsd[:,2]
            # Make a dict contains outliers and their RMSD
            restart_pdbs = [pdb for _, pdb in sorted(zip(rmsd, restart_pdbs))] 
            if DEBUG:
                print(("MIN, MAX, MEAN", np.min(rmsd), np.max(rmsd), np.mean(rmsd)), flush=True)
        else: 
            random.shuffle(restart_pdbs)
    else:
        print("No more restart PDBs. Starting random exploration.")

    if print_used_pdb_info and len(used_pdbs)>1:
        used_pdb_streams = [NamedStream(get_text_stream(filename, client), filename) for filename in used_pdbs[1:]]
        outlier_traj = mda.Universe(used_pdb_streams[0], used_pdb_streams)
        ref_traj = mda.Universe(ref_pdb_file) 
        R = RMSD(outlier_traj, ref_traj, select='protein and name CA') 
        R.run()    
        rmsd = R.rmsd[:,2]
        print(("MIN, MAX, MEAN", np.min(rmsd), np.max(rmsd), np.mean(rmsd)), flush=True)

    # Write record for next step 
    restart_points = restart_checkpnts + restart_pdbs
    if DEBUG:
        print (restart_points)

    if client.key_exists('outliers'):
        client.delete_dataset('outliers')

    if restart_points:
        outlier_dataset = Dataset('outliers')
        for point in restart_points:
            outlier_dataset.add_meta_string('points', point)

        client.put_dataset(outlier_dataset)

    outlier_idx = 0
    update_MD_exe_args(md_workers, outlier_idx)
    md_updated = False
    time.sleep(60)