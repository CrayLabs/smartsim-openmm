import os, random, json, shutil, time
import argparse 
import numpy as np 
from glob import glob
import MDAnalysis as mda
from utils import outliers_from_latent
from utils import find_frame, write_pdb_frame, make_dir_p 
from  MDAnalysis.analysis.rms import RMSD

from smartredis import Client, Dataset
import smartredis

DEBUG = 0

# Inputs 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--md", help="Input: MD simulation directory")
parser.add_argument("-p", "--pdb", help="Input: pdb file") 
parser.add_argument("-r", "--ref", default=None, help="Input: Reference pdb for RMSD")

args = parser.parse_args()

# Pdb file for MDAnalysis 
pdb_file = os.path.abspath(args.pdb) 
ref_pdb_file = os.path.abspath(args.ref) 

# Separate incoming ml from md workers
incoming_entities = os.getenv("SSKEYIN")
md_workers = []
ml_workers = []
md_timestamps = {}
for key in incoming_entities.split(":"):
    if key.startswith("openmm"):
        md_workers.append(key)
        md_timestamps[key] = 0.0
    if key.startswith("cvae"):
        ml_workers.append(key)

num_md_workers = len(md_workers)
num_ml_workers = len(ml_workers)


client = Client(None, bool(int(os.getenv("SS_CLUSTER", False))))
client.use_tensor_ensemble_prefix(False)

md_updated = True

eps_record = {} 

while True:

    for md_worker in md_workers:
        md_dataset = client.get_dataset(md_worker)
        timestamp = float(md_dataset.get_meta_strings('timestamps')[-1])
        if timestamp > md_timestamps[md_worker]:
            md_updated = True
            md_timestamps[md_worker] = timestamp

    if not md_updated:
        time.sleep(30)
        continue

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
        for (prefix, latent_dim) in zip(prefixes, latent_dims):
            loss = client.get_tensor(prefix+"_loss")[-1]
            if best_loss is None or best_loss > loss:
                best_worker_id = ml_worker
                best_loss = loss
                best_prefix = prefix
                model_dim = latent_dim

    if best_worker_id is None:
        print("Error: no ID found")
    else:
        print(f"Using model {best_prefix} with loss {best_loss}, hyper_dim: {model_dim}")
        
    # Outlier search 
    outlier_list = [] 

    ## eps records for next iteration 
    # eps_record_filepath = './eps_record.json' 
    # if os.path.exists(eps_record_filepath): 
    #     eps_file = open(eps_record_filepath, 'r')
    #     eps_record = json.load(eps_file) 
    #     eps_file.close() 
    # else: 
    #     eps_record = {} 


    # Find the trajectories and contact maps 
    traj_file_list = []  # sorted(glob(os.path.join(args.md, 'omm_runs_*/*.dcd'))) 
    checkpnt_list = []  # sorted(glob(os.path.join(args.md, 'omm_runs_*/checkpnt.chk'))) 

    cm_predict = np.empty(shape=(0, model_dim), dtype=np.float32)

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
        loc_predict = client.get_tensor("{"+md_worker+"}.latent")
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
        eps = eps_record[best_prefix]
    else:
        eps = 0.2

    # Search the right eps for DBSCAN 
    while True: 
        outliers = np.squeeze(outliers_from_latent(cm_predict, eps=eps)) 
        n_outlier = len(outliers) 
        print('dimension = {0}, eps = {1:.2f}, number of outliers found: {2}'.format(
            model_dim, eps, n_outlier))
        if n_outlier > 150: 
            eps = eps + 0.05 
        else: 
            eps_record[best_prefix] = eps 
            outlier_list.append(outliers) 
            break 

    ## Unique outliers 
    outlier_list_uni, outlier_count = np.unique(np.hstack(outlier_list), return_counts=True) 
    ## Save the eps for next iteration 
    # with open(eps_record_filepath, 'w') as eps_file: 
    #         json.dump(eps_record, eps_file) 

    if DEBUG: 
        print (outlier_list_uni)
        

    # Write the outliers using MDAnalysis 
    outliers_pdb_path = os.path.abspath('./outlier_pdbs') 
    make_dir_p(outliers_pdb_path) 
    if DEBUG:
        print ('Writing outliers in %s' % outliers_pdb_path)

    new_outliers_list = [] 
    for outlier in outlier_list_uni: 
        traj_file, num_frame = find_frame(traj_dict, outlier)  
        outlier_pdb_file = os.path.join(outliers_pdb_path, '{}_{:06d}.pdb'.format(os.path.basename(os.path.dirname(traj_file)), num_frame)) 
        # Only write new pdbs to reduce redundancy. 
        if not os.path.exists(outlier_pdb_file): 
            print ('Found a new outlier# {} at frame {} of {}'.format(outlier,
                num_frame, traj_file))
            outlier_pdb = write_pdb_frame(traj_file, pdb_file, num_frame, outlier_pdb_file)  
            print ('     Written as {}'.format(outlier_pdb_file))
        new_outliers_list.append(outlier_pdb_file) 

    # Clean up outdated outliers 
    outliers_list = glob(os.path.join(outliers_pdb_path, 'omm_runs*.pdb')) 
    for outlier in outliers_list: 
        if outlier not in new_outliers_list: 
            print (f'Old outlier {os.path.basename(outlier)} is now connected to a cluster and removing it' \
            + ' from the outlier list ')
            os.rename(outlier, os.path.join(os.path.dirname(outlier), '_'+os.path.basename(outlier))) 


    # Set up input configurations for next batch of MD simulations 
    ## Restarts from pdb
    # used_pdbs = glob(os.path.join(args.md, 'omm_runs_*/omm_runs_*.pdb'))
    # used_pdbs_basenames = [os.path.basename(used_pdb) for used_pdb in used_pdbs ]
    used_files = client.get_dataset('used_files')

    used_pdbs_basenames = used_files.get_meta_strings('pdbs')
    outliers_list = glob(os.path.join(outliers_pdb_path, 'omm_runs*.pdb'))
    restart_pdbs = [outlier for outlier in outliers_list if os.path.basename(outlier) not in used_pdbs_basenames] 

    ## Restarts from check point 
    #used_checkpnts = glob(os.path.join(args.md, 'omm_runs_*/omm_runs_*.chk'))
    used_checkpnts = used_files.get_meta_strings('checkpoints')
    restart_checkpnts = [] 
    for checkpnt in checkpnt_list: 
        checkpnt_filepath = os.path.join(outliers_pdb_path, os.path.basename(os.path.dirname(checkpnt) + '.chk'))
        if not os.path.basename(checkpnt_filepath) in used_checkpnts and not os.path.exists(checkpnt_filepath): 
            shutil.copy2(checkpnt, checkpnt_filepath) 
            if DEBUG:
                print ([os.path.basename(os.path.dirname(checkpnt)) in outlier for
                    outlier in outliers_list])
            if any(os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outliers_list):  
                restart_checkpnts.append(checkpnt_filepath) 

    if DEBUG: 
        print (restart_checkpnts)
        print (restart_pdbs)

    # rank the restart_pdbs according to their RMSD to local state 
    if ref_pdb_file: 
        outlier_traj = mda.Universe(restart_pdbs[0], restart_pdbs) 
        ref_traj = mda.Universe(ref_pdb_file) 
        R = RMSD(outlier_traj, ref_traj, select='protein and name CA') 
        R.run()    
        # Make a dict contains outliers and their RMSD
        restart_pdbs = [pdb for _, pdb in sorted(zip(R.rmsd[:,2], restart_pdbs))] 
        print((np.min(R.rmsd[:,2]), np.max(R.rmsd[:,2]), np.mean(R.rmsd[:,2])), flush=True)
    else: 
        random.shuffle(restart_pdbs) 


    # Write record for next step 
    restart_points = restart_checkpnts + restart_pdbs
    if DEBUG:
        print (restart_points)

    # restart_points_filepath = os.path.abspath('./restart_points.json') 
    # with open(restart_points_filepath, 'w') as restart_file:
    #     if len(restart_points) > 0:
    #         json.dump(restart_points, restart_file)
    #     else:
    #         restart_file.write("[]")

    if client.key_exists('outliers'):
        client.delete_dataset('outliers')

    outlier_dataset = Dataset('outliers')
    for point in restart_points:
        outlier_dataset.add_meta_string('points', point)

    client.put_dataset(outlier_dataset)

    md_updated = False
    time.sleep(60)