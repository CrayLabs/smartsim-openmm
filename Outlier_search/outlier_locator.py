import os, random, json, shutil 
import argparse 
import numpy as np 
from glob import glob
import MDAnalysis as mda
from utils import read_h5py_file, outliers_from_cvae, cm_to_cvae  
from utils import predict_from_cvae, outliers_from_latent
from utils import find_frame, write_pdb_frame, make_dir_p 
from  MDAnalysis.analysis.rms import RMSD

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
DEBUG = 1 

# Inputs 
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--md", help="Input: MD simulation directory")
# parser.add_argument("-o", help="output: cvae weight file. (Keras cannot load model directly, will check again...)")
parser.add_argument("-c", "--cvae", help="Input: CVAE model directory")
parser.add_argument("-p", "--pdb", help="Input: pdb file") 
parser.add_argument("-r", "--ref", default=None, help="Input: Reference pdb for RMSD") 

args = parser.parse_args()

# Pdb file for MDAnalysis 
pdb_file = os.path.abspath(args.pdb) 
ref_pdb_file = os.path.abspath(args.ref) 

# Find the trajectories and contact maps 
cm_files_list = sorted(glob(os.path.join(args.md, 'omm_runs_*/*_cm.h5')))
traj_file_list = sorted(glob(os.path.join(args.md, 'omm_runs_*/*.dcd'))) 
checkpnt_list = sorted(glob(os.path.join(args.md, 'omm_runs_*/checkpnt.chk'))) 

if cm_files_list == []: 
    raise IOError("No h5/traj file found, recheck your input filepath") 

# Find all the trained model weights 
model_weights = sorted(glob(os.path.join(args.cvae, 'cvae_runs_*/cvae_weight.h5'))) 
# model_losses = sorted(glob(os.path.join(args.cvae, 'cvae_runs_*/loss.npy')))
# model_final_loss = [np.load(model_loss)[-1] for model_loss in model_losses] 

# identify the latest models with lowest loss 
model_best = model_weights[0] 
loss_model_best = np.load(os.path.join(os.path.dirname(model_best), 'loss.npy'))[-1] 
for i in range(len(model_weights)):  
    if i + 1 < len(model_weights): 
        if int(os.path.basename(os.path.dirname(model_weights[i]))[10:12]) != int(os.path.basename(os.path.dirname(model_weights[i+1]))[10:12]):
            loss = np.load(os.path.join(os.path.dirname(model_weights[i]), 'loss.npy'))[-1]  
            if loss < loss_model_best: 
                model_best, loss_model_best = model_weights[i], loss 
    else: 
        loss = np.load(os.path.join(os.path.dirname(model_weights[i]), 'loss.npy'))[-1] 
        if loss < loss_model_best:
            model_best, loss_model_best = model_weights[i], loss

print ("Using model {} with loss {}".format(model_best, loss_model_best))
    
# Convert everything to cvae input 
cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files_list] 
cvae_input = cm_to_cvae(cm_data_lists)

# A record of every trajectory length
train_data_length = [cm_data.shape[1] for cm_data in cm_data_lists]
traj_dict = dict(zip(traj_file_list, train_data_length)) 


# Outlier search 
outlier_list = [] 

## eps records for next iteration 
eps_record_filepath = './eps_record.json' 
if os.path.exists(eps_record_filepath): 
    eps_file = open(eps_record_filepath, 'r')
    eps_record = json.load(eps_file) 
    eps_file.close() 
else: 
    eps_record = {} 

# for model_weight in model_weights: 
# Identify the latent dimensions 
model_dim = int(os.path.basename(os.path.dirname(model_best))[10:12]) 
print ('Model latent dimension: %d' % model_dim )
# Get the predicted embeddings 
cm_predict = predict_from_cvae(model_best, cvae_input, hyper_dim=model_dim) 
# initialize eps if empty 
if str(model_best) in eps_record.keys(): 
    eps = eps_record[model_best] 
else: 
    eps = 0.2 

# Search the right eps for DBSCAN 
while True: 
    outliers = np.squeeze(outliers_from_latent(cm_predict, eps=eps)) 
    n_outlier = len(outliers) 
    print('dimension = {0}, eps = {1:.2f}, number of outlier found: {2}'.format(
        model_dim, eps, n_outlier))
    if n_outlier > 150: 
        eps = eps + 0.05 
    else: 
        eps_record[model_best] = eps 
        outlier_list.append(outliers) 
        break 

## Unique outliers 
outlier_list_uni, outlier_count = np.unique(np.hstack(outlier_list), return_counts=True) 
## Save the eps for next iteration 
with open(eps_record_filepath, 'w') as eps_file: 
        json.dump(eps_record, eps_file) 

if DEBUG: 
    print (outlier_list_uni)
    

# Write the outliers using MDAnalysis 
outliers_pdb_path = os.path.abspath('./outlier_pdbs') 
make_dir_p(outliers_pdb_path) 
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
        print ('Old outlier {} is now connected to a cluster and removing it' \
        + ' from the outlier list '.format(os.path.basename(outlier)))
        os.rename(outlier, os.path.join(os.path.dirname(outlier), '_'+os.path.basename(outlier))) 


# Set up input configurations for next batch of MD simulations 
## Restarts from pdb
used_pdbs = glob(os.path.join(args.md, 'omm_runs_*/omm_runs_*.pdb'))
used_pdbs_basenames = [os.path.basename(used_pdb) for used_pdb in used_pdbs ]
outliers_list = glob(os.path.join(outliers_pdb_path, 'omm_runs*.pdb'))
restart_pdbs = [outlier for outlier in outliers_list if os.path.basename(outlier) not in used_pdbs_basenames] 

## Restarts from check point 
used_checkpnts = glob(os.path.join(args.md, 'omm_runs_*/omm_runs_*.chk')) 
restart_checkpnts = [] 
for checkpnt in checkpnt_list: 
    checkpnt_filepath = os.path.join(outliers_pdb_path, os.path.basename(os.path.dirname(checkpnt) + '.chk'))
    if not os.path.exists(checkpnt_filepath): 
        shutil.copy2(checkpnt, checkpnt_filepath) 
        print ([os.path.basename(os.path.dirname(checkpnt)) in outlier for
            outlier in outliers_list])
        if any(os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outliers_list):  
            restart_checkpnts.append(checkpnt_filepath) 

if DEBUG: 
    print (restart_checkpnts)


if DEBUG: 
    print (restart_pdbs)

# rank the restart_pdbs according to their RMSD to local state 
if ref_pdb_file: 
    outlier_traj = mda.Universe(restart_pdbs[0], restart_pdbs) 
    ref_traj = mda.Universe(ref_pdb_file) 
    R = RMSD(outlier_traj, ref_traj, select='protein and name CA') 
    R.run()    
    # Make a dict contains outliers and their RMSD
    # outlier_pdb_RMSD = dict(zip(restart_pdbs, R.rmsd[:,2]))
    restart_pdbs = [pdb for _, pdb in sorted(zip(R.rmsd[:,2], restart_pdbs))] 
else: 
    random.shuffle(restart_pdbs) 


# Write record for next step 
restart_points = restart_checkpnts + restart_pdbs
print (restart_points)

restart_points_filepath = os.path.abspath('./restart_points.json') 
with open(restart_points_filepath, 'w') as restart_file: 
    json.dump(restart_points, restart_file) 


