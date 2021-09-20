import os 
import numpy as np
import errno 
import MDAnalysis as mda 
from tensorflow.keras import backend as K 
from sklearn.cluster import DBSCAN 

def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full
    

def find_frame(traj_dict, frame_number=0): 
    local_frame = frame_number
    for key in sorted(traj_dict.keys()): 
        if local_frame - int(traj_dict[key]) < 0: 
            dir_name = os.path.dirname(key) 
            traj_file = os.path.join(dir_name, 'output.dcd')             
            return traj_file, local_frame
        else: 
            local_frame -= int(traj_dict[key])
    raise Exception('frame %d should not exceed the total number of frames, %d' % (frame_number, sum(np.array(traj_dict.values()).astype(int))))
    
    
def write_pdb_frame(traj_file, pdb_file, frame_number, output_pdb): 
    mda_traj = mda.Universe(pdb_file, traj_file)
    mda_traj.trajectory[frame_number] 
    PDB = mda.Writer(output_pdb)
    PDB.write(mda_traj.atoms)     
    return output_pdb
    

def make_dir_p(path_name): 
    try:
        os.mkdir(path_name)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def outliers_from_latent(cm_predict, eps=0.35): 
    db = DBSCAN(eps=eps, min_samples=10).fit(cm_predict)
    db_label = db.labels_
    outlier_list = np.where(db_label == -1)
    return outlier_list
