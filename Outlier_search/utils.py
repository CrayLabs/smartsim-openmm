import io
import os 
import numpy as np
import errno 
import MDAnalysis as mda 
from MDAnalysis.lib.util import NamedStream
from numpy.core.fromnumeric import put
from sklearn.cluster import DBSCAN 
import MDAnalysis.coordinates as MDCoords

from smartsim_utils import get_text_file, get_text_stream, put_strings_as_file, save_binary_file

binary_files = False

def find_frame(traj_dict, frame_number=0): 
    local_frame = frame_number
    for key in sorted(traj_dict.keys()): 
        if local_frame - int(traj_dict[key]) < 0: 
            dir_name = os.path.dirname(key) 
            traj_file = os.path.join(dir_name, 'output.dcd')             
            return traj_file, local_frame
        else: 
            local_frame -= int(traj_dict[key])
    
    total_length = np.sum(np.asarray(traj_dict.values()).astype(int))
    raise Exception('frame %d should not exceed the total number of frames, %d' % (frame_number, total_length))
    
    
def write_pdb_frame(traj_file, pdb_file, frame_number, output_pdb): 
    mda_traj = mda.Universe(pdb_file, traj_file)
    mda_traj.trajectory[frame_number] 
    PDB = mda.Writer(output_pdb)
    PDB.write(mda_traj.atoms)     
    return output_pdb


def write_pdb_frame_to_db(traj_file, pdb_file, frame_number, output_pdb, client):

    pdb_strings = get_text_file(pdb_file, client)
    pdb_stream = NamedStream(io.StringIO("\n".join(pdb_strings), newline="\n"), pdb_file)

    if binary_files:
        mda_traj = mda.Universe(pdb_stream, traj_file)
    else:
        # We cannot use a stream for MDAnalysis (DCDFile does not accept it)
        # We write the file because we need it
        if not os.path.exists(traj_file):
            save_binary_file(traj_file, client)
        mda_traj = mda.Universe(pdb_stream, traj_file)

    mda_traj.trajectory[frame_number]

    output_stream = NamedStream(io.StringIO(), output_pdb)
    PDB = MDCoords.PDB.PDBWriter(output_stream, multiframe=True)
    PDB.write(mda_traj.atoms) 
    PDB.close()
    del PDB
    output_stream.seek(0)
    try:
        put_strings_as_file(filename=output_pdb, strings=output_stream.readlines(), client=client)
    except IOError:
        # IOError means that the file already exists, we don't need to replace it as it would be the same
        return

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
