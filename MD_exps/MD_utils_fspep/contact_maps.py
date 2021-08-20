import os
import tables

import MDAnalysis as mda
from MDAnalysis.analysis import distances


def contact_maps_from_traj(pdb_file, traj_file, contact_cutoff=8.0, savefile=None):
    """
    Get contact map from trajectory.
    """
    
    mda_traj = mda.Universe(pdb_file, traj_file)
    traj_length = len(mda_traj.trajectory) 
    ca = mda_traj.select_atoms('name CA')
    
    if savefile:
        savefile = os.path.abspath(savefile)
        outfile = tables.open_file(savefile, 'w')
        atom = tables.Float64Atom()
        cm_table = outfile.create_earray(outfile.root, 'contact_maps', atom, shape=(traj_length, 0)) 

    contact_matrices = []
    for frame in mda_traj.trajectory:
        cm_matrix = (distances.self_distance_array(ca.positions) < contact_cutoff) * 1.0
        contact_matrices.append(cm_matrix)

    if savefile:
        cm_table.append(contact_matrices)
        outfile.close() 

    return contact_matrices
