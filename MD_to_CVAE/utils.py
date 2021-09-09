import numpy as np
import h5py 

def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full


def read_h5py_file(h5_file): 
    cm_h5 = h5py.File(h5_file, 'r', libver='latest', swmr=True)
    return cm_h5[u'contact_maps'] 


def cm_to_cvae(cm_data_lists): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae
    """
    cm_all = np.hstack(cm_data_lists)

    # transfer upper triangle to full matrix 
    cm_data_full = np.array([triu_to_full(cm_data) for cm_data in cm_all.T]) 

    # padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x%2 == 0 else (0,1) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))

    return cvae_input


def stamp_to_time(stamp): 
    import datetime
    return datetime.datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S') 
  
