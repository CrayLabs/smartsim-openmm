import torch

def triu_to_full(cm0):
    num_res = int(torch.ceil(torch.tensor([((len(cm0) * 2) ** 0.5)])).item())
    iu1 = torch.triu_indices(num_res, num_res, 1)

    cm_full = torch.zeros((num_res, num_res), dtype=cm0.dtype, device=cm0.device)
    cm_full[iu1[0,:],iu1[1,:]] = cm0
    cm_full[iu1[1,:],iu1[0,:]] = cm0
    cm_full.fill_diagonal_(1.)
    return cm_full


def cm_to_cvae(cm_data_lists): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae
    """
    cm_data_lists = [torch.from_numpy(cm) for cm in cm_data_lists]
    cm_all = torch.hstack(cm_data_lists)

    # transfer upper triangle to full matrix 
    cm_data_full = torch.stack([triu_to_full(cm_data) for cm_data in cm_all.T]) 

    # padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x%2 == 0 else (0,1) 
    padding_buffer = (0,0)
    for x in cm_data_full.shape[1:]: 
        # Torch padding is in reverse dimension direction
        padding_buffer = pad_f(x) + padding_buffer

    cm_data_full = torch.nn.functional.pad(cm_data_full, padding_buffer, mode='constant')

    # reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input


def stamp_to_time(stamp): 
    import datetime
    return datetime.datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S') 
  
