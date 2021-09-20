#import torch

def triu_to_full(cm0):
    num_res = int(torch.ceil(torch.tensor([((len(cm0) * 2) ** 0.5)])).item())
    iu1 = torch.triu_indices(num_res, num_res, 1)

    cm_full = torch.zeros((num_res, num_res), dtype=cm0.dtype, device=cm0.device)
    cm_full[iu1[0,:],iu1[1,:]] = cm0
    cm_full[iu1[1,:],iu1[0,:]] = cm0
    cm_full.fill_diagonal_(1.)
    return cm_full


def cm_to_cvae(tensor): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae
    """
    cm_all = tensor

    # transfer upper triangle to full matrix 
    cm_data_full = torch.stack([triu_to_full(cm_data) for cm_data in cm_all.T]) 

    padded_size = [dim for dim in cm_data_full.shape]
    
    for dim in range(1, len(cm_data_full.shape)):
        padded_size[dim] += cm_data_full.shape[dim]%2

    padded = torch.zeros(padded_size)
    padded[0:cm_data_full.shape[0], 0:cm_data_full.shape[1], 0:cm_data_full.shape[2]] = cm_data_full

    # reshape matrix to 4d tensor
    cvae_input = padded.reshape(padded.shape + (1,))

    return cvae_input
