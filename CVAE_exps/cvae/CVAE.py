import os, h5py
import numpy as np

from vae_conv_new import conv_variational_autoencoder

import tensorflow as tf
from smartredis import Client


def CVAE(input_shape, latent_dim=3): 
    image_size = input_shape[1:-1]
    channels = input_shape[-1]
    conv_layers = 4
    feature_maps = [64,64,64,64]
    filter_shapes = [(3,3),(3,3),(3,3),(3,3)]
    strides = [(1,1),(2,2),(1,1),(1,1)]
    dense_layers = 1
    dense_neurons = [128]
    dense_dropouts = [0]

    feature_maps = feature_maps[0:conv_layers]
    filter_shapes = filter_shapes[0:conv_layers]
    strides = strides[0:conv_layers]
    autoencoder = conv_variational_autoencoder(image_size,channels,conv_layers,feature_maps,
               filter_shapes,strides,dense_layers,dense_neurons,dense_dropouts,latent_dim)

    return autoencoder

def run_cvae(gpu_id, cm_file, hyper_dim=3, epochs=10): 
    # read contact map from h5 file 
    # cm_h5 = h5py.File(cm_file, 'r', libver='latest', swmr=True)
    # cm_data_input = cm_h5[u'contact_maps'] 

    client = Client(None, False)
    batches = None
    for i  in range(2):
        key = f"preproc_{i}"
        if client.key_exists(key):
            if batches is None:
                batches = client.get_tensor(key)
            else:
                new_batch = client.get_tensor(key)
                batches = np.concatenate((batches, new_batch), axis=0)
    cm_data_input = batches

    # splitting data into train and validation
    train_val_split = int(0.8 * len(cm_data_input))
    cm_data_train, cm_data_val = cm_data_input[:train_val_split], cm_data_input[train_val_split:] 
    input_shape = cm_data_train.shape
    #cm_h5.close()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id) 
    
    cvae = CVAE(input_shape, hyper_dim) 
    
    cvae.train(cm_data_train, validation_data=cm_data_val, batch_size =
               input_shape[0]//100, epochs=epochs)
    
    return cvae 
