import os
import numpy as np
import time

from vae_conv_new import conv_variational_autoencoder

import tensorflow as tf
from smartredis import Client
from smartredis.client import RedisReplyError

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

def run_cvae(gpu_id, hyper_dim=3, epochs=10, cm_data_input=None): 
    
    client = Client(None, bool(int(os.getenv("SS_CLUSTER", False))))
    client.use_tensor_ensemble_prefix(False)
    batches = None
    print("Starting cvae training.")

    print("Train dataset size: ", cm_data_input.shape)

    # splitting data into train and validation
    train_val_split = int(0.8 * len(cm_data_input))
    cm_data_train, cm_data_val = cm_data_input[:train_val_split], cm_data_input[train_val_split:] 
    input_shape = cm_data_train.shape

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id) 

    cvae = CVAE(input_shape, hyper_dim) 

    cvae.train(cm_data_train, validation_data=cm_data_val, batch_size =
                input_shape[0]//100, epochs=epochs)

    return cvae
