import os
import argparse 
from cvae.CVAE import run_cvae  
import numpy as np 
import time

from smartredis import Client, Dataset
from smartredis.util import Dtypes

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dim", default=3, help="Number of dimensions in latent space")
parser.add_argument("-gpu", default=0, help="gpu_id")

args = parser.parse_args()

client = Client(None, bool(int(os.getenv("SS_CLUSTER", False))))
client.use_tensor_ensemble_prefix(False)

if args.dim == 'SmartSim':
    hyper_dim = client.get_tensor(os.getenv("SSKEYOUT")+"_dim").astype(int)[0]
else:
    hyper_dim = int(args.dim)

gpu_id = args.gpu



def save_model_to_db(client, model, prefix):
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    input_names = [x.name.split(":")[0] for x in frozen_func.inputs]
    output_names = [x.name.split(":")[0] for x in frozen_func.outputs]

    model_serialized = frozen_func.graph.as_graph_def().SerializeToString(deterministic=True)

    client.set_model("_".join([prefix,model.name]), model=model_serialized, tag="",
                    backend="TF", device="GPU", inputs=input_names, outputs=output_names)




if __name__ == '__main__': 

    generation_id = 0

    if "SSKEYIN_SLURM" in os.environ:
        prefixes = os.getenv("SSKEYIN_SLURM").split(":")
    else:
        prefixes = os.getenv("SSKEYIN").split(",")

    next_batch = {}
    for prefix in prefixes:
        next_batch[prefix] = 0

    batches = None

    while True:

        # Acquire new batches -- assumption: order does not matter
        # for training, thus we can simply get the new ones and append
        # them to the previous ones
        for prefix in prefixes:
            key = "{" + prefix + "}.preproc_" + str(next_batch[prefix])
            print(key)
            attempts = 5
            while client.key_exists(key) and attempts>0:    
                try:
                    if batches is None:
                        batches = client.get_tensor(key)
                    else:
                        new_batch = client.get_tensor(key)
                        batches = np.concatenate((batches, new_batch), axis=0)
                    
                    # If batch exists, go to next one
                    next_batch[prefix] += 1
                    key = "{" + prefix + "}.preproc_" + str(next_batch[prefix])
                    attempts = 5
                    break

                except RedisReplyError:
                    time.sleep(5)
                    attempts -= 1            
                    if attempts == 0:
                        print(f"{key} exists but can not be accessed, proceeding without it", flush=True)                    
                

        if batches is None:
            time.sleep(15)
            continue

        cvae = run_cvae(gpu_id, hyper_dim=hyper_dim, epochs=100, cm_data_input = batches)
        if cvae is None:
            time.sleep(15)
            continue
        prefix = "_".join((str(generation_id), str(hyper_dim)))

        # We don't need this, we only need the encoder
        # save_model_to_db(client, cvae, prefix)

        client.put_tensor(prefix+"_loss", np.array(cvae.history_call.losses))
        save_model_to_db(client, cvae.encoder, prefix)

        # Write to db
        dataset_name = os.getenv("SSKEYOUT")
        print(f"Writing to {dataset_name}")
        if client.key_exists(dataset_name):
            dataset = client.get_dataset(dataset_name)
        else:
            dataset = Dataset(dataset_name)
        dataset.add_meta_string("prefixes", prefix)
        dataset.add_meta_scalar("latent_dims", int(hyper_dim))
        client.put_dataset(dataset)

        generation_id += 1
        time.sleep(60)
