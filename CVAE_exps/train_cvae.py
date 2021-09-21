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
                    backend="TF", device="CPU", inputs=input_names, outputs=output_names)


if __name__ == '__main__': 

    cvae = run_cvae(gpu_id, hyper_dim=hyper_dim)
    prefix = "_".join((str(int(time.time())), str(hyper_dim)))

    client = Client(None, bool(int(os.getenv("SS_CLUSTER", False))))
    client.use_tensor_ensemble_prefix(False)
    
    # We don't need this, we only need the encoder
    # save_model_to_db(client, cvae, prefix)

    client.put_tensor(prefix+"_loss", np.array(cvae.history_call.losses))
    save_model_to_db(client, cvae.encoder, prefix)

    dataset_name = os.getenv("SSKEYOUT")
    print(f"Writing to {dataset_name}")
    if client.key_exists(dataset_name):
        dataset = client.get_dataset(dataset_name)
        dtype = Dtypes.tensor_from_numpy(np.asarray(hyper_dim))
        dataset.add_meta_string("prefixes", prefix)
        dataset.add_meta_scalar("latent_dims", np.asarray(hyper_dim), dtype)
        super(type(client), client).put_dataset(dataset)
    else:
        dataset = Dataset(dataset_name)
        dataset.add_meta_string("prefixes", prefix)
        dataset.add_meta_scalar("latent_dims", int(hyper_dim))
        client.put_dataset(dataset)
