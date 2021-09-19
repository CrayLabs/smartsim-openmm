from google import protobuf
from CVAE_exps.cvae import CVAE
import numpy as np
import os
from pathlib import Path

from tensorflow.core.framework import graph_pb2

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from smartredis import Client

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

cm_data_train = np.random.rand(100,22,22,1).astype(np.float32)
cm_data_val = np.random.rand(40,22,22,1).astype(np.float32)

input_shape = cm_data_train.shape
cvae = CVAE.CVAE(input_shape, 3)

cvae.train(cm_data_train, cm_data_val, 5, 1)

prefix = "test"
client = Client(None, False)
save_model_to_db(client, cvae, prefix)
save_model_to_db(client, cvae.encoder, prefix)


# Change sampling to avoid randomness to run these
client.put_tensor("samples", cm_data_val)
client.run_model("_".join((prefix, "CVAE")), inputs=["samples"], outputs=["cvae_out"])
client.run_model("_".join((prefix, "encoder")), inputs=["samples"], outputs=["z_mean", "z_logvar", "z"])

cvae.save("the_model.h5")
cvae2 = CVAE.CVAE(input_shape, 3)
cvae2.load("the_model.h5")

inference1 = cvae.return_embeddings(cm_data_val)
inference2 = cvae2.return_embeddings(cm_data_val)
inferenceRAI = client.get_tensor("z")

print(np.mean(np.abs(inference1-inference2)))
print(np.mean(np.abs(inference1-inferenceRAI)))
