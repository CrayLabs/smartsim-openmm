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

from google.protobuf.text_format import Parse

from smartsim.tf.utils import freeze_model

cm_data_train = np.ones(shape=(100,22,22,1), dtype=np.float32)
cm_data_val = np.ones(shape=(40,22,22,1), dtype=np.float32)

input_shape = cm_data_train.shape
cvae = CVAE.CVAE(input_shape, 3)

cvae.train(cm_data_train, cm_data_val, 5, 1)

# cvae.train(cm_data_train, cm_data_val, 10, 5)

# model_path, inputs, outputs = freeze_model(cvae, os.getcwd(), "saved_model.pb")
# print(model_path, model_path, outputs)

client = Client(None, False)

full_model = tf.function(lambda x: cvae(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(cvae.inputs[0].shape, cvae.inputs[0].dtype)
)

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

input_names = [x.name.split(":")[0] for x in frozen_func.inputs]
output_names = [x.name.split(":")[0] for x in frozen_func.outputs]

model_serialized = frozen_func.graph.as_graph_def().SerializeToString(deterministic=True)

client.set_model("the_model", model=model_serialized, tag="",
                 backend="TF", device="CPU", inputs=input_names, outputs=output_names)

inference = cvae(cm_data_train[0:1])

cvae.save("cvae_weights.h5")

cvae2 = CVAE.CVAE(input_shape,3)
cvae2.load("cvae_weights.h5")

first_sample = cm_data_train[0:1,:,:,:]
print(first_sample.shape)

new_inference = cvae2(cm_data_train[0:1,:,:,:])

print(np.mean(new_inference-inference))

# cvae2.train(cm_data_train, cm_data_val, 10, 5)

client.put_tensor("sample", first_sample)
client.run_model("the_model", ["sample"], ["output"])
output = client.get_tensor("output")

print(np.mean(inference-output))


graph_str = client.get_model("the_model")

assert(graph_str == model_serialized)
