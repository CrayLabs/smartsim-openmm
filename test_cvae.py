from MD_exps.MD_utils_fspep import CVAE
import numpy as np
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

from smartsim.tf.utils import freeze_model

cm_data_train = np.random.rand(100,22,22,1)
cm_data_val = np.random.rand(40,22,22,1)

input_shape = cm_data_train.shape
cvae = CVAE.CVAE(input_shape, 3)

cvae.train(cm_data_train, cm_data_val, 10, 1)

model_path, inputs, outputs = freeze_model(cvae, os.getcwd(), "saved_model.pb")
print(model_path, model_path, outputs)

# loaded = tf.saved_model.load(
#     os.path.join(os.getcwd()))

# loaded.summary()
print(type(cvae.history_call.losses))