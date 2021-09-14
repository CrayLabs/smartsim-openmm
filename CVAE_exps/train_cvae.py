import os, sys, errno
import argparse 
from cvae.CVAE import run_cvae  
import numpy as np 

from smartsim.tf import freeze_model
from smartredis import Client


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--h5_file", dest="f", default='cvae_input.h5', help="Input: contact map h5 file")
# parser.add_argument("-o", help="output: cvae weight file. (Keras cannot load model directly, will check again...)")
parser.add_argument("-d", "--dim", default=3, help="Number of dimensions in latent space")
parser.add_argument("-gpu", default=0, help="gpu_id")

args = parser.parse_args()

cvae_input = args.f
hyper_dim = int(args.dim) 
gpu_id = args.gpu

if not os.path.exists(cvae_input):
    raise IOError('Input file doesn\'t exist...')


if __name__ == '__main__': 
    cvae = run_cvae(gpu_id, cvae_input, hyper_dim=hyper_dim)

    model_path, inputs, outputs = freeze_model(cvae, os.getcwd(), "cvae.pb")
    prefix = os.path.split(os.getcwd())[1]

    client = Client(None, False)
    client.set_model_from_file(
        prefix+"_cvae", model_path, "TF", device="CPU", inputs=inputs, outputs=outputs
    )
    client.put_tensor(prefix+"_loss", np.array(cvae.history_call.losses))

    model_weight = 'cvae_weight.h5' 
    model_file = 'cvae_model.h5' 
    loss_file = 'loss.npy' 

    batch = client.get_tensor("preproc_0").astype(np.float32)

    client.run_model(prefix+"_cvae", ["preproc_0"], ["output"])
    output = client.get_tensor("output")

    output_here = cvae(batch)

    print(np.mean(output_here-output))

    cvae.save(model_weight)
    # cvae.save(model_file)
    np.save(loss_file, cvae.history_call.losses) 
    