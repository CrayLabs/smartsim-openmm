import numpy as np
from smartredis import Client, Dataset
from smartredis.util import Dtypes

client = Client(None, False)

# out = np.zeros(shape=(12,25))

# dataset = Dataset("test_dataset")
# dataset.add_tensor("out", out)
# client.put_dataset(dataset)

# dataset1 = client.get_dataset("test_dataset")

# print(type(dataset), type(dataset1))
# dtype = Dtypes.tensor_from_numpy(out)
# dataset1.add_tensor("out_new", out, dtype)

# super(type(client), client).put_dataset(dataset1)

inp = np.ones(shape=(100,200))

for i in range(100):
    just_messing_memory = -np.ones(shape=(100,200,i+1))
    inp = np.vstack((inp, np.array([1]*200)))

client.put_tensor("inp", inp)
stored = client.get_tensor("inp")

print(np.sum(inp-stored))

#client.set_script_from_file("test_script", "/lus/scratch/arigazzi/smartsim-dev/smartsim-openmm/MD_to_CVAE/MD_to_CVAE_scripts.py", device="CPU")
