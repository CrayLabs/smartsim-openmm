import numpy as np
from smartredis import Client, Dataset
from smartredis.util import Dtypes

client = Client(None, False)

out = np.zeros(shape=(12,25))

dataset = Dataset("test_dataset")
dataset.add_tensor("out", out)
dataset.add_meta_scalar("scalars", out.shape[1])
dataset.add_meta_scalar("scalars", out.shape[1])
dataset.add_meta_scalar("scalars", out.shape[1])
client.put_dataset(dataset)

dataset1 = client.get_dataset("test_dataset")
scalars = dataset1.get_meta_scalars("scalars")
print(type(scalars), scalars.shape)
print(type(dataset), type(dataset1))
dtype = Dtypes.tensor_from_numpy(out)
dataset1.add_tensor("out_new", out, dtype)

super(type(client), client).put_dataset(dataset1)
