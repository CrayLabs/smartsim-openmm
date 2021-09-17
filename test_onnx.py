from sklearn.cluster import DBSCAN, KMeans
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np

data = np.random.rand(100,3)

dbs = KMeans()
initial_type = [('float_input', FloatTensorType([None, 3]))]
onx = convert_sklearn(dbs, initial_types=initial_type)


print("Conversion was successful")
