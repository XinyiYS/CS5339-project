import numpy as np


a = np.arange(10).reshape(1,-1)
b = np.arange(10).reshape(-1,1)
print(np.dot(a,b))
print(a.shape,b.shape)
