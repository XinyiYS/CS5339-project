import numpy as np
a =np.array( [1,2,3,3,2,3,2,32,1,4])
import timeit
digits=[2,3,4]
timeit.timeit([_ in digits for _ in a ])
timeit.timeit((sum([ a==digit for digit in digits]) ) >0)
