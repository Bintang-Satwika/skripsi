# import numpy as np

# a = [2,0]
# b = [1, 2, 3]

# b_iter = iter(b)
# print("b_iter: ", b_iter)
# if all(x in b_iter for x in a):
#     print("true")
# else:
#     print("false")

import numpy as np

a = [1,2]
b = [1, 2]

if np.all(np.isin(a, b)):
    print("true")
else:
    print("false")

