import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5, 6])
new_arr = np.delete(arr, np.s_[2:4])  # Removes indices 2 and 3

print(new_arr)
