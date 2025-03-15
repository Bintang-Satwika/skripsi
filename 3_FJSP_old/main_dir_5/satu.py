import numpy as np
states = np.ones((3, 14)) * 3  # sample data with values roughly between 0 and 100

# Define min and max values for each of the 14 features (states).
# Replace these example values with your actual min and max for each state.
state_mins = np.array([3,  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
state_maxs = np.array([11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2])

# Normalize each feature (column) using broadcasting.
normalized_states = (states - state_mins) / (state_maxs - state_mins)
print("states:\n", states)
print("normalized_states:\n", np.float32(normalized_states))
print("xxxxxxx\n",  np.float32(states - state_mins))