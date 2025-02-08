# import numpy as np

# # Contoh Q-values hasil dari neural network
# q_values = np.array([1.2, 0.5, -0.2, -0.5, 0.8])  # [Accept, Wait1, Wait2, Wait3, Continue]

# # Mask untuk aksi valid (1 = valid, 0 = tidak valid)
# mask = np.array([1, 1, 0, 1, 1])  # Misalnya, Wait2 tidak valid

# # Terapkan mask pada Q-values
# masked_q_values = np.where(mask == 1, q_values, -np.inf)

# # Pilih aksi terbaik menggunakan argmax
# best_action = np.argmax(masked_q_values)
# print(f"Aksi terbaik adalah: {best_action}")  # Output: Indeks aksi terbaik


a = {"B": [3,4]}
print(a.items())
for key, value in a.items():
    a[key].pop(1)

print(a)
print(str(list(a.keys())[0]))
