import numpy as np
import matplotlib.pyplot as plt

# Misalkan kita punya 10 kategori algoritma
kategories = [
    'FIFO', 'LIFO', 'SPT', 'LPT', 'STPT',
    'LTPT', 'LPTR', 'MPTR', 'MPSR', 'Random'
]

# Nilai mean setiap kategori (contoh data)
mean_values = np.array([320, 310, 315, 305, 300, 
                        320, 308, 311, 303, 330])

# Nilai error bar (misalnya standard deviation)
error_values = np.array([50, 40, 45, 55, 60, 
                         48, 52, 43, 55, 65])

# Selain itu ada kategori "Proposed" dengan data terpisah
# (opsional, sesuai gambar yang tampak memiliki satu bar tambahan)
kategori_baru = "Proposed"
mean_proposed = 280
error_proposed = 40

# Atur ukuran figure
plt.figure(figsize=(10, 6))

# Buat posisi x agar setiap bar memiliki posisi masing-masing di sepanjang sumbu x
x_pos = np.arange(len(kategories))

# Plot bar chart untuk kategories
plt.bar(x_pos, mean_values, yerr=error_values, 
        align='center', alpha=0.8, capsize=5, color='skyblue', 
        label='Algoritma Eksisting')

# Plot bar chart untuk Proposed, letakkan di sebelah paling kanan
# Caranya, tambahkan 1 posisi x di ujung
x_proposed = len(kategories)
plt.bar(x_proposed, mean_proposed, yerr=error_proposed,
        align='center', alpha=0.8, capsize=5, color='orange', 
        label='Proposed')

# Atur label pada sumbu x (agar menampilkan nama kategori)
# Kita tambahkan satu kategori baru, sehingga total len(kategories) + 1
all_categories = kategories + [kategori_baru]
plt.xticks(list(x_pos) + [x_proposed], all_categories, rotation=45)

# Tambahkan judul dan label sumbu
plt.title('Perbandingan Mean Makespan dengan Error Bars')
plt.xlabel('Metode / Algoritma')
plt.ylabel('Mean Value of Makespan')

# Tampilkan legenda
plt.legend()

# Atur margin agar label di sumbu x tidak terpotong
plt.tight_layout()

# Tampilkan grafik
plt.show()
