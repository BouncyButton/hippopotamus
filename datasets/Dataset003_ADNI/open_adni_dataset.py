import pandas as pd
import matplotlib.pyplot as plt

# disable pycharm viewer using another backend for matplotlib
plt.switch_backend('TkAgg')

# open pickled file
df = pd.read_pickle('adni_hippocampus_full.pkl', compression="gzip")

print(df.head())

plt.figure(figsize=(50, 10))

N = 5

for i in range(N):
    plt.subplot(N, 2, 2 * i + 1)
    x = df.iloc[0]['image_data']
    y = df.iloc[0]['data']
    plt.imshow(x[:, :, x.shape[2] // N * (i + 1)])
    plt.subplot(N, 2, 2 * i + 2)
    plt.imshow(y[:, :, y.shape[2] // N * (i + 1)])
plt.show()
