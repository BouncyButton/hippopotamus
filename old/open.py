import os


def view_data(title=None, N=15, dir='mri_dataset/',
              filename='s01/s01_hippolabels_hres_L_MNI.nii.gz'):
    import nibabel as nib
    import os

    img = nib.load(os.path.join(dir, filename))

    data = img.get_fdata()
    print(data.shape, data.sum())

    plot_data(title=title, N=N,
              data=data)


def plot_data(data, N=15, title='Data Visualization'):
    import matplotlib.pyplot as plt
    from math import sqrt

    plt.figure(figsize=(12, 12))

    L = data.shape[2]
    for i in range(N):
        plt.subplot(int(sqrt(N)) + 1, int(sqrt(N)) + 1, i + 1)
        plt.imshow(data[:, :, i * L // N])
        plt.axis('off')

    plt.title(title)
    plt.show()


f = open('labels/_L.nii.npz', 'rb')
import numpy as np

data = np.load(f)
data = data['arr_0']
print(data.shape)

print(data.min(), data.max())
