import os


def view_data(title=None, N=15, dir='mri_dataset/',
              filename='s01/s01_hippolabels_hres_L_MNI.nii.gz'):
    import nibabel as nib
    import os

    img = nib.load(os.path.join(dir, filename))

    data = img.get_fdata()
    print(data.shape)
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


import sys

print(sys.executable)


# view_data(dir='/home/bergamin/storage/mni-hisub25/mri_dataset/',
#           filename='s01/s01_hippolabels_t1w_standard_L_MNI.nii.gz')

def get_all_data(dir, filename_structure='s*/s*_', name='hippolabels_hres_R_MNI.nii.gz',
                 min_x=None, max_x=None,
                 min_y=None, max_y=None,
                 min_z=None, max_z=None):
    import nibabel as nib
    import numpy as np
    import os

    filepath = os.path.join(dir, filename_structure + name)
    import glob
    files = glob.glob(filepath)

    # calculate mid points
    avg_x = (min_x + max_x) // 2
    avg_y = (min_y + max_y) // 2
    avg_z = (min_z + max_z) // 2

    # save to disk a numpy file of all crops
    labels = []
    for f in files:
        img = nib.load(f)
        data = img.get_fdata()
        # cropped = data[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]
        cropped = data[avg_x - 32:avg_x + 32,
                  avg_y - 32:avg_y + 32,
                  avg_z - 32:avg_z + 32]
        labels.append(cropped)

    labels = np.array(labels)
    assert labels.shape[1:] == (64, 64, 64), f"Unexpected shape: {labels.shape}"
    # save to disk
    print(name, labels.shape, f"x({min_x},{max_x}), y({min_y},{max_y}), z({min_z},{max_z})")
    np.savez_compressed(os.path.join(dir, name + 'R.npz'), labels)
    return labels


# labels = get_all_labels(dir='/home/bergamin/storage/mni-hisub25/mri_dataset/',
#                         name='hippolabels_hres_L_MNI.nii.gz')
#
# labels = get_all_labels(dir='/home/bergamin/storage/mni-hisub25/mri_dataset/',
#                         name='hippolabels_hres_R_MNI.nii.gz')
#
# labels = get_all_labels(dir='/home/bergamin/storage/mni-hisub25/mri_dataset/',
#                         name='hippolabels_t1w_standard_L_MNI.nii.gz')
#
# labels = get_all_labels(dir='/home/bergamin/storage/mni-hisub25/mri_dataset/',
#                         name='hippolabels_t1w_standard_R_MNI.nii.gz')

# x(58,90), y(86,133), z(40,83)
# min_x = 58
# max_x = 90
# min_y = 86
# max_y = 133
# min_z = 40
# max_z = 83

# x(103,138), y(90,132), z(39,83)
min_x = 103
max_x = 138
min_y = 90
max_y = 132
min_z = 39
max_z = 83

labels = get_all_data(dir='/home/bergamin/storage/mni-hisub25/mri_dataset/',
                      name='t1w_standard_defaced_MNI.nii.gz',
                      min_x=min_x, max_x=max_x,
                      min_y=min_y, max_y=max_y,
                      min_z=min_z, max_z=max_z)
