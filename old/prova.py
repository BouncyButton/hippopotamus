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

def get_all_labels(dir, filename_structure='s*/s*_', name='hippolabels_hres_R_MNI.nii.gz'):
    import nibabel as nib
    import numpy as np
    import os

    filepath = os.path.join(dir, filename_structure + name)
    import glob
    files = glob.glob(filepath)

    aggreg = None

    for f in files:
        img = nib.load(f)
        data = img.get_fdata()
        if aggreg is None:
            aggreg = data.copy()
        else:
            aggreg += data

    aggreg = np.array(aggreg)
    # compute space taken from array
    print(f"Array size: {aggreg.nbytes / (1024 ** 3):.2f} GB")

    # crop to leave out the zeros
    # find the min-max indices along each axis

    def find_min_max(axis=0):
        min_v = None
        for i in range(aggreg.shape[axis]):
            idx = [slice(None)] * 3
            idx[axis] = i
            if np.any(aggreg[tuple(idx)]):
                min_v = i
                break

        max_v = None
        for i in range(aggreg.shape[axis] - 1, -1, -1):
            idx = [slice(None)] * 3
            idx[axis] = i
            if np.any(aggreg[tuple(idx)]):
                max_v = i
                break

        return min_v, max_v

    min_x, max_x = find_min_max(axis=0)
    min_y, max_y = find_min_max(axis=1)
    min_z, max_z = find_min_max(axis=2)

    print(f"Cropping indices: x({min_x},{max_x}), y({min_y},{max_y}), z({min_z},{max_z})")

    if max_x - min_x > 64 or max_y - min_y > 64 or max_z - min_z > 64:
        raise ValueError('Uhmmmm we may have a problem')

    # calculate mid points
    avg_x = (min_x + max_x) // 2
    avg_y = (min_y + max_y) // 2
    avg_z = (min_z + max_z) // 2

    aggreg = aggreg[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1].copy()
    # plot_data(N=25, data=aggreg)
    print(aggreg.shape)

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
    np.savez_compressed(os.path.join(dir, name + '.npz'), labels)
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

labels = get_all_labels(dir='/home/bergamin/storage/mni-hisub25/mri_dataset/',
                        name='t1w_standard_defaced_MNI.nii.gz')