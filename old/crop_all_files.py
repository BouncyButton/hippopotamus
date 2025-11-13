# use x(124,206), y(195,313), z(99,209) for L hres
# use  x(237,327), y(204,311), z(96,207) for R hres


# hippolabels_hres_L_MNI.nii.gz (25, 83, 119, 111) x(124,206), y(195,313), z(99,209)
# hippolabels_hres_R_MNI.nii.gz (25, 91, 108, 112) x(237,327), y(204,311), z(96,207)
# hippolabels_t1w_standard_L_MNI.nii.gz (25, 33, 48, 44) x(58,90), y(86,133), z(40,83)
# hippolabels_t1w_standard_R_MNI.nii.gz (25, 36, 43, 45) x(103,138), y(90,132), z(39,83)

def save_cropped_data(dir, filename_structure='s*/s*_hippolabels_hres_L_MNI.nii.gz'):
    import nibabel as nib
    import numpy as np
    import os

    filepath = os.path.join(dir, filename_structure)
    import glob
    files = glob.glob(filepath)
    
    # save to disk a numpy file of all crops
    labels = []
    for f in files:
        img = nib.load(f)
        data = img.get_fdata()
        cropped = data[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]
        labels.append(cropped)

    labels = np.array(labels)
    # save to disk
    np.savez_compressed(os.path.join(dir, 'hippolabels_hres_L_MNI.npz'), labels)
    return labels
