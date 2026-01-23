import numpy as np
import wget
import tarfile
import os
import nibabel as nib
import wandb
import pandas as pd

import sys

TARGET_FOLDER = sys.argv[1] if len(sys.argv) > 1 else 'MNI'

def get_data(dir, filename_structure='s*/',
             name='s*_hippolabels_hres_R_MNI.nii.gz',
             half=None):
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
    print(aggreg.shape)

    # crop to leave out the zeros
    # find the min-max indices along each axis

    def find_min_max(axis=0):
        min_v = None
        for i in range(aggreg.shape[0]):
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

    print(f"Processing {name}")
    print(f"Cropping indices: x({min_x},{max_x}), y({min_y},{max_y}), z({min_z},{max_z})")

    half_x = 40 // 2
    half_y = 56 // 2
    half_z = 48 // 2

    if half is not None:
        half_x = half_y = half_z = half

        if max_x - min_x > 2 * half or max_y - min_y > 2 * half or max_z - min_z > 2 * half:
            raise ValueError('Uhmmmm we may have a problem')

    # calculate mid points
    avg_x = (min_x + max_x) // 2
    avg_y = (min_y + max_y) // 2
    avg_z = (min_z + max_z) // 2

    data_list = []
    crop_idxs = []
    subject_ids = [os.path.basename(f)[:3] for f in files]
    direction = 'R' if 'R' in name else 'L'
    for f in files:
        img = nib.load(f)
        data = img.get_fdata()
        crop_idx = ((avg_x - half_x, avg_x + half_x),
                    (avg_y - half_y, avg_y + half_y),
                    (avg_z - half_z, avg_z + half_z))

        cropped = data[avg_x - half_x:avg_x + half_x,
                  avg_y - half_y:avg_y + half_y,
                  avg_z - half_z:avg_z + half_z]
        data_list.append(cropped)
        crop_idxs.append(crop_idx)

    df = pd.DataFrame({'data': data_list, 'filename': files, 'subject_id': subject_ids,
                       'direction': [direction] * len(files), 'crop_idx': crop_idxs})

    return df


def merge_image_data_to_label_data(df, dir, name='s*_t1w_standard_defaced_MNI.nii.gz'):
    import glob
    files = glob.glob(os.path.join(dir, name))

    image_data_dict = {}
    for f in files:
        img = nib.load(f)
        data = img.get_fdata()
        filename = os.path.basename(f)
        subject_id = filename.split('_')[0]
        image_data_dict[subject_id] = data

    # now merge
    image_data_list = []
    for idx, row in df.iterrows():
        subject_id = row['subject_id']
        image_data = image_data_dict.get(subject_id, None)
        if image_data is None:
            print(f"Warning: image data not found for subject_id={subject_id}")
        else:
            crop_idx = row['crop_idx']
            cropped_image_data = image_data[
                                 crop_idx[0][0]:crop_idx[0][1],
                                 crop_idx[1][0]:crop_idx[1][1],
                                 crop_idx[2][0]:crop_idx[2][1]
                                 ].copy()
            image_data_list.append(cropped_image_data)

    df['image_data'] = image_data_list
    return df


def simple_progress(current, total, width=80):
    print(f"\rDownloaded {current}/{total} bytes ({current / total:.2%})", end="")


if not os.path.exists('mni-hisub25/'):
    wget.download('https://mni-hisub25.projects.nitrc.org/downloads/mni-hisub25.tar', bar=simple_progress)

    with tarfile.open('mni-hisub25.tar', 'r') as tar_ref:
        tar_ref.extractall('./')

    with tarfile.open('mni-hisub25/mri_dataset.tar.gz', 'r:gz') as tar_ref:
        tar_ref.extractall('./')

df_L = get_data('mri_dataset', filename_structure='s*/', name='s*_hippolabels_t1w_standard_L_MNI.nii.gz',
                half=None)
df_R = get_data('mri_dataset', filename_structure='s*/', name='s*_hippolabels_t1w_standard_R_MNI.nii.gz',
                half=None)

df = pd.concat([df_L, df_R], ignore_index=True)

merge_image_data_to_label_data(df, dir='mri_dataset/s*/')
df.to_pickle('mni_hippocampus_full.pkl', compression="gzip")

identity_affine = np.eye(4)

for index, row in df.iterrows():
    image_id = row['subject_id']
    direction = row['direction']  # Get the direction (L or R)
    image_data = row['image_data']
    label_data = row['data']

    # Create NIfTI image for image_data with direction in filename
    image_nifti = nib.Nifti1Image(image_data, identity_affine)
    image_filename_dataset = os.path.join(TARGET_FOLDER, 'imagesTr', f'hippocampus_mni_{image_id}_{direction}_0000.nii.gz')
    nib.save(image_nifti, image_filename_dataset)

    # Create NIfTI image for label_data with direction in filename
    label_nifti = nib.Nifti1Image(label_data, identity_affine)
    label_filename_dataset = os.path.join(TARGET_FOLDER, 'labelsTr', f'hippocampus_mni_{image_id}_{direction}.nii.gz')
    nib.save(label_nifti, label_filename_dataset)

print(f"Successfully saved {len(df)} image and label files.")

# save all into a private wandb to avoid re-downloading
wandb.init(project="hippopotamus")

artifact = wandb.Artifact("Dataset102_MNI", type="dataset")
artifact.add_dir(TARGET_FOLDER)

wandb.log_artifact(artifact)
wandb.finish()
