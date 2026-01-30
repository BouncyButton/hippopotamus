# let's gather everything i did for Dataset003_ADNI
import nibabel as nib
import numpy as np
import glob

import pandas as pd
import wget
import zipfile
import os
from tqdm import tqdm
import sys
import wandb
import argparse

# 1. Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='Dataset103_ADNI', help='Target folder name')
parser.add_argument('--rebuild', action='store_true', help='Force download from ADNI and rebuild pkl', default=False)
args = parser.parse_args()

TARGET_FOLDER = args.target
WANDB_PROJECT = "hippopotamus-project"
WANDB_ENTITY = "hippopotamus"
ARTIFACT_NAME = f"{WANDB_ENTITY}/{WANDB_PROJECT}/Dataset103_ADNI:latest"


def download_from_wandb():
    print(f"Checking for artifact {ARTIFACT_NAME}...")
    try:
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="dataset-download")
        artifact = run.use_artifact(ARTIFACT_NAME, type='dataset')
        artifact_dir = artifact.download(root=TARGET_FOLDER)
        print(f"Successfully downloaded to {TARGET_FOLDER}")
        run.finish()
        return True
    except Exception as e:
        print(f"W&B download failed: {e}")
        if 'run' in locals():
            run.finish()
        return False


# 2. Orchestration Logic
# We run the rebuild if --rebuild is passed
should_rebuild = args.rebuild

if not should_rebuild:
    # If folder exists but we aren't rebuilding, try to sync from W&B if data is missing
    # (e.g., folder exists but is empty)
    print("Attempting to use existing data or download from W&B...")
    if not download_from_wandb():
        print("Falling back to full rebuild...")
        should_rebuild = True

if not should_rebuild:
    print("Data downloaded from wandb successfully.")
    sys.exit(0)

# makedir
os.makedirs(TARGET_FOLDER, exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'labelsTr'), exist_ok=True)


def get_all_labels(dir, name=''):
    files = glob.glob(os.path.join(dir, name))

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

    def find_min_max(single, axis=0):
        min_v = None
        for i in range(single.shape[0]):
            idx = [slice(None)] * 3
            idx[axis] = i

            if np.any(single[tuple(idx)]):
                min_v = i
                break

        max_v = None
        for i in range(single.shape[axis] - 1, -1, -1):
            idx = [slice(None)] * 3
            idx[axis] = i
            if np.any(single[tuple(idx)]):
                max_v = i
                break

        return min_v, max_v

    # save to disk a numpy file of all crops
    import pandas as pd

    files = [f for f in files if 'CSF' not in f]  # exclude CSF images

    data_list = []
    crop_idxs = []

    half_x = 32 // 2
    half_y = 48 // 2
    half_z = 40 // 2

    for f in files:
        img = nib.load(f)
        data = img.get_fdata()
        # cropped = data[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

        min_x, max_x = find_min_max(data, axis=0)
        min_y, max_y = find_min_max(data, axis=1)
        min_z, max_z = find_min_max(data, axis=2)

        print(f"Cropping indices: x({min_x},{max_x}), y({min_y},{max_y}), z({min_z},{max_z})")

        # calculate mid points
        avg_x = (min_x + max_x) // 2
        avg_y = (min_y + max_y) // 2
        avg_z = (min_z + max_z) // 2

        crop_idx = ((avg_x - half_x, avg_x + half_x),
                    (avg_y - half_y, avg_y + half_y),
                    (avg_z - half_z, avg_z + half_z))

        cropped = data[avg_x - half_x:avg_x + half_x,
                  avg_y - half_y:avg_y + half_y,
                  avg_z - half_z:avg_z + half_z].copy()

        # subsample 2x
        # cropped = cropped[::2, ::2, ::2]

        data_list.append(cropped)
        crop_idxs.append(crop_idx)

    # now subsample all images and labels 2x

    # a filename has this structure:
    # ADNI_nnn_S_nnnn_xxxxx_D.nii
    # the subject id is nnn_S_nnnn
    direction = 'R' if '_R.mnc' in name else 'L'
    metadata = [os.path.basename(f).split('_' + direction + '.mnc')[0] for f in files]
    metadata = [sid.replace('ADNI_', '') for sid in metadata]
    metadata = [("_".join(sid.split('_')[:-1]), sid.split('_')[-1]) for sid in metadata]
    subject_ids = [sid for sid, _ in metadata]
    image_ids = [img_id for _, img_id in metadata]
    df = pd.DataFrame({'data': data_list, 'filename': files, 'subject_id': subject_ids, 'image_id': image_ids,
                       'direction': [direction] * len(files), 'crop_idx': crop_idxs})

    # data = np.array(data)
    # assert data.shape[1:] == (2 * half, 2 * half, 2 * half), f"Unexpected shape: {data.shape}"
    # save to disk
    # name = name.replace('*', '')
    # print(name, data.shape, f"x({min_x},{max_x}), y({min_y},{max_y}), z({min_z},{max_z})")
    # np.savez_compressed(name + '.npz', data)
    # return data

    return df


def merge_image_data_to_label_data(df, dir):
    print('merge start')
    files = glob.glob(os.path.join(dir, '*/*.mnc'))

    image_data_dict = {}
    for f in tqdm(files):
        img = nib.load(f)
        data = img.get_fdata()
        filename = os.path.basename(f)
        subject_id = "_".join(("_".join(filename.split('_')[:5])).split('_')[1:4])
        image_id = filename.split('_')[4]
        key = (subject_id, image_id)
        image_data_dict[key] = data
        del img

    # now merge
    image_data_list = []
    for idx, row in tqdm(df.iterrows()):
        image_id = row['image_id']
        subject_id = row['subject_id']
        key = (subject_id, image_id)
        image_data = image_data_dict.get(key, None)
        if image_data is None:
            print(f"Warning: image data not found for subject_id={subject_id}, image_id={image_id}")
        else:
            crop_idx = row['crop_idx']
            cropped_image_data = image_data[
                                 crop_idx[0][0]:crop_idx[0][1],
                                 crop_idx[1][0]:crop_idx[1][1],
                                 crop_idx[2][0]:crop_idx[2][1]
                                 ]
            # subsample 2x
            # cropped_image_data = cropped_image_data[::2, ::2, ::2]
            image_data_list.append(cropped_image_data)

    df['image_data'] = image_data_list
    return df


if __name__ == '__main__':
    # check if file exists
    if not os.path.exists('adni_hippocampus_labels.pkl'):
        if not os.path.exists('Released_data_MNC_v1.3.zip'):
            wget.download('http://hippocampal-protocol.net/SOPs/LINK_PAGE/FINAL_RELEASE/Released_data_MNC_v1.3.zip')
            with zipfile.ZipFile('Released_data_MNC_v1.3.zip', 'r') as zip_ref:
                zip_ref.extractall('./adni_data/')

        df_L = get_all_labels(dir='adni_data/Labels*/', name='*_L.mnc')
        df_R = get_all_labels(dir='adni_data/Labels*/', name='*_R.mnc')

        # join
        df = pd.concat([df_L, df_R], ignore_index=True)

        # save to disk
        df.to_pickle('adni_hippocampus_labels.pkl', compression="gzip")

        del df_L
        del df_R

    else:
        df = pd.read_pickle('adni_hippocampus_labels.pkl', compression="gzip")

    print(len(df))

    # print image ids to download the original MRI images
    print(",".join(df['image_id'].unique()))

    # http://hippocampal-protocol.net/SOPs/LINK_PAGE/FINAL_RELEASE/Released_ACPC_brainScans_MNC.zip
    if not os.path.exists('Released_ACPC_brainScans_MNC.zip'):
        wget.download(
            'http://hippocampal-protocol.net/SOPs/LINK_PAGE/FINAL_RELEASE/Released_ACPC_brainScans_MNC.zip')
        with zipfile.ZipFile('Released_ACPC_brainScans_MNC.zip', 'r') as zip_ref:
            zip_ref.extractall('./Released_ACPC_brainScans_MNC/')

    merge_image_data_to_label_data(df, dir='Released_ACPC_brainScans_MNC/')
    df.to_pickle(os.path.join(TARGET_FOLDER, 'adni_hippocampus_full.pkl'), compression="gzip")

    identity_affine = np.eye(4)

    for index, row in df.iterrows():
        image_id = row['image_id']
        direction = row['direction']  # Get the direction (L or R)
        image_data = row['image_data']
        label_data = row['data']

        # Create NIfTI image for image_data with direction in filename
        image_nifti = nib.Nifti1Image(image_data, identity_affine)
        image_filename_dataset = os.path.join(TARGET_FOLDER, 'imagesTr',
                                              f'hippocampus_adni_{image_id}_{direction}_0000.nii.gz')
        nib.save(image_nifti, image_filename_dataset)

        # Create NIfTI image for label_data with direction in filename
        label_nifti = nib.Nifti1Image(label_data, identity_affine)
        label_filename_dataset = os.path.join(TARGET_FOLDER, 'labelsTr',
                                              f'hippocampus_adni_{image_id}_{direction}.nii.gz')
        nib.save(label_nifti, label_filename_dataset)

    print(f"Successfully saved {len(df)} image and label files.")

    # save all into a private wandb to avoid re-downloading
    wandb.init(project="hippopotamus-project", entity='hippopotamus')

    artifact = wandb.Artifact("Dataset103_ADNI", type="dataset")
    artifact.add_dir(TARGET_FOLDER)

    wandb.log_artifact(artifact)
    wandb.finish()

    # ok ignore this

    # refer to https://www.hippocampal-protocol.net/SOPs/screenshots/harp_final_release/search.png to find image data
    # from http://www.hippocampal-protocol.net/SOPs/labels.php#final
    # if not os.path.exists('HarP.zip'):
    #     wget.download(
    #         'https://ida.loni.usc.edu/download/files/ida1/cb159b77-4d79-4e9e-b543-b49574b63417/HarP.zip')
    #     with zipfile.ZipFile('HarP.zip', 'r') as zip_ref:
    #         zip_ref.extractall('./')

    # download from Analysis Ready Cohort (ARC) Builder.
    # https://ida.loni.usc.edu/explore/jsp/search_v2/search.jsp?project=ADNI
    # after getting official access to the data, below there's "Create New Filter".
    # add the image ids printed above
    # (Image Filters > Choose images from a list of image IDs that you provide > Create new filter > Enter image IDs)

    # # my generated link was this, but it will probably expire in the future
    # if not os.path.exists('Data_for_Hippocampal-protocol.net_labels_MRI.zip'):
    #     wget.download(
    #         'https://ida.loni.usc.edu/download/files/ida1/'
    #         '64a3f247-4d06-4212-a2dd-e898d94441e9/Data_for_Hippocampal-protocol.net_labels_MRI.zip')
    #     with zipfile.ZipFile('Data_for_Hippocampal-protocol.net_labels_MRI.zip', 'r') as zip_ref:
    #         zip_ref.extractall('./')
    #
    # # make output dir
    # output_dir = 'ADNI_nii_image_data/'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # # convert DCM to NII
    # from nipype.interfaces.dcm2nii import Dcm2niix
    #
    # converter = Dcm2niix()
    # converter.inputs.source_dir = 'Dataset003_ADNI/'
    # converter.inputs.output_dir = output_dir
    # converter.inputs.compress = 'y'
    # converter.run()
    #
    # df = merge_image_data_to_label_data(df, dir='Dataset003_ADNI/')
    #
    # # save to disk
    # df.to_pickle('adni_hippocampus_labels_with_images.pkl', compression="gzip")
