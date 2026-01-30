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
parser.add_argument('--rebuild', action='store_true', help='Force rebuild from source', default=False)
args = parser.parse_args()

TARGET_FOLDER = args.target
WANDB_PROJECT = "hippopotamus-project"
WANDB_ENTITY = "hippopotamus"
ARTIFACT_NAME = f"{WANDB_ENTITY}/{WANDB_PROJECT}/Dataset103_ADNI:latest"

# URLs for the aligned MINC releases
URL_LABELS_MINC = 'http://hippocampal-protocol.net/SOPs/LINK_PAGE/FINAL_RELEASE/Released_labels_MNC.zip'
URL_SCANS_MINC = 'http://hippocampal-protocol.net/SOPs/LINK_PAGE/FINAL_RELEASE/Released_ACPC_brainScans_MNC.zip'


def download_from_wandb():
    print(f"Checking for artifact {ARTIFACT_NAME}...")
    try:
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="dataset-download")
        artifact = run.use_artifact(ARTIFACT_NAME, type='dataset')
        artifact.download(root=TARGET_FOLDER)
        print(f"Successfully downloaded to {TARGET_FOLDER}")
        run.finish()
        return True
    except Exception as e:
        print(f"W&B download failed: {e}")
        if 'run' in locals(): run.finish()
        return False


def find_min_max(single, axis=0):
    indices = np.where(single > 0)
    if len(indices[axis]) == 0: return 0, 0
    return indices[axis].min(), indices[axis].max()


def get_all_minc_data():
    """Download and prepare the MINC files."""
    if not os.path.exists('Released_ACPC_brainScans_MNC.zip'):
        print("Downloading MINC Scans...")
        wget.download(URL_SCANS_MINC)
        with zipfile.ZipFile('Released_ACPC_brainScans_MNC.zip', 'r') as z:
            z.extractall('./minc_scans/')

    if not os.path.exists('Released_labels_MNC.zip'):
        print("\nDownloading MINC Labels...")
        wget.download(URL_LABELS_MINC)
        with zipfile.ZipFile('Released_labels_MNC.zip', 'r') as z:
            z.extractall('./minc_labels/')

    label_files = glob.glob('./minc_labels/Labels*/*.mnc')
    data_list = []

    half_x, half_y, half_z = 32 // 2, 48 // 2, 40 // 2

    print(f"Processing {len(label_files)} label files...")
    for l_path in tqdm(label_files):
        # Parse IDs
        bname = os.path.basename(l_path)
        parts = bname.replace('.mnc', '').split('_')
        direction = parts[-1]
        image_id = parts[4]
        subject_id = f"{parts[1]}_{parts[2]}_{parts[3]}"

        # Match with Scan
        s_path = f"./minc_scans/{subject_id}/{subject_id}_{image_id}.mnc"
        if not os.path.exists(s_path):
            continue

        l_img = nib.load(l_path)
        l_data = l_img.get_fdata()
        s_img = nib.load(s_path)
        s_data = s_img.get_fdata()

        # Find crop center based on label
        min_x, max_x = find_min_max(l_data, axis=0)
        min_y, max_y = find_min_max(l_data, axis=1)
        min_z, max_z = find_min_max(l_data, axis=2)
        avg_x, avg_y, avg_z = (min_x + max_x) // 2, (min_y + max_y) // 2, (min_z + max_z) // 2

        # Crop both
        c_label = l_data[avg_x - half_x:avg_x + half_x, avg_y - half_y:avg_y + half_y,
                  avg_z - half_z:avg_z + half_z].copy()
        c_image = s_data[avg_x - half_x:avg_x + half_x, avg_y - half_y:avg_y + half_y,
                  avg_z - half_z:avg_z + half_z].copy()

        data_list.append({
            'label_data': c_label,
            'image_data': c_image,
            'affine': l_img.affine,
            'image_id': image_id,
            'subject_id': subject_id,
            'direction': direction
        })

    return pd.DataFrame(data_list)


if __name__ == '__main__':
    should_rebuild = args.rebuild
    if not should_rebuild:
        if not download_from_wandb():
            should_rebuild = True

    if not should_rebuild:
        print("Data processed via W&B.")
        sys.exit(0)

    # Rebuild logic
    os.makedirs(TARGET_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(TARGET_FOLDER, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_FOLDER, 'labelsTr'), exist_ok=True)

    if not os.path.exists('adni_hippocampus_minc.pkl'):
        df = get_all_minc_data()
        df.to_pickle('adni_hippocampus_all.pkl', compression="gzip")
    else:
        df = pd.read_pickle('adni_hippocampus_all.pkl', compression="gzip")

    print(f"Exporting {len(df)} cases to NIfTI...")
    for _, row in tqdm(df.iterrows()):
        out_name = f"hippo_{row['image_id']}_{row['direction']}"

        # Save Image (_0000 suffix for nnU-Net)
        img_nii = nib.Nifti1Image(row['image_data'], row['affine'])
        nib.save(img_nii, os.path.join(TARGET_FOLDER, 'imagesTr', f'{out_name}_0000.nii.gz'))

        # Save Label
        lbl_nii = nib.Nifti1Image(row['label_data'].astype(np.uint8), row['affine'])
        nib.save(lbl_nii, os.path.join(TARGET_FOLDER, 'labelsTr', f'{out_name}.nii.gz'))

    # W&B Upload
    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    artifact = wandb.Artifact("Dataset103_ADNI", type="dataset")
    artifact.add_dir(TARGET_FOLDER)
    run.log_artifact(artifact)
    run.finish()
    print("Done. Check your W&B dashboard.")