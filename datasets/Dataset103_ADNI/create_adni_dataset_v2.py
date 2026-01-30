import nibabel as nib
import numpy as np
import glob
import pandas as pd
import wget
import zipfile
import os
import sys
import wandb
import argparse
from tqdm import tqdm

# 1. Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='Dataset103_ADNI_MINC', help='Target folder name')
parser.add_argument('--rebuild', action='store_true', help='Force rebuild', default=False)
args = parser.parse_args()

TARGET_FOLDER = args.target
os.makedirs(TARGET_FOLDER, exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'labelsTr'), exist_ok=True)

# URL for MINC Labels (To match the MINC scans)
URL_LABELS_MINC = 'http://hippocampal-protocol.net/SOPs/LINK_PAGE/FINAL_RELEASE/Released_labels_MNC.zip'
URL_SCANS_MINC = 'http://hippocampal-protocol.net/SOPs/LINK_PAGE/FINAL_RELEASE/Released_ACPC_brainScans_MNC.zip'


def find_min_max(single, axis=0):
    """Find bounds of the non-zero mask."""
    indices = np.where(single > 0)
    if len(indices[axis]) == 0: return 0, 0
    return indices[axis].min(), indices[axis].max()


def process_minc_dataset():
    # Download Scans
    if not os.path.exists('Released_ACPC_brainScans_MNC.zip'):
        print("Downloading MINC Scans...")
        wget.download(URL_SCANS_MINC)
    if not os.path.exists('./minc_scans/'):
        with zipfile.ZipFile('Released_ACPC_brainScans_MNC.zip', 'r') as z:
            z.extractall('./minc_scans/')

    # Download MINC Labels
    if not os.path.exists('Released_labels_MNC.zip'):
        print("\nDownloading MINC Labels...")
        wget.download(URL_LABELS_MINC)
    if not os.path.exists('./minc_labels/'):
        with zipfile.ZipFile('Released_labels_MNC.zip', 'r') as z:
            z.extractall('./minc_labels/')

    # Gather all Label files
    label_files = glob.glob('./minc_labels/Labels*/*.mnc')
    data_rows = []

    # Constants for cropping
    half_x, half_y, half_z = 32 // 2, 48 // 2, 40 // 2

    print(f"Processing {len(label_files)} files...")
    for l_path in tqdm(label_files):
        # 1. Load Label
        l_img = nib.load(l_path)
        l_data = l_img.get_fdata()
        l_affine = l_img.affine

        # Determine L/R and IDs from filename
        # Format: ADNI_002_S_0223_13501_L.mnc
        bname = os.path.basename(l_path)
        parts = bname.replace('.mnc', '').split('_')
        direction = parts[-1]  # L or R
        image_id = parts[4]
        subject_id = f"{parts[1]}_{parts[2]}_{parts[3]}"

        # 2. Find and Load corresponding Scan
        # Scan path format: ./minc_scans/ADNI_002_S_0223/ADNI_002_S_0223_13501.mnc
        s_path = f"./minc_scans/{parts[1]}_{parts[2]}_{parts[3]}/{parts[1]}_{parts[2]}_{parts[3]}_{image_id}.mnc"

        if not os.path.exists(s_path):
            continue

        s_img = nib.load(s_path)
        s_data = s_img.get_fdata()

        # 3. Handle Cropping (Crucial: Use the same indices for both)
        min_x, max_x = find_min_max(l_data, axis=0)
        min_y, max_y = find_min_max(l_data, axis=1)
        min_z, max_z = find_min_max(l_data, axis=2)

        avg_x, avg_y, avg_z = (min_x + max_x) // 2, (min_y + max_y) // 2, (min_z + max_z) // 2

        # Crop both using the label-derived center
        cropped_label = l_data[avg_x - half_x:avg_x + half_x, avg_y - half_y:avg_y + half_y,
                        avg_z - half_z:avg_z + half_z]
        cropped_image = s_data[avg_x - half_x:avg_x + half_x, avg_y - half_y:avg_y + half_y,
                        avg_z - half_z:avg_z + half_z]

        # 4. Save to nnU-Net format
        # IMPORTANT: Use the original affine from the MINC file, not identity!
        # This preserves voxel spacing (e.g. 1.0mm) which nnU-Net needs.
        out_name = f"adni_{image_id}_{direction}"

        img_nii = nib.Nifti1Image(cropped_image, l_affine)
        lbl_nii = nib.Nifti1Image(cropped_label.astype(np.uint8), l_affine)

        nib.save(img_nii, os.path.join(TARGET_FOLDER, 'imagesTr', f'{out_name}_0000.nii.gz'))
        nib.save(lbl_nii, os.path.join(TARGET_FOLDER, 'labelsTr', f'{out_name}.nii.gz'))

    print(f"Finished. Saved to {TARGET_FOLDER}")


if __name__ == '__main__':
    process_minc_dataset()