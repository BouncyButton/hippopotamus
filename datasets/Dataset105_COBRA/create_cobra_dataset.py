import tarfile
import nibabel as nib
import numpy as np
import os
import sys

import pandas as pd
import wget
import wandb
import argparse

# 1. Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='Dataset105_COBRA', help='Target folder name')
parser.add_argument('--rebuild', action='store_true', help='Force download from COBRA and rebuild pkl', default=False)
args = parser.parse_args()

TARGET_FOLDER = args.target
WANDB_PROJECT = "hippopotamus-project"
WANDB_ENTITY = "hippopotamus"
ARTIFACT_NAME = f"{WANDB_ENTITY}/{WANDB_PROJECT}/Dataset105_COBRA:latest"


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

url = 'https://cobralab.net/files/brains_t2.tar.bz2'

if not os.path.exists('brains_t2.tar.bz2'):
    wget.download(url)

if not os.path.exists('brains_t2'):
    file = tarfile.open('brains_t2.tar.bz2')
    file.extractall('./')

os.system('git clone https://github.com/CoBrALab/atlases/')

# Configuration
INPUT_LABELS_DIR = "atlases/hippocampus-subfields/labels"  # Current folder for .mnc labels
INPUT_IMAGES_DIR = "brains_t2"
OUTPUT_IMAGES_DIR = "imagesTr"
OUTPUT_LABELS_DIR = "labelsTr"
TARGET_SIZE = np.array([96, 128, 160])

# Label definitions
LEFT_LABELS = [101, 102, 104, 105, 106]
RIGHT_LABELS = [1, 2, 4, 5, 6]

label_mapping = {
    101: 1,  # right ca1
    102: 2,  # right subiculum
    104: 3,  # right ca4
    105: 4,  # right ca2-3
    106: 5,  # right stratum
    1: 1,  # left ca1
    2: 2,  # left subiculum
    4: 3,  # left ca4
    5: 4,  # left ca2-3
    6: 5  # left stratum
}

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)


def crop_and_save(data, affine, center_coords, output_path):
    """Crops data to TARGET_SIZE centered around center_coords and saves as NIfTI."""
    half_size = TARGET_SIZE // 2

    # Calculate start and end indices
    start = np.floor(center_coords - half_size).astype(int)
    end = start + TARGET_SIZE

    # Handle image boundaries with padding if necessary
    pad_before = np.maximum(0, -start)
    pad_after = np.maximum(0, end - np.array(data.shape))

    # Adjust start/end to be within array bounds for slicing
    slice_start = np.maximum(0, start)
    slice_end = np.minimum(np.array(data.shape), end)

    # Extract the crop
    crop = data[slice_start[0]:slice_end[0],
           slice_start[1]:slice_end[1],
           slice_start[2]:slice_end[2]]

    # Apply padding if the crop is smaller than TARGET_SIZE
    padded_crop = np.pad(crop,
                         ((pad_before[0], pad_after[0]),
                          (pad_before[1], pad_after[1]),
                          (pad_before[2], pad_after[2])),
                         mode='constant', constant_values=0)

    # subsample the image 2x
    padded_crop = padded_crop[::2, ::2, ::2]

    # Update affine to reflect the new origin (translation)
    # The new origin is the old origin + the start index in world space
    new_affine = affine.copy()
    new_affine[:3, 3] = nib.affines.apply_affine(affine, start)

    new_img = nib.Nifti1Image(padded_crop.astype(data.dtype), new_affine)
    nib.save(new_img, output_path)

    return padded_crop


data_records = []
# Process brains 1 to 5
for i in range(1, 6):
    label_fname = f"brain{i}_labels.mnc"
    image_fname = f"brain{i}_t2.mnc"  # Assuming image filename matches pattern

    label_path = os.path.join(INPUT_LABELS_DIR, label_fname)
    image_path = os.path.join(INPUT_IMAGES_DIR, image_fname)

    if not os.path.exists(label_path) or not os.path.exists(image_path):
        print(f"Skipping brain{i}: missing file(s).")
        continue

    print(f"Processing brain{i}...")

    # Load data
    lbl_img = nib.load(label_path)
    img_img = nib.load(image_path)

    lbl_data = lbl_img.get_fdata()
    img_data = img_img.get_fdata()
    affine = img_img.affine

    original_lbl_data = lbl_data.copy()

    for side_code, labels in [("L", LEFT_LABELS), ("R", RIGHT_LABELS)]:
        lbl_data = original_lbl_data
        # Find the center of the specific hippocampus
        mask = np.isin(lbl_data, labels)
        coords = np.argwhere(mask)

        if coords.size == 0:
            print(f"  Warning: No {side_code} hippocampus labels in brain{i}")
            continue

        # Calculate geometric center of the labels
        center = (coords.min(axis=0) + coords.max(axis=0)) / 2

        # remap labels
        remapped_lbl_data = np.zeros_like(lbl_data)
        for original_label, new_label in label_mapping.items():
            remapped_lbl_data[lbl_data == original_label] = new_label

        lbl_data = remapped_lbl_data

        # Prepare filenames
        # Image: hippocampus_cobra_ID_SIDE_0000.nii.gz
        # Label: hippocampus_cobra_ID_SIDE.nii.gz
        img_out = os.path.join(TARGET_FOLDER, OUTPUT_IMAGES_DIR, f"hippocampus_cobra_{i}_{side_code}_0000.nii.gz")
        lbl_out = os.path.join(TARGET_FOLDER, OUTPUT_LABELS_DIR, f"hippocampus_cobra_{i}_{side_code}.nii.gz")

        # Crop and save Image
        cropped_image_data = crop_and_save(img_data, affine, center, img_out)
        # Crop and save Labels
        cropped_label_data = crop_and_save(lbl_data, affine, center, lbl_out)

        # 2. Append metadata and paths to your list
        data_records.append({
            "subject_id": i,
            "side": side_code,
            "center_coords": center.tolist(),
            "image_data": cropped_image_data,
            "label_data": cropped_label_data
        })

print("\nDone! Files saved to imagesTr and labelsTr.")

# 3. Create DataFrame and save to PKL
df = pd.DataFrame(data_records)
pkl_filename = "cobra_hippocampus_full.pkl"
df.to_pickle(os.path.join(TARGET_FOLDER, pkl_filename), compression='gzip')
print(f"DataFrame saved to {pkl_filename}")

# save all into a private wandb to avoid re-downloading
wandb.init(project="hippopotamus-project", entity='hippopotamus')

artifact = wandb.Artifact("Dataset105_COBRA", type="dataset")
artifact.add_dir(TARGET_FOLDER)

wandb.log_artifact(artifact)
wandb.finish()
