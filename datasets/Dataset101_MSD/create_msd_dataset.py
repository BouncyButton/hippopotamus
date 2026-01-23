import os
import sys
import tarfile
import nibabel as nib
import pandas as pd
import wget
import shutil
import wandb

import argparse

# 1. Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='Dataset101_MSD', help='Target folder name')
parser.add_argument('--rebuild', action='store_true', help='Force download from MSD and rebuild pkl',
                    default=False)
args = parser.parse_args()

TARGET_FOLDER = args.target
WANDB_PROJECT = "hippopotamus-project"
WANDB_ENTITY = "hippopotamus"
ARTIFACT_NAME = f"{WANDB_ENTITY}/{WANDB_PROJECT}/Dataset101_MSD:latest"


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

# 1. Setup Folders
os.makedirs(os.path.join(TARGET_FOLDER, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'imagesTs'), exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'labelsTr'), exist_ok=True)

# 2. Download and Extract
url = 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar'
if not os.path.exists('Task04_Hippocampus.tar'):
    wget.download(url)

with tarfile.open('Task04_Hippocampus.tar') as file:
    file.extractall('./temp_extract')

# 3. Move files DIRECTLY into TARGET_FOLDER
# Note: MSD tar usually contains a folder named 'Task04_Hippocampus'
src_base = './temp_extract/Task04_Hippocampus'

for folder in ['imagesTr', 'imagesTs', 'labelsTr']:
    src_dir = os.path.join(src_base, folder)
    dst_dir = os.path.join(TARGET_FOLDER, folder)

    for f in os.listdir(src_dir):
        # Move and overwrite if exists
        shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

shutil.rmtree('./temp_extract')

# 4. Rename to nnUNet format (_0000)
# ONLY rename images. Labels MUST NOT have the _0000 suffix.
for folder in ['imagesTr', 'imagesTs']:
    folder_path = os.path.join(TARGET_FOLDER, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.nii.gz') and not filename.endswith('_0000.nii.gz'):
            new_name = filename.replace('.nii.gz', '_0000.nii.gz')
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))

# 5. Data Collection
records = []
train_image_dir = os.path.join(TARGET_FOLDER, 'imagesTr')
train_label_dir = os.path.join(TARGET_FOLDER, 'labelsTr')

print("Reading files into DataFrame...")
for filename in sorted(os.listdir(train_image_dir)):
    if filename.endswith('_0000.nii.gz'):
        # Correct mapping:
        # Image: hippocampus_001_0000.nii.gz
        # Label: hippocampus_001.nii.gz
        subject_id = filename.replace('_0000.nii.gz', '')
        img_path = os.path.join(train_image_dir, filename)
        lbl_path = os.path.join(train_label_dir, f"{subject_id}.nii.gz")

        if os.path.exists(lbl_path):
            img_obj = nib.load(img_path)
            img_data = img_obj.get_fdata()
            lbl_data = nib.load(lbl_path).get_fdata()

            records.append({
                "subject_id": subject_id,
                "image_data": img_data,
                "label_data": lbl_data,
                "dims": img_data.shape
            })

df_msd = pd.DataFrame(records)
if not df_msd.empty:
    msd_pkl_path = os.path.join(TARGET_FOLDER, "msd_hippocampus_full.pkl")
    df_msd.to_pickle(msd_pkl_path, compression='gzip')
    print(f"Done! Saved {len(df_msd)} subjects.")
else:
    print("DataFrame is still empty. Check if labels exist in labelsTr.")

# 6. W&B Logging
# save all into a private wandb to avoid re-downloading
wandb.init(project="hippopotamus-project", entity='hippopotamus')

artifact = wandb.Artifact("Dataset101_MSD", type="dataset")
artifact.add_dir(TARGET_FOLDER)

wandb.log_artifact(artifact)
wandb.finish()
