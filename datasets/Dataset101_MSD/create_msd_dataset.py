import os
import sys
import tarfile
import nibabel as nib
import pandas as pd
import wget
import wandb

TARGET_FOLDER = sys.argv[1] if len(sys.argv) > 1 else 'Dataset105_COBRA'

# makedir
os.makedirs(TARGET_FOLDER, exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(TARGET_FOLDER, 'labelsTr'), exist_ok=True)

wget.download('https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar')

file = tarfile.open('Task04_Hippocampus.tar')
file.extractall('./')

# move imagesTr, imagesTs, labelsTr to the upper folder
import shutil

shutil.move('Task04_Hippocampus/imagesTr', './imagesTr')
shutil.move('Task04_Hippocampus/imagesTs', './imagesTs')
shutil.move('Task04_Hippocampus/labelsTr', './labelsTr')

shutil.rmtree('Task04_Hippocampus')

# rename each file in imagesTr and imagesTs to nnUNet format
import os

for folder in ['imagesTr', 'imagesTs']:
    for filename in os.listdir(folder):
        if filename.endswith('.nii.gz'):
            new_filename = filename.replace('.nii.gz', '_0000.nii.gz')
            os.rename(os.path.join(TARGET_FOLDER, folder, filename), os.path.join(TARGET_FOLDER, folder, new_filename))

# 1. Prepare to collect data
records = []
# We focus on imagesTr because those have corresponding labelsTr
train_image_dir = os.path.join(TARGET_FOLDER, 'imagesTr')
train_label_dir = os.path.join(TARGET_FOLDER, 'labelsTr')

print("Reading files into DataFrame...")

# 2. Iterate through the training images
for filename in os.listdir(train_image_dir):
    if filename.endswith('_0000.nii.gz'):
        # Construct paths
        # Image: hippocampus_001_0000.nii.gz -> Label: hippocampus_001.nii.gz
        subject_id = filename.replace('_0000.nii.gz', '')
        img_path = os.path.join(train_image_dir, filename)
        lbl_path = os.path.join(train_label_dir, f"{subject_id}.nii.gz")

        if os.path.exists(lbl_path):
            # Load the actual data arrays
            img_data = nib.load(img_path).get_fdata()
            lbl_data = nib.load(lbl_path).get_fdata()

            records.append({
                "subject_id": subject_id,
                "image_data": img_data,
                "label_data": lbl_data,
                "dims": img_data.shape
            })

# 3. Create DataFrame and Save
df_msd = pd.DataFrame(records)
msd_pkl_path = "msd_hippocampus_full.pkl"
df_msd.to_pickle(os.path.join(TARGET_FOLDER, msd_pkl_path), compression='gzip')

print(f"Done! Saved {len(df_msd)} subjects to {msd_pkl_path}")

# save all into a private wandb to avoid re-downloading
wandb.init(project="hippopotamus-project", entity='hippopotamus')

artifact = wandb.Artifact("Dataset101_MSD", type="dataset")
artifact.add_dir(TARGET_FOLDER)

wandb.log_artifact(artifact)
wandb.finish()
