import tarfile
import nibabel as nib
import numpy as np
import os

import wget

url = 'https://cobralab.net/files/brains_t2.tar.bz2'

wget.download(url)

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

    # Update affine to reflect the new origin (translation)
    # The new origin is the old origin + the start index in world space
    new_affine = affine.copy()
    new_affine[:3, 3] = nib.affines.apply_affine(affine, start)

    new_img = nib.Nifti1Image(padded_crop.astype(data.dtype), new_affine)
    nib.save(new_img, output_path)


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

    for side_code, labels in [("L", LEFT_LABELS), ("R", RIGHT_LABELS)]:
        # Find the center of the specific hippocampus
        mask = np.isin(lbl_data, labels)
        coords = np.argwhere(mask)

        if coords.size == 0:
            print(f"  Warning: No {side_code} hippocampus labels in brain{i}")
            continue

        # Calculate geometric center of the labels
        center = (coords.min(axis=0) + coords.max(axis=0)) / 2

        # Prepare filenames
        # Image: hippocampus_cobra_ID_SIDE_0000.nii.gz
        # Label: hippocampus_cobra_ID_SIDE.nii.gz
        img_out = os.path.join(OUTPUT_IMAGES_DIR, f"hippocampus_cobra_{i}_{side_code}_0000.nii.gz")
        lbl_out = os.path.join(OUTPUT_LABELS_DIR, f"hippocampus_cobra_{i}_{side_code}.nii.gz")

        # Crop and save Image
        crop_and_save(img_data, affine, center, img_out)
        # Crop and save Labels
        crop_and_save(lbl_data, affine, center, lbl_out)

print("\nDone! Files saved to imagesTr and labelsTr.")