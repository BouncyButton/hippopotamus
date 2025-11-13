import wget
import tarfile

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
            os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))