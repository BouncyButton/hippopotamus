# set env variables
import os

absolute_path_of_datasets = os.path.abspath("../../nnUnet_raw/")
absolute_path_of_preprocessed = os.path.abspath("../../preprocessed/nnUnet/")
absolute_path_of_results = os.path.abspath("../../results/nnUnet/")

os.environ["nnUNet_raw"] = absolute_path_of_datasets
os.environ["nnUNet_preprocessed"] = absolute_path_of_preprocessed
os.environ["nnUNet_results"] = absolute_path_of_results

# run preproc
os.system("nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity")

# run training
os.system("nnUNetv2_train 101 3d_fullres")
