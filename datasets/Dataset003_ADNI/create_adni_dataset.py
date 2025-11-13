# let's gather everything i did for Dataset003_ADNI
import nibabel as nib
import numpy as np
import glob

import pandas as pd
import wget
import zipfile
import os
from tqdm import tqdm


def get_all_labels(dir, name='', half=44):
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

    print(f"Cropping indices: x({min_x},{max_x}), y({min_y},{max_y}), z({min_z},{max_z})")

    if max_x - min_x > 2 * half or max_y - min_y > 2 * half or max_z - min_z > 2 * half:
        raise ValueError('Uhmmmm we may have a problem')

    # calculate mid points
    avg_x = (min_x + max_x) // 2
    avg_y = (min_y + max_y) // 2
    avg_z = (min_z + max_z) // 2

    aggreg = aggreg[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1].copy()

    # save to disk a numpy file of all crops
    import pandas as pd

    files = [f for f in files if 'CSF' not in f]  # exclude CSF images

    data_list = []
    crop_idxs = []
    for f in files:
        img = nib.load(f)
        data = img.get_fdata()
        # cropped = data[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]
        crop_idx = ((avg_x - half, avg_x + half),
                    (avg_y - half, avg_y + half),
                    (avg_z - half, avg_z + half))

        cropped = data[avg_x - half:avg_x + half,
                  avg_y - half:avg_y + half,
                  avg_z - half:avg_z + half].copy()
        data_list.append(cropped)
        crop_idxs.append(crop_idx)

    # a filename has this structure:
    # ADNI_nnn_S_nnnn_xxxxx_D.nii
    # the subject id is nnn_S_nnnn
    direction = 'R' if '_R.nii' in name else 'L'
    metadata = [os.path.basename(f).split('_' + direction + '.nii')[0] for f in files]
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
            image_data_list.append(cropped_image_data)

    df['image_data'] = image_data_list
    return df


if __name__ == '__main__':
    # check if file exists
    if not os.path.exists('adni_hippocampus_labels.pkl'):
        if not os.path.exists('Released_data_NII_v1.3.zip'):
            wget.download('http://hippocampal-protocol.net/SOPs/LINK_PAGE/FINAL_RELEASE/Released_data_NII_v1.3.zip')
            with zipfile.ZipFile('Released_data_NII_v1.3.zip', 'r') as zip_ref:
                zip_ref.extractall('./adni_data/')

        df_L = get_all_labels(dir='adni_data/Labels*/', name='*_L.nii', half=42)
        df_R = get_all_labels(dir='adni_data/Labels*/', name='*_R.nii', half=42)

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
    df.to_pickle('adni_hippocampus_full.pkl', compression="gzip")

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
