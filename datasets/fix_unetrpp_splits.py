import argparse
import os
import pickle
import re
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import GroupKFold


def get_patient_id_MNI(case_name):
    # Logic: extract 'sXX' from filename
    match = re.search(r's\d+', case_name)
    return match.group() if match else case_name


def get_patient_id_ADNI(case_name):
    match = re.search(r'adni_\d+', case_name)
    return match.group() if match else case_name


def get_patient_id_COBRA(case_name):
    match = re.search(r'cobra_\d+', case_name)
    return match.group() if match else case_name


def infer_dataset_from_path(path):
    base = os.path.basename(os.path.dirname(path))
    # Expected: Task###_MNI, Task###_ADNI, Task###_COBRA, etc.
    match = re.search(r'_(MNI|ADNI|COBRA)\b', base, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def get_patient_id(case_name, dataset):
    if dataset == 'MNI':
        return get_patient_id_MNI(case_name)
    if dataset == 'ADNI':
        return get_patient_id_ADNI(case_name)
    if dataset == 'COBRA':
        return get_patient_id_COBRA(case_name)
    raise ValueError('unknown dataset!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Original splits_final.pkl")
    parser.add_argument("-o", "--output", default=None, help="Corrected splits_final.pkl (default: overwrite input)")
    parser.add_argument('-d', '--dataset', type=str, default=None, help='dataset codename (MNI, ADNI, COBRA)')
    args = parser.parse_args()

    output_path = args.output or args.input
    dataset = args.dataset.upper() if args.dataset else infer_dataset_from_path(args.input)
    if dataset is None:
        raise ValueError("Could not infer dataset from path. Please provide --dataset (MNI, ADNI, COBRA).")

    with open(args.input, 'rb') as f:
        original_splits = pickle.load(f)

    # 1. Collect every unique case across all existing folds
    all_cases = set()
    for fold in original_splits:
        all_cases.update(fold['train'])
        all_cases.update(fold['val'])

    all_cases = sorted(list(all_cases))

    # 2. Map cases to their Patient Groups
    groups = [get_patient_id(c, dataset) for c in all_cases]

    # 3. Use GroupKFold to ensure each patient is in VAL exactly once across 5 folds
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)

    new_splits = []
    cases_array = np.array(all_cases)

    for train_idx, val_idx in gkf.split(all_cases, groups=groups):
        new_splits.append(OrderedDict({
            "train": np.array(sorted(cases_array[train_idx].tolist())),
            "val": np.array(sorted(cases_array[val_idx].tolist())),
        }))

    # 4. Final verification
    for i, fold in enumerate(new_splits):
        train_p = set(get_patient_id(c, dataset) for c in fold['train'])
        val_p = set(get_patient_id(c, dataset) for c in fold['val'])
        overlap = train_p.intersection(val_p)

        print(f"Fold {i}: {len(fold['train'])} files (train), {len(fold['val'])} files (val).")
        if overlap:
            print(f"  !! WARNING: Leakage detected for patients: {overlap}")
        else:
            print("  ✓ No patient leakage.")

    with open(output_path, 'wb') as f:
        pickle.dump(new_splits, f)
    print(f"\nSaved 5-fold GroupKFold splits to {output_path}")


if __name__ == "__main__":
    main()
