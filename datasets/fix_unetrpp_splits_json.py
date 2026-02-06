import argparse
import json
import os
import re

import numpy as np
from sklearn.model_selection import GroupKFold


def get_patient_id_MNI(case_name):
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
    match = re.search(r'_(MNI|ADNI|COBRA)\b', base, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def get_patient_id(case_name, dataset):
    if dataset == "MNI":
        return get_patient_id_MNI(case_name)
    if dataset == "ADNI":
        return get_patient_id_ADNI(case_name)
    if dataset == "COBRA":
        return get_patient_id_COBRA(case_name)
    raise ValueError("unknown dataset!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input JSON path")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    parser.add_argument("-d", "--dataset", default=None, help="dataset codename (MNI, ADNI, COBRA)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for shuffle")

    args = parser.parse_args()

    with open(args.input, "r") as f:
        payload = json.load(f)

    original_splits = payload["splits"]

    all_cases = set()
    for fold in original_splits:
        all_cases.update(fold["train"])
        all_cases.update(fold["val"])

    all_cases = sorted(list(all_cases))

    dataset = args.dataset.upper() if args.dataset else infer_dataset_from_path(args.output)
    if dataset is None:
        raise ValueError("Could not infer dataset from output path. Please provide --dataset.")

    groups = [get_patient_id(c, dataset) for c in all_cases]

    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=args.seed)

    new_splits = []
    cases_array = np.array(all_cases)
    groups_array = np.array(groups)
    for train_idx, val_idx in gkf.split(cases_array, groups=groups_array):
        new_splits.append({
            "train": sorted(cases_array[train_idx].tolist()),
            "val": sorted(cases_array[val_idx].tolist()),
        })

    for i, fold in enumerate(new_splits):
        train_p = set(get_patient_id(c, dataset) for c in fold["train"])
        val_p = set(get_patient_id(c, dataset) for c in fold["val"])
        overlap = train_p.intersection(val_p)

        print(f"Fold {i}: {len(fold['train'])} files (train), {len(fold['val'])} files (val).")
        if overlap:
            print(f"  !! WARNING: Leakage detected for patients: {overlap}")
        else:
            print("  ✓ No patient leakage.")

    with open(args.output, "w") as f:
        json.dump({"splits": new_splits}, f, indent=2)
    print(f"Saved JSON splits to {args.output}")


if __name__ == "__main__":
    main()
