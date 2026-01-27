import json
import argparse
import numpy as np
from sklearn.model_selection import GroupKFold
import re


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Original splits_final.json")
    parser.add_argument("-o", "--output", required=True, help="Corrected splits_final.json")
    parser.add_argument('-d', '--dataset', type=str, default='MNI', help='dataset codename')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        original_splits = json.load(f)

    # 1. Collect every unique case across all existing folds
    all_cases = set()
    for fold in original_splits:
        all_cases.update(fold['train'])
        all_cases.update(fold['val'])

    all_cases = sorted(list(all_cases))

    # 2. Map cases to their Patient Groups
    # Example: 's01' is the group for both 's01_L' and 's01_R'
    if args.dataset == 'MNI':
        groups = [get_patient_id_MNI(c) for c in all_cases]
    elif args.dataset == 'ADNI':
        groups = [get_patient_id_ADNI(c) for c in all_cases]
    elif args.dataset == 'COBRA':
        groups = [get_patient_id_COBRA(c) for c in all_cases]
    else:
        raise ValueError('unknown dataset!')

    # 3. Use GroupKFold to ensure each patient is in VAL exactly once across 5 folds
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)

    new_splits = []
    # We convert all_cases to a numpy array for easy indexing
    cases_array = np.array(all_cases)

    for train_idx, val_idx in gkf.split(all_cases, groups=groups):
        new_splits.append({
            "train": sorted(cases_array[train_idx].tolist()),
            "val": sorted(cases_array[val_idx].tolist())
        })

    # 4. Final verification
    for i, fold in enumerate(new_splits):
        train_p = set(get_patient_id_MNI(c) for c in fold['train'])
        val_p = set(get_patient_id_MNI(c) for c in fold['val'])
        overlap = train_p.intersection(val_p)

        print(f"Fold {i}: {len(fold['train'])} files (train), {len(fold['val'])} files (val).")
        if overlap:
            print(f"  !! WARNING: Leakage detected for patients: {overlap}")
        else:
            print(f"  ✓ No patient leakage.")

    with open(args.output, 'w') as f:
        json.dump(new_splits, f, indent=4)
    print(f"\nSaved 5-fold GroupKFold splits to {args.output}")


if __name__ == "__main__":
    main()
