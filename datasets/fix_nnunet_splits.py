import json
import argparse
import random
import os
import re


def get_patient_id(case_name):
    """
    Extracts patient ID.
    Logic: Looks for 's' followed by numbers (e.g., s01, s105).
    If your naming is different, you can adjust this regex.
    """
    match = re.search(r's\d+', case_name)
    if match:
        return match.group()
    # Fallback: split by underscores and take the first part that looks like a subject
    return case_name.split('_')[2] if len(case_name.split('_')) > 2 else case_name


def main():
    parser = argparse.ArgumentParser(description="Fix nnU-Net splits to ensure patient-level stratification.")
    parser.add_argument("-i", "--input", required=True, help="Path to the original splits_final.json")
    parser.add_argument("-o", "--output", required=True, help="Path to save the corrected splits_final.json")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("-p", "--ratio", type=float, default=0.8, help="Train/Val split ratio (default 0.8)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return

    with open(args.input, 'r') as f:
        splits = json.load(f)

    corrected_splits = []
    random.seed(args.seed)

    for i, fold in enumerate(splits):
        # Combine all cases to get the total population for this fold
        all_cases = list(set(fold['train'] + fold['val']))

        # Group cases by Patient ID
        patient_map = {}
        for case in all_cases:
            pid = get_patient_id(case)
            if pid not in patient_map:
                patient_map[pid] = []
            patient_map[pid].append(case)

        unique_patients = list(patient_map.keys())
        random.shuffle(unique_patients)

        # Split at patient level
        split_idx = int(len(unique_patients) * args.ratio)
        train_p = unique_patients[:split_idx]
        val_p = unique_patients[split_idx:]

        # Reconstruct the case lists
        new_train = [case for p in train_p for case in patient_map[p]]
        new_val = [case for p in val_p for case in patient_map[p]]

        corrected_splits.append({
            "train": sorted(new_train),
            "val": sorted(new_val)
        })

        # Verification check
        overlap = set(train_p).intersection(set(val_p))
        print(f"Fold {i}: Train Patients: {len(train_p)} | Val Patients: {len(val_p)} | Leakage: {len(overlap)}")

    with open(args.output, 'w') as f:
        json.dump(corrected_splits, f, indent=4)

    print(f"\nSuccessfully saved corrected splits to {args.output}")


if __name__ == "__main__":
    main()