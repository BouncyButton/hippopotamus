import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def infer_dataset_code(dataset_name: str) -> str:
    if "_" in dataset_name:
        return dataset_name.split("_", 1)[1].upper()
    return dataset_name.upper()


def patient_id(case_name: str, dataset_code: str) -> str:
    code = dataset_code.upper()
    if code == "MNI":
        m = re.search(r"s\d+", case_name)
        return m.group(0) if m else case_name
    if code == "ADNI":
        m = re.search(r"adni_\d+", case_name)
        return m.group(0) if m else case_name
    if code == "COBRA":
        m = re.search(r"cobra_\d+", case_name)
        return m.group(0) if m else case_name
    # MSD and unknown datasets are treated case-wise.
    return case_name


def load_cases_from_splits(path: Path) -> List[str]:
    with open(path, "r") as f:
        splits = json.load(f)
    all_cases = set()
    for fold in splits:
        all_cases.update(fold["train"])
        all_cases.update(fold["val"])
    return sorted(all_cases)


def create_holdout_split(
    cases: Sequence[str],
    dataset_code: str,
    test_size: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    case_array = np.array(sorted(cases))
    groups = np.array([patient_id(c, dataset_code) for c in case_array])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(case_array, groups=groups))
    train_cases = sorted(case_array[train_idx].tolist())
    test_cases = sorted(case_array[test_idx].tolist())
    return train_cases, test_cases


def create_cv_splits(train_cases: Sequence[str], dataset_code: str, n_splits: int, seed: int) -> List[Dict[str, List[str]]]:
    case_array = np.array(sorted(train_cases))
    groups = np.array([patient_id(c, dataset_code) for c in case_array])

    # Deterministic shuffle before GroupKFold to allow a tunable seed.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(case_array))
    case_array = case_array[perm]
    groups = groups[perm]

    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, val_idx in gkf.split(case_array, groups=groups):
        splits.append(
            OrderedDict(
                train=sorted(case_array[train_idx].tolist()),
                val=sorted(case_array[val_idx].tolist()),
            )
        )
    return splits


def verify_no_group_leak(train_cases: Sequence[str], test_cases: Sequence[str], dataset_code: str):
    train_groups = set(patient_id(c, dataset_code) for c in train_cases)
    test_groups = set(patient_id(c, dataset_code) for c in test_cases)
    overlap = sorted(train_groups.intersection(test_groups))
    if overlap:
        raise RuntimeError(f"Group leakage between train and test split: {overlap[:10]}")


def verify_cv_no_group_leak(cv_splits: Sequence[Dict[str, Sequence[str]]], dataset_code: str):
    for i, split in enumerate(cv_splits):
        train_groups = set(patient_id(c, dataset_code) for c in split["train"])
        val_groups = set(patient_id(c, dataset_code) for c in split["val"])
        overlap = sorted(train_groups.intersection(val_groups))
        if overlap:
            raise RuntimeError(f"Group leakage in CV fold {i} train/val: {overlap[:10]}")


def verify_test_not_in_any_val(cv_splits: Sequence[Dict[str, Sequence[str]]], test_cases: Sequence[str], dataset_code: str):
    test_groups = set(patient_id(c, dataset_code) for c in test_cases)
    for i, split in enumerate(cv_splits):
        val_groups = set(patient_id(c, dataset_code) for c in split["val"])
        overlap = sorted(val_groups.intersection(test_groups))
        if overlap:
            raise RuntimeError(f"Test patient(s) found in CV val fold {i}: {overlap[:10]}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic train/test holdout plus 5-fold CV splits on the train pool only."
    )
    parser.add_argument("--dataset-dir", required=True, help="Dataset folder (e.g. datasets/Dataset103_ADNI)")
    parser.add_argument(
        "--input-splits",
        default="splits_final.json",
        help="Existing split file used only to collect all available case IDs.",
    )
    parser.add_argument("--dataset-code", default=None, help="Dataset code (MNI/ADNI/COBRA/MSD). Inferred if omitted.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test proportion (group-aware).")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds on the train pool.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument(
        "--holdout-output",
        default="train_test_split.json",
        help="Output file for holdout split inside --dataset-dir.",
    )
    parser.add_argument(
        "--cv-output",
        default="splits_final_train.json",
        help="Output file for CV folds (train-only) inside --dataset-dir.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    input_splits = dataset_dir / args.input_splits
    if not input_splits.exists():
        raise FileNotFoundError(f"Missing input split file: {input_splits}")

    dataset_code = args.dataset_code.upper() if args.dataset_code else infer_dataset_code(dataset_dir.name)
    all_cases = load_cases_from_splits(input_splits)
    train_cases, test_cases = create_holdout_split(all_cases, dataset_code, args.test_size, args.seed)
    verify_no_group_leak(train_cases, test_cases, dataset_code)
    cv_splits = create_cv_splits(train_cases, dataset_code, args.n_folds, args.seed)
    verify_cv_no_group_leak(cv_splits, dataset_code)
    verify_test_not_in_any_val(cv_splits, test_cases, dataset_code)

    holdout_payload = OrderedDict(train=train_cases, test=test_cases, seed=args.seed, test_size=args.test_size)
    cv_payload = cv_splits

    holdout_output = dataset_dir / args.holdout_output
    cv_output = dataset_dir / args.cv_output
    with open(holdout_output, "w") as f:
        json.dump(holdout_payload, f, indent=2)
    with open(cv_output, "w") as f:
        json.dump(cv_payload, f, indent=2)

    print(f"All cases: {len(all_cases)}")
    print(f"Train cases: {len(train_cases)}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Wrote holdout split: {holdout_output}")
    print(f"Wrote CV splits: {cv_output}")


if __name__ == "__main__":
    main()
