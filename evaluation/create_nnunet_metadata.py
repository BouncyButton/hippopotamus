import argparse
import json
import os
from pathlib import Path
import shutil

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)  # e.g. Dataset102_MNI
    parser.add_argument("--dataset-code", type=str, required=True)  # e.g. MNI
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--project", type=str, default="hippopotamus-project")
    parser.add_argument("--entity", type=str, default="hippopotamus")
    parser.add_argument(
        "--run-preprocess",
        action="store_true",
        help="Run nnUNetv2_plan_and_preprocess before packaging metadata",
    )
    parser.add_argument(
        "--holdout-split-name",
        type=str,
        default="train_test_split.json",
        help="If this holdout file exists for the dataset, preprocessing is blocked to avoid train/test leakage.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    datasets_dir = root / "datasets"
    preproc_dir = root / "preprocessed"
    results_dir = root / "results"

    os.environ["nnUNet_raw"] = str(datasets_dir)
    os.environ["nnUNet_preprocessed"] = str(preproc_dir)
    os.environ["nnUNet_results"] = str(results_dir)

    if args.run_preprocess:
        holdout_split = datasets_dir / args.dataset_name / args.holdout_split_name
        if holdout_split.exists():
            raise RuntimeError(
                f"Refusing to preprocess {args.dataset_name}: found holdout split {holdout_split}. "
                "Create/use a train-only dataset folder for nnUNet preprocessing."
            )
        # run preprocessing to generate plans and splits
        import subprocess
        subprocess.check_call([
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(args.dataset_id),
            "--verify_dataset_integrity",
        ])

        # splits are assumed to be already fixed/supplied

    # collect metadata files
    bundle_dir = root / "evaluation" / "nnunet_metadata" / args.dataset_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # nnUNet plans
    plans_file = preproc_dir / args.dataset_name / "nnUNetPlans.json"
    if plans_file.exists():
        shutil.copy(plans_file, bundle_dir / "nnUNetPlans.json")

    # splits
    splits_file = preproc_dir / args.dataset_name / "splits_final.json"
    if splits_file.exists():
        shutil.copy(splits_file, bundle_dir / "splits_final.json")

    # dataset.json (prefer preprocessed if present, else datasets)
    dataset_json = preproc_dir / args.dataset_name / "dataset.json"
    if not dataset_json.exists():
        dataset_json = datasets_dir / args.dataset_name / "dataset.json"
    if dataset_json.exists():
        shutil.copy(dataset_json, bundle_dir / "dataset.json")

    # log artifact
    run = wandb.init(project=args.project, entity=args.entity)
    artifact = wandb.Artifact(
        name=f"nnunet-metadata-{args.dataset_code}",
        type="metadata",
    )
    artifact.add_dir(str(bundle_dir))
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
