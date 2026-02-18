import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import nibabel as nib


def _load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata())


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    oh = torch.nn.functional.one_hot(labels.long(), num_classes=num_classes)
    return oh.permute(0, 4, 1, 2, 3).float()


def compute_metrics_hard_soft(probs: torch.Tensor, labels: torch.Tensor, num_classes: int):
    hard = torch.argmax(probs, dim=1)
    y_true = one_hot(labels, num_classes)
    y_hard = one_hot(hard, num_classes)

    y_true_fg = y_true[:, 1:]
    y_hard_fg = y_hard[:, 1:]
    probs_fg = probs[:, 1:]

    tp = (y_hard_fg * y_true_fg).sum(dim=(0, 2, 3, 4))
    fp = (y_hard_fg * (1 - y_true_fg)).sum(dim=(0, 2, 3, 4))
    fn = ((1 - y_hard_fg) * y_true_fg).sum(dim=(0, 2, 3, 4))

    dice_hard = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()
    iou = (tp / (tp + fp + fn + 1e-8)).mean().item()

    soft_tp = (probs_fg * y_true_fg).sum(dim=(0, 2, 3, 4))
    soft_dice = (2 * soft_tp / (probs_fg.sum(dim=(0, 2, 3, 4)) + y_true_fg.sum(dim=(0, 2, 3, 4)) + 1e-8)).mean().item()

    recall = (tp / (tp + fn + 1e-8)).mean().item()
    precision = (tp / (tp + fp + 1e-8)).mean().item()
    acc = (hard == labels.squeeze(1)).float().mean().item()

    return {
        "dice_hard": dice_hard,
        "dice_soft": soft_dice,
        "iou": iou,
        "jaccard": iou,
        "recall": recall,
        "precision": precision,
        "accuracy": acc,
    }


def compute_hd95(hard: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    from monai.metrics import HausdorffDistanceMetric

    metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=True)
    y_true = one_hot(labels, num_classes)
    y_hard = one_hot(hard, num_classes)
    metric(y_hard, y_true)
    hd, _ = metric.aggregate()
    return float(hd.cpu().item())


def _checkpoint_name(kind: str) -> str:
    if kind == "best":
        return "model_best"
    if kind == "latest":
        return "model_latest"
    return "model_final_checkpoint"


def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _resolve_splits_json(dataset_root: Path, dataset_name: str, explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    candidates = [
        Path("datasets") / dataset_name / "splits_final_train.json",
        dataset_root / dataset_name / "splits_final_train.json",
        Path("datasets") / dataset_name / "splits_final.json",
        dataset_root / dataset_name / "splits_final.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing CV splits for {dataset_name}")


def _resolve_holdout_json(dataset_root: Path, dataset_name: str, explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    candidates = [
        Path("datasets") / dataset_name / "train_test_split.json",
        dataset_root / dataset_name / "train_test_split.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing holdout split file for {dataset_name}")


def _prepare_model_dir(model_root: Path, folds: List[int]) -> Path:
    # If the folder already has fold subdirs, use it directly.
    if all((model_root / f"fold_{f}").exists() for f in folds):
        return model_root

    # Backward-compatible path: a single-fold artifact with checkpoint files at root.
    if len(folds) != 1:
        raise RuntimeError(
            "Model root does not contain fold_* subdirs; only single-fold mode is possible for this artifact."
        )

    model_dir = model_root / "unetrpp_model"
    fold_dir = model_dir / f"fold_{folds[0]}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    for item in model_root.iterdir():
        if item.is_file():
            shutil.copy(item, fold_dir / item.name)
    for item in model_root.iterdir():
        if item.is_file() and item.name.endswith(".pkl"):
            shutil.copy(item, model_dir / item.name)
    return model_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--fold", type=int, default=None, help="Backward-compatible single fold.")
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Model folds to ensemble.")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--splits-json", default=None, help="CV split file; defaults to splits_final_train.json")
    parser.add_argument("--holdout-json", default=None, help="Holdout split file; defaults to train_test_split.json")
    parser.add_argument("--checkpoint", default="final", choices=["latest", "best", "final"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    unetrpp_root = repo_root / "baselines" / "unetr_plus_plus"
    if str(unetrpp_root) not in sys.path:
        sys.path.insert(0, str(unetrpp_root))

    from unetr_pp.inference.predict import predict_cases

    model_root = Path(args.model_root)
    dataset_root = Path(args.dataset_root)
    dataset_name = args.dataset_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = args.folds if args.folds is not None else ([] if args.fold is None else [args.fold])
    if not folds:
        folds = [0, 1, 2, 3, 4]

    if args.eval_split == "test":
        holdout_json = _resolve_holdout_json(dataset_root, dataset_name, args.holdout_json)
        holdout = _read_json(holdout_json)
        eval_cases = holdout["test"]
    else:
        splits_json = _resolve_splits_json(dataset_root, dataset_name, args.splits_json)
        splits = _read_json(splits_json)
        ref_fold = folds[0]
        if ref_fold >= len(splits):
            raise ValueError(f"Fold {ref_fold} not available")
        eval_cases = splits[ref_fold]["val"]

    if args.max_cases is not None:
        eval_cases = eval_cases[:args.max_cases]

    # prefer nnUNet_raw if set
    nnunet_raw = os.environ.get("nnUNet_raw")
    if nnunet_raw:
        images_dir = Path(nnunet_raw) / dataset_name / "imagesTr"
        labels_dir = Path(nnunet_raw) / dataset_name / "labelsTr"
    else:
        images_dir = dataset_root / dataset_name / "imagesTr"
        labels_dir = dataset_root / dataset_name / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        images_dir = Path("datasets") / dataset_name / "imagesTr"
        labels_dir = Path("datasets") / dataset_name / "labelsTr"

    model_dir = _prepare_model_dir(model_root, folds)

    input_lists = []
    output_files = []
    for case in eval_cases:
        img = images_dir / f"{case}_0000.nii.gz"
        if not img.exists():
            continue
        input_lists.append([str(img)])
        output_files.append(str(output_dir / f"{case}.nii.gz"))

    if len(input_lists) == 0:
        raise RuntimeError("No valid cases to predict")

    predict_cases(
        str(model_dir),
        input_lists,
        output_files,
        folds=tuple(folds),
        save_npz=True,
        num_threads_preprocessing=2,
        num_threads_nifti_save=2,
        do_tta=False,
        mixed_precision=True,
        overwrite_existing=True,
        all_in_gpu=False,
        step_size=0.5,
        checkpoint_name=_checkpoint_name(args.checkpoint),
    )

    metrics_accum = []
    hd_values = []
    n_cases = 0
    for case in eval_cases:
        prob_file = output_dir / f"{case}.npz"
        label_file = labels_dir / f"{case}.nii.gz"
        if not prob_file.exists() or not label_file.exists():
            continue
        probs = np.load(prob_file)["probabilities"]
        labels = _load_nifti(label_file).astype(np.int64)
        probs_t = torch.from_numpy(probs).unsqueeze(0)
        labels_t = torch.from_numpy(labels).unsqueeze(0).unsqueeze(0)
        metrics_accum.append(compute_metrics_hard_soft(probs_t, labels_t, probs.shape[0]))
        hard = torch.argmax(probs_t, dim=1)
        hd_values.append(compute_hd95(hard, labels_t, probs.shape[0]))
        n_cases += 1

    if n_cases == 0:
        raise RuntimeError("No valid predictions to score")
    agg = {k: float(np.mean([m[k] for m in metrics_accum])) for k in metrics_accum[0].keys()}
    agg["hd95"] = float(np.mean(hd_values))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()
