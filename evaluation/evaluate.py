import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
import nibabel as nib
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import Compose, EnsureChannelFirstd, NormalizeIntensityd, EnsureTyped, DivisiblePadd
from monai.data import Dataset as MonaiDataset, DataLoader


def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


DATASET_ARTIFACTS = {
    "MSD": ("Dataset101_MSD", "msd_hippocampus_full.pkl"),
    "MNI": ("Dataset102_MNI", "mni_hippocampus_full.pkl"),
    "ADNI": ("Dataset103_ADNI", "adni_hippocampus_full.pkl"),
    "COBRA": ("Dataset105_COBRA", "cobra_hippocampus_full.pkl"),
}

MODEL_ARTIFACTS = {
    "nnunet": "nnunet-model-{dataset}-fold{fold}",
    "unetrpp": "unetrpp-model-{dataset}-fold{fold}",
    "swinunetr": "swinunetr-model-{dataset}-fold{fold}",
}


@dataclass
class EvalConfig:
    project: str
    entity: str
    methods: List[str]
    datasets: List[str]
    folds: List[int]
    checkpoint: str
    output_dir: Path
    use_wandb: bool
    batch_size: int
    repo_root: str


@dataclass
class FoldResult:
    metrics: Dict[str, float]
    n_cases: int


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_artifact(api: wandb.Api, name: str, dst: Path) -> Path:
    artifact = api.artifact(name)
    ensure_dir(dst)
    return Path(artifact.download(root=str(dst)))


def load_dataset_from_artifact(dataset_code: str, dataset_root: Path) -> MonaiDataset:
    artifact_name, pkl_name = DATASET_ARTIFACTS[dataset_code]
    pkl_path = dataset_root / pkl_name
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing pkl: {pkl_path}")

    import pandas as pd

    df = pd.read_pickle(pkl_path, compression="gzip")
    if dataset_code in {"MSD", "COBRA"}:
        image_key = "image_data"
        label_key = "label_data"
    else:
        image_key = "image_data"
        label_key = "data"

    records = []
    for _, row in df.iterrows():
        records.append({"image": row[image_key], "label": row[label_key]})

    transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        DivisiblePadd(keys=["image", "label"], k=32, mode=("constant", "constant")),
        EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.long)),
    ])
    return MonaiDataset(data=records, transform=transforms)


def infer_num_classes(dataset: MonaiDataset) -> int:
    max_label = 0
    for item in dataset.data:
        arr = np.asarray(item["label"])
        if arr.size == 0:
            continue
        max_label = max(max_label, int(np.nanmax(arr)))
    return max_label + 1


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    oh = torch.nn.functional.one_hot(labels.long(), num_classes=num_classes)
    return oh.permute(0, 4, 1, 2, 3).float()


def compute_metrics_hard_soft(
        probs: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
) -> Dict[str, float]:
    # probs: (B, C, ...), labels: (B, 1, ...) or (B, ...)
    with torch.no_grad():
        hard = torch.argmax(probs, dim=1)
        y_true = one_hot(labels, num_classes)
        y_hard = one_hot(hard, num_classes)

        # exclude background
        y_true_fg = y_true[:, 1:]
        y_hard_fg = y_hard[:, 1:]
        probs_fg = probs[:, 1:]

        tp = (y_hard_fg * y_true_fg).sum(dim=(0, 2, 3, 4))
        fp = (y_hard_fg * (1 - y_true_fg)).sum(dim=(0, 2, 3, 4))
        fn = ((1 - y_hard_fg) * y_true_fg).sum(dim=(0, 2, 3, 4))

        dice_hard = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()
        iou = (tp / (tp + fp + fn + 1e-8)).mean().item()

        # soft dice
        soft_tp = (probs_fg * y_true_fg).sum(dim=(0, 2, 3, 4))
        soft_dice = (2 * soft_tp / (
                    probs_fg.sum(dim=(0, 2, 3, 4)) + y_true_fg.sum(dim=(0, 2, 3, 4)) + 1e-8)).mean().item()

        recall = (tp / (tp + fn + 1e-8)).mean().item()
        precision = (tp / (tp + fp + 1e-8)).mean().item()

        total = y_true.numel() / num_classes
        # overall accuracy
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
    metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=True)
    y_true = one_hot(labels, num_classes)
    y_hard = one_hot(hard, num_classes)
    metric(y_hard, y_true)
    hd, _ = metric.aggregate()
    return float(hd.cpu().item())


def evaluate_swinunetr(
        model_path: Path,
        dataset: MonaiDataset,
        num_classes: int,
        device: torch.device,
        batch_size: int,
        pred_dir: Optional[Path] = None,
        max_cases: Optional[int] = None,
) -> FoldResult:
    from monai.networks.nets import SwinUNETR

    model = SwinUNETR(in_channels=1, out_channels=num_classes, use_checkpoint=False).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    metrics_accum = []
    hd_values = []
    n_cases = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            probs = torch.softmax(model(images), dim=1)
            metrics_accum.append(compute_metrics_hard_soft(probs, labels, num_classes))
            hard = torch.argmax(probs, dim=1)
            hd_values.append(compute_hd95(hard, labels, num_classes))
            batch_size_actual = images.shape[0]
            if pred_dir is not None:
                for i in range(batch_size_actual):
                    idx = n_cases + i
                    np.save(pred_dir / f"case_{idx:04d}_pred.npy", hard[i].cpu().numpy())
                    np.save(pred_dir / f"case_{idx:04d}_prob.npy", probs[i].cpu().numpy())
            n_cases += batch_size_actual
            if max_cases is not None and n_cases >= max_cases:
                break

    # aggregate
    agg = {k: float(np.mean([m[k] for m in metrics_accum])) for k in metrics_accum[0].keys()}
    agg["hd95"] = float(np.mean(hd_values))
    return FoldResult(metrics=agg, n_cases=n_cases)


def _load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata())


def _checkpoint_name_nnUNet(kind: str) -> str:
    if kind == "best":
        return "checkpoint_best.pth"
    if kind == "latest":
        return "checkpoint_latest.pth"
    return "checkpoint_final.pth"


def evaluate_nnunet(
        model_root: Path,
        metadata_root: Path,
        repo_root: Path,
        dataset_root: Path,
        dataset_name: str,
        fold: int,
        max_cases: Optional[int],
        checkpoint_kind: str,
        device: torch.device,
        output_dir: Path,
) -> Optional[FoldResult]:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    dataset_json = metadata_root / "dataset.json"
    plans_json = metadata_root / "nnUNetPlans.json"
    splits_json = metadata_root / "splits_final.json"

    if not splits_json.exists():
        splits_json = repo_root / Path("datasets") / dataset_name / "splits_final.json"
    if not dataset_json.exists() or not plans_json.exists() or not splits_json.exists():
        print(f"Missing metadata files in {metadata_root}")
        print(f"Expected dataset.json at: {dataset_json}")
        print(f"Expected nnUNetPlans.json at: {plans_json}")
        print(f"Expected splits_final.json at: {splits_json}")
        try:
            present = sorted([p.name for p in Path(metadata_root).iterdir()])
        except Exception:
            present = []
        print(f"Files present: {present}")
        return None

    with open(dataset_json, "r") as f:
        dataset_info = json.load(f)
    file_ending = dataset_info["file_ending"]

    with open(splits_json, "r") as f:
        splits = json.load(f)
    if fold >= len(splits):
        print(f"Fold {fold} not available in splits")
        return None
    val_cases = splits[fold]["val"]
    if max_cases is not None:
        val_cases = val_cases[:max_cases]

    # assemble model folder
    model_dir = model_root / "nnunet_model"
    fold_dir = model_dir / f"fold_{fold}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)

    # copy checkpoints into fold_dir
    for item in model_root.iterdir():
        if item.is_file():
            shutil.copy(item, fold_dir / item.name)

    # add required metadata files
    shutil.copy(plans_json, model_dir / "plans.json")
    shutil.copy(dataset_json, model_dir / "dataset.json")

    ckpt_name = _checkpoint_name_nnUNet(checkpoint_kind)
    if not (fold_dir / ckpt_name).exists():
        print(f"Missing checkpoint {ckpt_name} in {fold_dir}")
        return None

    predictor = nnUNetPredictor(device=device, verbose=False)
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=(fold,),
        checkpoint_name=ckpt_name,
    )

    # prefer real nnUNet_raw/datasets if available (e.g., /content/datasets)
    nnunet_raw = os.environ.get("nnUNet_raw")
    if nnunet_raw:
        images_dir = Path(nnunet_raw) / dataset_name / "imagesTr"
        labels_dir = Path(nnunet_raw) / dataset_name / "labelsTr"
    else:
        images_dir = dataset_root / dataset_name / "imagesTr"
        labels_dir = dataset_root / dataset_name / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        # fallback to local repo datasets/
        images_dir = Path("datasets") / dataset_name / "imagesTr"
        labels_dir = Path("datasets") / dataset_name / "labelsTr"
    input_dir = output_dir / "nnunet_inputs"
    pred_dir = output_dir / "predictions"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for case in val_cases:
        src = images_dir / f"{case}_0000{file_ending}"
        if not src.exists():
            print(f"Missing input image {src}")
            continue
        shutil.copy(src, input_dir / src.name)

    predictor.predict_from_files(
        str(input_dir),
        str(pred_dir),
        save_probabilities=True,
        overwrite=True,
    )

    metrics_accum = []
    hd_values = []
    n_cases = 0
    num_classes = int(np.max(_load_nifti(labels_dir / f"{val_cases[0]}{file_ending}"))) + 1

    for case in val_cases:
        pred_file = pred_dir / f"{case}{file_ending}"
        prob_file = pred_dir / f"{case}.npz"
        label_file = labels_dir / f"{case}{file_ending}"
        if not pred_file.exists() or not prob_file.exists() or not label_file.exists():
            continue
        probs = np.load(prob_file)["probabilities"]
        labels = _load_nifti(label_file).astype(np.int64)
        probs_t = torch.from_numpy(probs).unsqueeze(0)
        labels_t = torch.from_numpy(labels).unsqueeze(0).unsqueeze(0)
        # align prediction/label shapes via center-crop of the larger tensor
        if labels_t.shape[2:] != probs_t.shape[2:]:
            label_shape = labels_t.shape[2:]
            pred_shape = probs_t.shape[2:]

            # If label shape is a permutation of prediction shape, permute labels to match
            if sorted(label_shape) == sorted(pred_shape):
                import itertools

                for perm in itertools.permutations([0, 1, 2]):
                    if tuple(label_shape[p] for p in perm) == pred_shape:
                        labels_t = labels_t.permute(0, 1, 2 + perm[0], 2 + perm[1], 2 + perm[2])
                        label_shape = labels_t.shape[2:]
                        break

            # If still mismatched, center-crop both to the overlapping size
            if label_shape != pred_shape:
                target = tuple(min(label_shape[d], pred_shape[d]) for d in range(3))

                def _center_crop(t, target_shape):
                    in_shape = t.shape[2:]
                    start = [(in_shape[d] - target_shape[d]) // 2 for d in range(3)]
                    end = [start[d] + target_shape[d] for d in range(3)]
                    return t[:, :, start[0]:end[0], start[1]:end[1], start[2]:end[2]]

                if pred_shape != target:
                    probs_t = _center_crop(probs_t, target)
                if label_shape != target:
                    labels_t = _center_crop(labels_t, target)
        metrics_accum.append(compute_metrics_hard_soft(probs_t, labels_t, probs.shape[0]))
        hard = torch.argmax(probs_t, dim=1)
        hd_values.append(compute_hd95(hard, labels_t, probs.shape[0]))
        n_cases += 1

    if n_cases == 0:
        return None
    agg = {k: float(np.mean([m[k] for m in metrics_accum])) for k in metrics_accum[0].keys()}
    agg["hd95"] = float(np.mean(hd_values))
    return FoldResult(metrics=agg, n_cases=n_cases)


def save_predictions_as_artifact(run: wandb.sdk.wandb_run.Run, artifact_name: str, pred_dir: Path):
    artifact = wandb.Artifact(name=artifact_name, type="predictions")
    artifact.add_dir(str(pred_dir))
    run.log_artifact(artifact)


def latex_table(results: Dict[str, Dict[str, List[float]]], metric_name: str) -> str:
    datasets = sorted(next(iter(results.values())).keys())
    methods = sorted(results.keys())
    lines = []
    header = "Method & " + " & ".join(datasets) + " \\\\"
    lines.append("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append("\\hline")
    lines.append(header)
    lines.append("\\hline")
    for method in methods:
        row = [method]
        for ds in datasets:
            vals = results[method][ds]
            if not vals:
                row.append("N/A")
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                row.append(f"{mean:.4f}\\,\\scriptsize{{({std:.4f})}}")
        lines.append(" & ".join(row) + " \\\\ ")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    title = f"% Metric: {metric_name}"
    return "\n".join([title] + lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="hippopotamus-project")
    parser.add_argument("--entity", default="hippopotamus")
    parser.add_argument("--methods", nargs="+", default=["nnunet", "unetrpp", "swinunetr"])
    parser.add_argument("--datasets", nargs="+", default=["MSD", "MNI", "ADNI", "COBRA"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--checkpoint", choices=["latest", "best", "final"], default="final")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="evaluation/output")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of validation cases per fold")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--repo-root", type=str, default="/content/hippopotamus",
                        help="Root of the local repository (for nnUNet metadata fallback)")
    args = parser.parse_args()

    cfg = EvalConfig(
        project=args.project,
        entity=args.entity,
        methods=args.methods,
        datasets=args.datasets,
        folds=args.folds,
        checkpoint=args.checkpoint,
        output_dir=Path(args.output_dir),
        use_wandb=not args.no_wandb,
        batch_size=args.batch_size,
        repo_root=args.repo_root,
    )

    ensure_dir(cfg.output_dir)
    seed_everything(0)

    api = wandb.Api()
    wandb_run = None
    if cfg.use_wandb:
        wandb_run = wandb.init(project=cfg.project, entity=cfg.entity)

    results: Dict[str, Dict[str, List[float]]] = {m: {d: [] for d in cfg.datasets} for m in cfg.methods}

    for dataset_code in cfg.datasets:
        if dataset_code not in DATASET_ARTIFACTS:
            print(f"Unknown dataset: {dataset_code}")
            continue
        dataset_artifact_name, _ = DATASET_ARTIFACTS[dataset_code]
        dataset_artifact_id = f"{cfg.entity}/{cfg.project}/{dataset_artifact_name}:latest"
        dataset_root = cfg.output_dir / "artifacts" / dataset_artifact_name
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        dataset_path = download_artifact(api, dataset_artifact_id, dataset_root)
        dataset = load_dataset_from_artifact(dataset_code, dataset_root)
        num_classes = infer_num_classes(dataset)

        metadata_root = None
        metadata_artifact_id = f"{cfg.entity}/{cfg.project}/nnunet-metadata-{dataset_code}:latest"
        try:
            metadata_root = download_artifact(
                api,
                metadata_artifact_id,
                cfg.output_dir / "artifacts" / "nnunet_metadata" / dataset_code,
            )
        except Exception as e:
            metadata_root = None
            print(f"Missing nnUNet metadata artifact {metadata_artifact_id}: {e}")

        for method in cfg.methods:
            if method not in MODEL_ARTIFACTS:
                print(f"Unknown method: {method}")
                continue

            for fold in cfg.folds:
                model_artifact = MODEL_ARTIFACTS[method].format(dataset=dataset_code, fold=fold)
                model_artifact_id = f"{cfg.entity}/{cfg.project}/{model_artifact}:latest"
                model_root = cfg.output_dir / "artifacts" / method / dataset_code / f"fold{fold}"
                repo_root = cfg.repo_root
                if model_root.exists():
                    shutil.rmtree(model_root)

                try:
                    model_path = download_artifact(api, model_artifact_id, model_root)
                except Exception as e:
                    print(f"Missing model artifact {model_artifact_id}: {e}")
                    continue

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                if method == "swinunetr":
                    ckpt_file = model_path / "model.pt"
                    if not ckpt_file.exists():
                        print(f"Missing checkpoint: {ckpt_file}")
                        continue
                    pred_dir = cfg.output_dir / "predictions" / method / dataset_code / f"fold{fold}"
                    ensure_dir(pred_dir)
                    fold_result = evaluate_swinunetr(
                        ckpt_file,
                        dataset,
                        num_classes,
                        device,
                        cfg.batch_size,
                        pred_dir=pred_dir,
                        max_cases=args.max_cases,
                    )
                elif method == "nnunet":
                    if metadata_root is None:
                        print(f"No metadata available for nnUNet {dataset_code}")
                        continue
                    pred_dir = cfg.output_dir / "predictions" / method / dataset_code / f"fold{fold}"
                    ensure_dir(pred_dir)
                    fold_result = evaluate_nnunet(
                        model_root=model_path,
                        metadata_root=Path(metadata_root),
                        repo_root=Path(repo_root),
                        dataset_root=dataset_root,
                        dataset_name=dataset_artifact_name,
                        fold=fold,
                        max_cases=args.max_cases,
                        checkpoint_kind=cfg.checkpoint,
                        device=device,
                        output_dir=pred_dir,
                    )
                    if fold_result is None:
                        continue
                else:
                    print(f"Method {method} evaluation not implemented in this script yet.")
                    continue

                results[method][dataset_code].append(fold_result.metrics["dice_hard"])

                if wandb_run is not None:
                    wandb_run.log({"dataset": dataset_code, "method": method, "fold": fold, **fold_result.metrics})
                with open(pred_dir / "metrics.json", "w") as f:
                    json.dump(fold_result.metrics, f, indent=2)

            pred_root = cfg.output_dir / "predictions" / method / dataset_code
            if wandb_run is not None and pred_root.exists():
                save_predictions_as_artifact(wandb_run, f"predictions-{method}-{dataset_code}", pred_root)

    print(latex_table(results, metric_name="dice_hard"))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
