from monai.utils import set_determinism
from monai.transforms import Compose, EnsureChannelFirstd, NormalizeIntensityd, Resized, EnsureTyped, Lambdad, DivisiblePadd
from monai.data import DataLoader, Dataset as MonaiDataset
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
import torch
import numpy as np
import pandas as pd
import argparse
import re
import json
from sklearn.model_selection import GroupKFold


set_determinism(seed=0)
torch.manual_seed(0)


def _one_hot(labels, num_classes):
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    if labels.ndim != 4:
        raise ValueError(f"Expected labels with shape (B,H,W,D), got {labels.shape}")
    oh = torch.nn.functional.one_hot(labels.long(), num_classes=num_classes)
    return oh.permute(0, 4, 1, 2, 3).float()


def evaluate(model, val_loader, device, num_classes):
    hard_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    soft_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)

            soft_preds = torch.softmax(outputs, dim=1)
            hard_preds = torch.argmax(outputs, dim=1)

            y = _one_hot(labels, num_classes)
            y_soft = soft_preds
            y_hard = _one_hot(hard_preds, num_classes)

            soft_metric(y_soft, y)
            hard_metric(y_hard, y)

    soft_score, _ = soft_metric.aggregate()
    hard_score, _ = hard_metric.aggregate()
    return soft_score.cpu().item(), hard_score.cpu().item()


def build_optimizer_and_scheduler(mode, model, num_epochs):
    if mode == "adamw_0.01":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return optimizer, scheduler
    if mode == "nnunetv2":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-2,
            momentum=0.99,
            nesterov=True,
            weight_decay=3e-5,
        )
        scheduler = PolyLRScheduler(optimizer, initial_lr=1e-2, max_epochs=num_epochs, exponent=0.9)
        return optimizer, scheduler
    raise ValueError(f"Unknown optimizer mode: {mode}")


class PolyLRScheduler:
    def __init__(self, optimizer, initial_lr, max_epochs, exponent=0.9):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.exponent = exponent

    def step(self, epoch):
        lr = self.initial_lr * (1 - epoch / self.max_epochs) ** self.exponent
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


def train_with_optimizer(
    model,
    train_loader,
    val_loader,
    num_epochs,
    num_classes,
    optimizer,
    scheduler,
):
    device = next(model.parameters()).device
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step(epoch + 1)

        avg_loss = epoch_loss / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6g}")
        soft_dice, hard_dice = evaluate(model, val_loader, device, num_classes=num_classes)
        print(f"Val Dice (soft): {soft_dice:.4f}")
        print(f"Val Dice (hard): {hard_dice:.4f}")


def _case_name_from_row(row, dataset):
    dataset = dataset.upper()
    if dataset == "MSD":
        subject_id = str(row["subject_id"])
        return f"hippocampus_{subject_id}"
    if dataset == "MNI":
        subject_id = str(row["subject_id"])
        direction = str(row["direction"])
        return f"hippocampus_mni_{subject_id}_{direction}"
    if dataset == "ADNI":
        image_id = str(row["image_id"])
        direction = str(row["direction"])
        return f"hippocampus_adni_{image_id}_{direction}"
    if dataset == "COBRA":
        subject_id = str(row["subject_id"])
        side = str(row["side"])
        return f"hippocampus_cobra_{subject_id}_{side}"
    raise ValueError(f"Unknown dataset: {dataset}")


def _patient_id_from_case(case_name, dataset):
    dataset = dataset.upper()
    if dataset == "MNI":
        match = re.search(r"s\\d+", case_name)
        return match.group() if match else case_name
    if dataset == "ADNI":
        match = re.search(r"adni_\\d+", case_name)
        return match.group() if match else case_name
    if dataset == "COBRA":
        match = re.search(r"cobra_\\d+", case_name)
        return match.group() if match else case_name
    if dataset == "MSD":
        return case_name
    raise ValueError(f"Unknown dataset: {dataset}")


def _load_pkl_dataframe(pkl_path):
    df = pd.read_pickle(pkl_path, compression="gzip")
    if df.empty:
        raise ValueError(f"Empty dataframe in {pkl_path}")
    return df


def _infer_num_classes(df, dataset):
    dataset = dataset.upper()
    if dataset in {"MSD", "COBRA"}:
        label_key = "label_data"
    elif dataset in {"MNI", "ADNI"}:
        label_key = "data"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    max_label = 0
    for _, row in df.iterrows():
        arr = np.asarray(row[label_key])
        if arr.size == 0:
            continue
        max_label = max(max_label, int(np.nanmax(arr)))
    return max_label + 1


def _build_monai_dataset_from_pkl(df, dataset, num_classes, spatial_size=(64, 64, 64), do_resize=False):
    dataset = dataset.upper()
    if dataset in {"MSD", "COBRA"}:
        image_key = "image_data"
        label_key = "label_data"
    elif dataset in {"MNI", "ADNI"}:
        image_key = "image_data"
        label_key = "data"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    records = []
    for _, row in df.iterrows():
        records.append({
            "case_name": _case_name_from_row(row, dataset),
            "image": row[image_key],
            "label": row[label_key],
        })

    xforms = [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # nnU-Net default for non-CT modalities is per-case z-score
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        DivisiblePadd(keys=["image", "label"], k=32, mode=("constant", "constant")),
        Lambdad(
            keys=["label"],
            func=lambda x: torch.clamp(torch.round(torch.as_tensor(x)), 0, num_classes - 1),
        ),
        EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.long)),
    ]
    if do_resize:
        xforms.insert(
            2,
            Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        )
    transforms = Compose(xforms)

    return MonaiDataset(data=records, transform=transforms)


def _split_with_groupkfold(case_names, dataset, fold, n_splits=5):
    case_names = sorted(case_names)
    groups = [_patient_id_from_case(c, dataset) for c in case_names]
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    case_array = np.array(case_names)
    splits = []
    for train_idx, val_idx in gkf.split(case_names, groups=groups):
        splits.append({
            "train": sorted(case_array[train_idx].tolist()),
            "val": sorted(case_array[val_idx].tolist()),
        })

    if fold < 0 or fold >= len(splits):
        raise ValueError(f"Fold must be in [0, {len(splits) - 1}]")
    return splits, splits[fold]


def _load_splits_json(path):
    with open(path, "r") as f:
        splits = json.load(f)
    if not isinstance(splits, list) or not splits:
        raise ValueError("splits JSON must be a non-empty list")
    for i, fold in enumerate(splits):
        if "train" not in fold or "val" not in fold:
            raise ValueError(f"fold {i} missing train/val keys")
    return splits


def _print_batch_stats(images, labels, prefix="Train"):
    img = images.detach().float()
    lbl = labels.detach().float()
    img_min = img.min().item()
    img_max = img.max().item()
    img_mean = img.mean().item()
    img_std = img.std().item()
    nonzero = img != 0
    if nonzero.any():
        nz_mean = img[nonzero].mean().item()
        nz_std = img[nonzero].std().item()
    else:
        nz_mean = float("nan")
        nz_std = float("nan")
    lbl_min = lbl.min().item()
    lbl_max = lbl.max().item()
    uniq = torch.unique(labels.detach().cpu()).tolist()
    uniq = sorted(uniq)[:20]
    print(
        f"{prefix} batch stats | image min/max/mean/std: "
        f"{img_min:.4f}/{img_max:.4f}/{img_mean:.4f}/{img_std:.4f} | "
        f"nonzero mean/std: {nz_mean:.4f}/{nz_std:.4f} | "
        f"label min/max: {lbl_min:.0f}/{lbl_max:.0f} | labels: {uniq}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to dataset .pkl (gzip) file")
    parser.add_argument("--dataset", required=True, choices=["MSD", "MNI", "ADNI", "COBRA"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--resize", action="store_true", help="Enable resize to --spatial-size")
    parser.add_argument("--splits-out", default=None, help="Optional JSON output path for all splits")
    parser.add_argument("--splits-json", default=None, help="Use an existing splits JSON for exact train/val")
    parser.add_argument("--debug-stats", action="store_true", help="Print one batch of image/label stats")
    parser.add_argument("--num-classes", type=int, default=None, help="Override inferred number of classes")
    parser.add_argument(
        "--optim-mode",
        default="adamw_0.01",
        choices=["adamw_0.01", "nnunetv2"],
        help="Optimizer/schedule preset",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = _load_pkl_dataframe(args.pkl)
    num_classes = args.num_classes or _infer_num_classes(df, args.dataset)
    monai_dataset = _build_monai_dataset_from_pkl(
        df,
        args.dataset,
        num_classes,
        spatial_size=tuple(args.spatial_size),
        do_resize=args.resize,
    )

    case_names = [item["case_name"] for item in monai_dataset.data]
    if args.splits_json:
        all_splits = _load_splits_json(args.splits_json)
        if args.fold < 0 or args.fold >= len(all_splits):
            raise ValueError(f"Fold must be in [0, {len(all_splits) - 1}]")
        split = all_splits[args.fold]
        print(json.dumps(all_splits, indent=4))
    else:
        all_splits, split = _split_with_groupkfold(case_names, args.dataset, args.fold)
        print(json.dumps(all_splits, indent=4))
        if args.splits_out:
            with open(args.splits_out, "w") as f:
                json.dump(all_splits, f, indent=4)
            print(f"Saved splits to {args.splits_out}")

    train_set = set(split["train"])
    val_set = set(split["val"])

    train_items = [item for item in monai_dataset.data if item["case_name"] in train_set]
    val_items = [item for item in monai_dataset.data if item["case_name"] in val_set]

    train_ds = MonaiDataset(data=train_items, transform=monai_dataset.transform)
    val_ds = MonaiDataset(data=val_items, transform=monai_dataset.transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SwinUNETR(in_channels=1, out_channels=num_classes, use_checkpoint=True).to(device)
    if args.debug_stats:
        for batch in train_loader:
            _print_batch_stats(batch["image"], batch["label"], prefix="Train")
            break
    optimizer, scheduler = build_optimizer_and_scheduler(args.optim_mode, model, args.epochs)
    train_with_optimizer(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        num_classes=num_classes,
        optimizer=optimizer,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
