from monai.utils import set_determinism
from monai.transforms import Compose, EnsureChannelFirstd, ScaleIntensityd, Resized, EnsureTyped
from monai.data import DataLoader, Dataset as MonaiDataset
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.optimizers import WarmupCosineSchedule
import torch
import numpy as np
import pandas as pd
import argparse
import re
import json
from sklearn.model_selection import GroupKFold


set_determinism(seed=0)
torch.manual_seed(0)


def evaluate(model, val_loader, device):
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch["image"].to(device), batch["label"].to(device).squeeze(1)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).squeeze(1)
            dice_metric(preds, labels)
    dice_score, _ = dice_metric.aggregate()
    return dice_score.cpu().item()


def train(model, train_loader, val_loader, num_epochs=20):
    device = next(model.parameters()).device
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    steps = num_epochs * len(train_loader)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=steps // 10, t_total=steps)

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

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        dice_score = evaluate(model, val_loader, device)
        print(f"Val Dice: {dice_score:.4f}")
        scheduler.step()


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


def _build_monai_dataset_from_pkl(df, dataset, spatial_size=(64, 64, 64)):
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

    transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.long)),
    ])

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to dataset .pkl (gzip) file")
    parser.add_argument("--dataset", required=True, choices=["MSD", "MNI", "ADNI", "COBRA"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--splits-out", default=None, help="Optional JSON output path for all splits")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = _load_pkl_dataframe(args.pkl)
    monai_dataset = _build_monai_dataset_from_pkl(
        df,
        args.dataset,
        spatial_size=tuple(args.spatial_size),
    )

    case_names = [item["case_name"] for item in monai_dataset.data]
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

    model = SwinUNETR(in_channels=1, out_channels=3, use_checkpoint=True).to(device)
    train(model, train_loader, val_loader, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
