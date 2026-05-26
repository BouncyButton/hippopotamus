import argparse
from pathlib import Path

import ltn
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.optimizers import WarmupCosineSchedule
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, Spacingd
from monai.utils import set_determinism
from sklearn.model_selection import KFold
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F

set_determinism(seed=0)
torch.manual_seed(0)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AVAILABLE_CONSTRAINTS = ("prox", "size", "not-nested")
CONSTRAINT_ALIASES = {
    "all": "all",
    "none": "none",
    "prox": "prox",
    "proximity": "prox",
    "chamfer": "prox",
    "size": "size",
    "dimension": "size",
    "dim": "size",
    "not-nested": "not-nested",
    "not_nested": "not-nested",
    "notnested": "not-nested",
    "nested": "not-nested",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structured training script preserving the original hippocampus logic."
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "constraint-informed"),
        default="constraint-informed",
        help="Training objective to use.",
    )
    parser.add_argument(
        "--constraints",
        nargs="*",
        default=("all",),
        help=(
            "Constraint terms to include with constraint-informed training. "
            "Use one or more of: prox, size, not-nested. Also accepts all/none "
            "and comma-separated values, e.g. --constraints prox,size."
        ),
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--train-fraction", type=float, default=0.1, help="Fraction of each training fold to use.")
    parser.add_argument("--folds", type=int, default=5, help="Number of K-fold splits.")
    parser.add_argument("--fold", type=int, default=1, help="Single fold to run, starting from 1.")
    parser.add_argument("--root-dir", default="./tmp", help="MONAI dataset root directory.")
    parser.add_argument("--task", default="Task04_Hippocampus", help="Decathlon task identifier.")
    parser.add_argument("--pixdim", type=float, nargs=3, default=(1.5, .5, 1.5), help="Resampling spacing.")
    parser.add_argument(
        "--spatial-size",
        type=int,
        nargs=3,
        default=(64, 64, 64),
        help="Spatial size used by Resized.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--save-dir", default=".", help="Directory where model checkpoints are written.")
    parser.add_argument("--wandb-project", default="hippopotamus-project", help="W&B project name.")
    parser.add_argument("--wandb-entity", default="hippopotamus", help="W&B entity/account.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb-group", default=None, help="Optional W&B group for aggregating single-fold runs.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B experiment tracking.")
    parser.add_argument(
        "--load-weights",
        default='model_state_dict_ltn_fold1-tr=0.25.pth',
        help="Optional path to a model state_dict checkpoint to load before training.",
    )
    parser.add_argument(
        "--save-val-examples",
        action="store_true",
        default=True,
        help="Save validation image/GT/pred slice visualizations after each epoch.",
    )
    parser.add_argument(
        "--val-example-dir",
        default="val_examples",
        help="Directory where validation visualizations are written.",
    )
    parser.add_argument(
        "--val-example-samples",
        type=int,
        default=3,
        help="Number of validation volumes to visualize per epoch.",
    )
    parser.add_argument(
        "--val-example-max-slices",
        type=int,
        default=0,
        help="Maximum number of slices to save per sample. Use 0 to save all slices.",
    )
    parser.add_argument(
        "--benchmark-constraints",
        action="store_true",
        help="Benchmark the local constraint functions before training.",
    )
    args = parser.parse_args()
    try:
        args.constraints = normalize_constraints(args.constraints)
    except ValueError as exc:
        parser.error(str(exc))
    if args.fold < 1 or args.fold > args.folds:
        parser.error(f"--fold must be in the range 1..{args.folds}.")
    return args


def normalize_constraints(raw_constraints):
    selected = []
    for raw_value in raw_constraints:
        for item in raw_value.split(","):
            name = item.strip().lower()
            if not name:
                continue
            if name not in CONSTRAINT_ALIASES:
                valid = ", ".join((*AVAILABLE_CONSTRAINTS, "all", "none"))
                raise ValueError(f"Unknown constraint '{item}'. Valid values: {valid}.")
            alias = CONSTRAINT_ALIASES[name]
            if alias == "all":
                selected = list(AVAILABLE_CONSTRAINTS)
                continue
            if alias == "none":
                selected = []
                continue
            if alias not in selected:
                selected.append(alias)
    return tuple(selected)


def make_run_name(args):
    size = "x".join(map(str, args.spatial_size))
    pixdim = "x".join(f"{value:g}" for value in args.pixdim)
    constraints = "none" if not args.constraints else "+".join(args.constraints)
    return (
        f"{args.mode}_{constraints}_fold{args.fold}_ep{args.epochs}_bs{args.batch_size}_lr{args.lr:g}_"
        f"wd{args.weight_decay:g}_tr{args.train_fraction:g}_size{size}_pix{pixdim}"
    )


def make_wandb_group(args):
    size = "x".join(map(str, args.spatial_size))
    pixdim = "x".join(f"{value:g}" for value in args.pixdim)
    constraints = "none" if not args.constraints else "+".join(args.constraints)
    return (
        f"{args.mode}_{constraints}_ep{args.epochs}_bs{args.batch_size}_lr{args.lr:g}_"
        f"wd{args.weight_decay:g}_tr{args.train_fraction:g}_folds{args.folds}_size{size}_pix{pixdim}"
    )


def wandb_config(args):
    config = vars(args).copy()
    config["device"] = str(DEVICE)
    config["spatial_size"] = list(args.spatial_size)
    config["pixdim"] = list(args.pixdim)
    return config


def log_epoch_metrics(metrics):
    if wandb.run is not None:
        wandb.log(metrics)


def latest_checkpoint_path(args):
    return Path(args.save_dir) / "latest_model_weights.pth"


def save_latest_model(model, args):
    checkpoint_path = latest_checkpoint_path(args)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    if wandb.run is not None:
        wandb.save(str(checkpoint_path), policy="now")
    return checkpoint_path


def nested(hard):
    mask1 = hard == 1
    mask2 = hard == 2
    mask1 = mask1.unsqueeze(1)
    mask2 = mask2.unsqueeze(1)
    return sample_and_check(mask1, mask2)


def sample_and_check(mask1, mask2, num_samples=20):
    b, c, h, w, d = mask1.shape
    counts = torch.zeros(b, dtype=torch.int, device=DEVICE)

    for i in range(b):
        mask1_indices = torch.nonzero(mask1[i] == 1)

        if mask1_indices.shape[0] < 2:
            continue

        src_dst_pairs = mask1_indices[torch.randint(0, mask1_indices.shape[0], (num_samples, 2))]

        for src, dst in src_dst_pairs:
            src_coords = src[1:]
            dst_coords = dst[1:]

            steps = torch.linspace(0, 1, 50, device=DEVICE)
            interpolated_points = torch.stack(
                [steps * (dst_coords[i] - src_coords[i]) + src_coords[i] for i in range(3)],
                dim=-1,
            )

            for point in interpolated_points:
                point_rounded = torch.round(point).long()
                point_rounded = torch.clamp(
                    point_rounded,
                    min=torch.tensor(0, device=DEVICE),
                    max=torch.tensor([h - 1, w - 1, d - 1], device=DEVICE),
                )

                if mask2[i, 0, point_rounded[0], point_rounded[1], point_rounded[2]] == 1:
                    counts[i] += 1
                    break
            if counts[i] > 0:
                break

    return counts


import torch
import torch.nn.functional as F


def soft_nested(pred):
    soft = torch.softmax(pred, dim=1)
    p1 = soft[:, 1:2]  # keep channel dim
    p2 = soft[:, 2:3]

    return soft_nested_score(p1, p2)


def soft_nested_score(p1, p2, num_pairs=50, num_steps=20, strategy='max'):
    """
    p1, p2: Softmax probabilities (B, 1, H, W, D)
    """
    B, C, H, W, D = p1.shape
    device = p1.device

    # 1. Randomly sample endpoint coordinates in range [-1, 1] for grid_sample
    # (B, num_pairs, 2, 3) -> 2 points (src, dst) per pair, 3 coordinates each
    coords = torch.rand(B, num_pairs, 2, 3, device=device) * 2 - 1
    src = coords[:, :, 0, :]  # (B, num_pairs, 3)
    dst = coords[:, :, 1, :]  # (B, num_pairs, 3)

    # 2. Linear interpolation between src and dst
    # t shape: (num_steps,)
    t = torch.linspace(0, 1, num_steps, device=device).view(1, 1, num_steps, 1)
    # paths shape: (B, num_pairs, num_steps, 3)
    paths = src.unsqueeze(2) * (1 - t) + dst.unsqueeze(2) * t

    # 3. Sample p1 at endpoints and p2 along the path
    # Reshape paths for grid_sample: (B, num_pairs * num_steps, 1, 1, 3)
    sampling_grid = paths.view(B, -1, 1, 1, 3)

    # Sample p2 along the whole path
    sampled_p2 = F.grid_sample(p2, sampling_grid, align_corners=True)
    sampled_p2 = sampled_p2.view(B, num_pairs, num_steps)

    # Sample p1 only at the start and end points
    src_p1 = F.grid_sample(p1, src.view(B, num_pairs, 1, 1, 3), align_corners=True).view(B, num_pairs)
    dst_p1 = F.grid_sample(p1, dst.view(B, num_pairs, 1, 1, 3), align_corners=True).view(B, num_pairs)

    # 4. Compute Violation
    # We want to penalize if (src is M1 AND dst is M1) AND (any point in path is M2)
    # Using SoftMaximum (or just mean) for the path violation
    if strategy == 'max':
        path_violation = torch.max(sampled_p2, dim=2)[0]
    elif strategy == 'mean':
        path_violation = sampled_p2.mean(dim=2)
    elif strategy == 'logsumexp':
        temp = 1.0
        path_violation = temp * torch.logsumexp(sampled_p2 / temp, dim=2)
    elif strategy == 'prob':
        path_violation = 1 - torch.prod(1 - sampled_p2, dim=2)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # The total loss is the product of probabilities
    loss = (src_p1 * dst_p1 * path_violation).mean(dim=1)  # mean over pairs

    return loss


def dimension(pred, gamma=0.0001, epsilon=5000.0):
    hard = pred.argmax(dim=1)
    mask1 = hard == 1
    mask2 = hard == 2

    batch_size = mask1.shape[0]
    dims = torch.zeros(batch_size, device=mask1.device)

    for i in range(batch_size):
        n1 = torch.sum(mask1[i] > 0).item()
        n2 = torch.sum(mask2[i] > 0).item()
        diff = torch.clamp(torch.abs(torch.tensor(n1 - n2, device=mask1.device)) - epsilon, min=0)
        dims[i] = torch.exp(torch.tensor(-gamma * (diff ** 2), device=mask1.device))

    return dims


def dimension_soft(pred, gamma=1e-6):
    probs = torch.softmax(pred, dim=1)
    p1 = probs[:, 1]
    p2 = probs[:, 2]

    n1 = p1.sum(dim=(1, 2, 3))
    n2 = p2.sum(dim=(1, 2, 3))

    diff = n1 - n2
    return 1.0 / (1.0 + gamma * diff ** 2)


def dimension_soft_2(pred, gt, gamma=1.0):
    gt = torch.nn.functional.one_hot(gt.long(), num_classes=3).squeeze(1).permute(0, 4, 1, 2, 3).float()
    gt_p1 = gt[:, 1]
    gt_p2 = gt[:, 2]

    mu = gt_p1.sum(dim=(1, 2, 3)) - gt_p2.sum(dim=(1, 2, 3))
    diff = pred.sum(dim=(1, 2, 3))[:, 1] - pred.sum(dim=(1, 2, 3))[:, 2]

    return torch.exp(-gamma * (diff - mu) ** 2)


class DimensionConstraint(torch.nn.Module):
    def __init__(self, dl: DataLoader):
        super().__init__()
        self.mu, self.std = self.compute_global_stats(dl)

    def compute_global_stats(self, data_loader):
        all_diffs = []
        for batch in tqdm(data_loader, desc="Computing dimension constraint stats"):
            labels = batch["label"].to(DEVICE)
            hard = labels.squeeze(1).long()
            mask1 = hard == 1
            mask2 = hard == 2

            n1 = mask1.float().mean(dim=(1, 2, 3))
            n2 = mask2.float().mean(dim=(1, 2, 3))

            diff = torch.log(n1 / (n2 + 1e-8) + 1e-8)
            all_diffs.append(diff)

        all_diffs = torch.cat(all_diffs).float()
        mu = all_diffs.mean().item()
        var = all_diffs.var(unbiased=False).item()
        std = all_diffs.std(unbiased=False).item()
        print(f"dimension constraint stats: mean={mu:.8f}, std={std:.8f}")

        return mu, std

    def forward(self, pred):
        probs = torch.softmax(pred, dim=1)
        V = probs.mean(dim=(2, 3, 4))  # [B, C]

        V1 = V[:, 1]
        V2 = V[:, 2]

        d = torch.log(V1 / (V2 + 1e-8) + 1e-8)

        z = (d - self.mu) / (self.std + 1e-8)

        # score = torch.exp(-0.5 * z ** 2)
        margin = 1.96
        # apply a margin to encourage being within the 95% confidence interval of the observed distribution with a relu

        score = torch.exp(-0.5 * torch.clamp(z.abs() - margin, min=0.0) ** 2)
        return score


def chamfer_distance(pred):
    hard = pred.argmax(dim=1)
    mask1 = hard == 1
    mask2 = hard == 2

    batch_size = mask1.shape[0]
    chamfer_dists = torch.zeros(batch_size, device=mask1.device)

    for i in range(batch_size):
        coords1 = torch.nonzero(mask1[i], as_tuple=False).float()
        coords2 = torch.nonzero(mask2[i], as_tuple=False).float()

        if coords1.numel() == 0 or coords2.numel() == 0:
            chamfer_dists[i] = float("inf")
            continue

        dists = torch.cdist(coords1, coords2, p=2)
        min_dists1 = torch.min(dists, dim=1)[0]
        min_dists2 = torch.min(dists, dim=0)[0]
        chamfer_dists[i] = torch.mean(min_dists1) + torch.mean(min_dists2)

    return chamfer_dists


def soft_chamfer_pooled(pred, tau=1.0, pool=5, eps=1e-8):
    probs = torch.softmax(pred, dim=1)
    p1 = probs[:, 1:2]  # keep channel dim
    p2 = probs[:, 2:3]

    # --- downsample ---
    if pool > 1:
        p1 = torch.nn.functional.max_pool3d(p1, kernel_size=pool, stride=pool)
        p2 = torch.nn.functional.max_pool3d(p2, kernel_size=pool, stride=pool)

    B, _, H, W, D = p1.shape
    device = pred.device

    # grid
    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        torch.arange(D, device=device),
        indexing="ij"
    ), dim=-1).float().view(-1, 3)

    N = coords.shape[0]

    p1 = p1.view(B, N)
    p2 = p2.view(B, N)

    dists = torch.cdist(coords, coords)  # now much smaller!

    out = []
    for b in range(B):
        p1_b = p1[b] + eps
        p2_b = p2[b] + eps

        p1_b = p1_b / p1_b.sum()
        p2_b = p2_b / p2_b.sum()

        log_p2 = torch.log(p2_b).unsqueeze(0)
        d1 = -1.0 / tau * torch.logsumexp(log_p2 - tau * dists, dim=1)

        log_p1 = torch.log(p1_b).unsqueeze(0)
        d2 = -1.0 / tau * torch.logsumexp(log_p1 - tau * dists, dim=1)

        cd = (p1_b * d1).sum() + (p2_b * d2).sum()
        out.append(cd)

    return torch.stack(out)


def soft_chamfer_distance(pred, tau=10.0, eps=1e-8):
    """
    Fully differentiable Chamfer distance using softmin (log-sum-exp).

    Args:
        pred: logits (B, C, H, W, D)
        tau: temperature (higher = closer to true min)
        eps: numerical stability

    Returns:
        (B,) tensor of distances
    """
    probs = torch.softmax(pred, dim=1)
    p1 = probs[:, 1]  # (B, H, W, D)
    p2 = probs[:, 2]

    B, H, W, D = p1.shape
    device = pred.device

    # Flatten spatial grid
    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        torch.arange(D, device=device),
        indexing="ij"
    ), dim=-1).float()  # (H, W, D, 3)

    coords = coords.view(-1, 3)  # (N, 3)
    N = coords.shape[0]

    # Flatten probabilities
    p1_flat = p1.view(B, N)  # (B, N)
    p2_flat = p2.view(B, N)

    # Pairwise distances (N, N)
    dists = torch.cdist(coords, coords, p=2)  # heavy but exact

    results = []

    for b in range(B):
        p1_b = p1_flat[b] + eps
        p2_b = p2_flat[b] + eps

        # normalize to proper distributions
        p1_b = p1_b / p1_b.sum()
        p2_b = p2_b / p2_b.sum()

        # --- softmin distances ---
        # d_tau(x, p2) = -1/tau log sum_y p2(y) exp(-tau d(x,y))

        log_p2 = torch.log(p2_b).unsqueeze(0)  # (1, N)
        softmin_1_to_2 = -1.0 / tau * torch.logsumexp(
            log_p2 - tau * dists, dim=1
        )  # (N,)

        log_p1 = torch.log(p1_b).unsqueeze(0)
        softmin_2_to_1 = -1.0 / tau * torch.logsumexp(
            log_p1 - tau * dists, dim=1
        )

        # expectation
        cd = (p1_b * softmin_1_to_2).sum() + (p2_b * softmin_2_to_1).sum()

        results.append(cd)

    return torch.stack(results)


def build_transforms(pixdim, spatial_size):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=tuple(pixdim), mode=("bilinear", "nearest")),
            Resized(keys=["image", "label"], spatial_size=tuple(spatial_size), mode=("bilinear", "nearest")),
        ]
    )


def build_dataset(args):
    return DecathlonDataset(
        root_dir=args.root_dir,
        task=args.task,
        section="training",
        download=True,
        transform=build_transforms(args.pixdim, args.spatial_size),
    )


def build_fold_subsets(dataset, folds, fold_number, train_fraction):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    for current_fold, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
        if current_fold != fold_number:
            continue

        train_limit = max(1, int(train_fraction * len(train_idx)))
        train_subset = torch.utils.data.Subset(dataset, train_idx[:train_limit])
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        return train_subset, val_subset

    raise ValueError(f"Fold {fold_number} is outside the available range 1..{folds}.")


def evaluate(model, val_loader, device):
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).squeeze(1)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).squeeze(1)
            dice_metric(preds, labels)
    dice_score, _ = dice_metric.aggregate()
    dice_metric.reset()
    return dice_score.cpu().item()


def flatten_tensor_list(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0)


def zero_gradient_vector(parameters):
    return torch.zeros(
        sum(parameter.numel() for parameter in parameters),
        device=parameters[0].device,
        dtype=parameters[0].dtype,
    )


def gradient_vector(parameters, scalar, retain_graph):
    if not scalar.requires_grad:
        return zero_gradient_vector(parameters)

    grads = torch.autograd.grad(
        scalar,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    filled = []
    for parameter, grad in zip(parameters, grads):
        if grad is None:
            filled.append(torch.zeros_like(parameter))
        else:
            filled.append(grad.detach())
    return flatten_tensor_list(filled)


def gradient_norm(gradient):
    return torch.linalg.vector_norm(gradient).item()


def cosine_similarity(grad_a, grad_b):
    denom = torch.linalg.vector_norm(grad_a) * torch.linalg.vector_norm(grad_b)
    if denom.item() == 0:
        return 0.0
    return torch.dot(grad_a, grad_b).div(denom).item()


dice_loss = DiceLoss(to_onehot_y=True, softmax=True, reduction="none")


def eq_fn3d(u, v, alpha=0.3):
    return torch.exp(-alpha * torch.sqrt(torch.sum(torch.square(u - v), dim=1))).mean(dim=(1, 2, 3))


def eq_fn(u, v, alpha=1e-3):
    return torch.exp(-alpha * torch.sqrt(torch.square(u - v)))


def my_dice_loss(outputs, labels):
    return 1.0 - dice_loss(outputs, labels).mean(dim=1)


def save_validation_examples(model, val_loader, device, args, epoch):
    model.eval()
    output_root = Path(args.val_example_dir) / f"epoch_{epoch + 1:03d}"
    output_root.mkdir(parents=True, exist_ok=True)

    saved_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for sample_idx in range(images.shape[0]):
                if saved_samples >= args.val_example_samples:
                    return

                global_sample_idx = saved_samples
                sample_dir = output_root / f"sample_{global_sample_idx:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                image_volume = images[sample_idx, 0].detach().cpu().float().numpy()
                gt_volume = labels[sample_idx, 0].detach().cpu().numpy()
                pred_volume = preds[sample_idx].detach().cpu().numpy()

                depth = image_volume.shape[-1]
                if args.val_example_max_slices > 0:
                    slice_indices = np.linspace(
                        0,
                        depth - 1,
                        num=min(depth, args.val_example_max_slices),
                        dtype=int,
                    )
                else:
                    slice_indices = np.arange(depth)

                image_min = float(image_volume.min())
                image_max = float(image_volume.max())
                denom = image_max - image_min

                for slice_idx in slice_indices:
                    image_slice = image_volume[:, :, slice_idx]
                    if denom > 0:
                        image_slice = (image_slice - image_min) / denom
                    else:
                        image_slice = np.zeros_like(image_slice)

                    gt_slice = gt_volume[:, :, slice_idx]
                    pred_slice = pred_volume[:, :, slice_idx]

                    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                    panels = (
                        (image_slice, "image", "gray", None),
                        (gt_slice, "ground truth", "viridis", (0, 2)),
                        (pred_slice, "prediction", "viridis", (0, 2)),
                    )

                    for ax, (panel, title, cmap, limits) in zip(axes, panels):
                        if limits is None:
                            ax.imshow(panel, cmap=cmap)
                        else:
                            ax.imshow(panel, cmap=cmap, vmin=limits[0], vmax=limits[1])
                        ax.set_title(title)
                        ax.axis("off")

                    fig.tight_layout()
                    fig.savefig(sample_dir / f"slice_{slice_idx:03d}.png", dpi=120, bbox_inches="tight")
                    plt.close(fig)

                saved_samples += 1


def metric_key(name):
    return name.replace("-", "_")


def build_constraint_context(train_loader, args, device):
    context = {
        "forall": ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f"),
        "sat_agg": ltn.fuzzy_ops.SatAgg(),
        "dice": ltn.Predicate(func=my_dice_loss),
    }
    if "prox" in args.constraints:
        context["eq"] = ltn.Predicate(func=eq_fn)
        context["min_dst"] = ltn.Function(func=soft_chamfer_pooled)
        context["zero"] = ltn.Constant(torch.tensor(0.0, device=device))
    if "size" in args.constraints:
        context["sim_dim"] = ltn.Function(model=DimensionConstraint(train_loader).to(device))
    if "not-nested" in args.constraints:
        context["nested"] = ltn.Function(func=soft_nested)
        context["not"] = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    return context


def compute_constraint_satisfaction(outputs, labels, context, selected_constraints):
    pred = ltn.Variable("pred", outputs)
    y = ltn.Variable("y", labels)
    forall = context["forall"]

    sats = {
        "seg": forall(ltn.diag(pred, y), context["dice"](pred, y)).value,
    }
    if "prox" in selected_constraints:
        sats["prox"] = forall(pred, context["eq"](context["min_dst"](pred), context["zero"])).value
    if "size" in selected_constraints:
        sats["size"] = forall(pred, context["sim_dim"](pred)).value
    if "not-nested" in selected_constraints:
        sats["not-nested"] = forall(pred, context["not"](context["nested"](pred))).value

    satisfaction_terms = [sats["seg"], *(sats[name] for name in selected_constraints)]
    total_sat = context["sat_agg"](*satisfaction_terms)
    return total_sat, sats


def satisfaction_metrics(total_sat, sats, selected_constraints):
    metrics = {
        f"{metric_key(name)}_sat": sats[name].item()
        for name in ("seg", *selected_constraints)
    }
    metrics["total_sat"] = total_sat.item()
    return metrics


def evaluate_constraint_satisfaction(model, data_loader, context, selected_constraints, device):
    model.eval()
    totals = {}
    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            batch_size = inputs.shape[0]

            outputs = model(inputs)
            total_sat, sats = compute_constraint_satisfaction(outputs, labels, context, selected_constraints)
            batch_metrics = satisfaction_metrics(total_sat, sats, selected_constraints)

            for name, value in batch_metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            sample_count += batch_size

    if sample_count == 0:
        return {}
    return {name: value / sample_count for name, value in totals.items()}


def compute_gradient_metrics(parameters, sats, selected_constraints):
    ordered_names = ("seg", *selected_constraints)
    grads = {}
    for name in ordered_names:
        grads[name] = gradient_vector(parameters, sats[name], retain_graph=True)

    metrics = {
        f"{metric_key(name)}_norm": gradient_norm(grads[name])
        for name in ordered_names
    }
    for name in selected_constraints:
        metrics[f"seg_{metric_key(name)}_cos"] = cosine_similarity(grads["seg"], grads[name])
    return metrics


def format_satisfaction_metrics(metrics, selected_constraints):
    names = ("seg", *selected_constraints, "total")
    return ", ".join(f"{metric_key(name)}={metrics[metric_key(name) + '_sat']:.4f}" for name in names)


def format_gradient_norms(metrics, selected_constraints):
    names = ("seg", *selected_constraints)
    return ", ".join(f"|{metric_key(name)}|={metrics[metric_key(name) + '_norm']:.4e}" for name in names)


def format_gradient_cosines(metrics, selected_constraints):
    if not selected_constraints:
        return ""
    return ", ".join(
        f"cos(seg,{metric_key(name)})={metrics['seg_' + metric_key(name) + '_cos']:.4f}"
        for name in selected_constraints
    )


def build_log_metrics(
    args,
    fold,
    epoch,
    epoch_loss,
    optimizer,
    dice_score,
    checkpoint_path,
    last_metrics,
    last_grad_metrics,
    val_constraint_metrics,
):
    metrics = {
        "fold": fold,
        "epoch": epoch + 1,
        "global_epoch": (fold - 1) * args.epochs + epoch + 1,
        "train/loss": epoch_loss,
        "train/lr": optimizer.param_groups[0]["lr"],
        "val/dice": dice_score,
        "checkpoint/latest_path": str(checkpoint_path),
    }
    for name, value in last_metrics.items():
        metrics[f"constraint/{name}"] = value
    for name, value in last_grad_metrics.items():
        metrics[f"constraint_grad/{name}"] = value
    for name, value in val_constraint_metrics.items():
        metrics[f"val_constraint/{name}"] = value
    return metrics


def train_one_fold(model, train_loader, val_loader, args, device, fold, schedule='base'):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train_loader)
    scheduler = None
    if schedule == 'cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=max(1, steps // 10), t_total=max(1, steps))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    baseline_loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    print("tracked constraints:", ", ".join(args.constraints) if args.constraints else "none")
    if args.mode == "constraint-informed":
        print("active constraints:", ", ".join(args.constraints) if args.constraints else "none")
    constraint_context = build_constraint_context(train_loader, args, device)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        last_metrics = {}
        last_grad_metrics = {}
        for batch in tqdm(train_loader, desc=f"{args.mode} epoch {epoch + 1}/{args.epochs}"):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            if args.mode == "baseline":
                loss = baseline_loss_fn(outputs, labels)
            else:
                satisfaction, sats = compute_constraint_satisfaction(
                    outputs,
                    labels,
                    constraint_context,
                    args.constraints,
                )
                loss = 1.0 - satisfaction
                last_grad_metrics = compute_gradient_metrics(parameters, sats, args.constraints)
                last_metrics = satisfaction_metrics(satisfaction, sats, args.constraints)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / max(1, len(train_loader))
        print(f"{args.mode} epoch {epoch + 1}/{args.epochs}, loss: {mean_loss:.4f}")
        if last_metrics:
            print("constraint sats:", format_satisfaction_metrics(last_metrics, args.constraints))
            print("constraint grads:", format_gradient_norms(last_grad_metrics, args.constraints))
            grad_cosines = format_gradient_cosines(last_grad_metrics, args.constraints)
            if grad_cosines:
                print("constraint cos:", grad_cosines)
        dice_score = evaluate(model, val_loader, device)
        print(f"val dice score: {dice_score:.4f}")
        val_constraint_metrics = evaluate_constraint_satisfaction(
            model,
            val_loader,
            constraint_context,
            args.constraints,
            device,
        )
        if val_constraint_metrics:
            print("val constraint sats:", format_satisfaction_metrics(val_constraint_metrics, args.constraints))
        save_validation_examples(model, val_loader, device, args, epoch)
        checkpoint_path = save_latest_model(model, args)
        log_epoch_metrics(
            build_log_metrics(
                args,
                fold,
                epoch,
                mean_loss,
                optimizer,
                dice_score,
                checkpoint_path,
                last_metrics,
                last_grad_metrics,
                val_constraint_metrics,
            )
        )
        if scheduler is not None:
            scheduler.step()
    return model


def run_single_fold_experiment(args):
    run = None
    if not args.disable_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or make_run_name(args),
            group=args.wandb_group or make_wandb_group(args),
            config=wandb_config(args),
            job_type="training",
        )
        wandb.define_metric("global_epoch")
        wandb.define_metric("train/*", step_metric="global_epoch")
        wandb.define_metric("val/*", step_metric="global_epoch")
        wandb.define_metric("constraint/*", step_metric="global_epoch")
        wandb.define_metric("constraint_grad/*", step_metric="global_epoch")
        wandb.define_metric("val_constraint/*", step_metric="global_epoch")

    dataset = build_dataset(args)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fold = args.fold
    print(f"Fold {fold}/{args.folds}")

    train_subset, val_subset = build_fold_subsets(dataset, args.folds, fold, args.train_fraction)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SwinUNETR(in_channels=1, out_channels=3, use_checkpoint=True).to(DEVICE)
    if args.load_weights:
        checkpoint = torch.load(args.load_weights, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint)
        print(f"loaded weights from: {args.load_weights}")
    trained_model = train_one_fold(model, train_loader, val_loader, args, DEVICE, fold)

    fold_dice = evaluate(trained_model, val_loader, DEVICE)
    print(f"Fold {fold} dice: {fold_dice:.4f}")
    log_epoch_metrics({"fold": fold, "fold/final_dice": fold_dice})

    if run is not None:
        run.summary["fold"] = fold
        run.summary["fold/final_dice"] = fold_dice
        run.finish()


def main():
    args = parse_args()
    if args.benchmark_constraints:
        benchmark_constraint_computation()
    run_single_fold_experiment(args)


def benchmark_constraint_computation():
    import time

    a = torch.randn(2, 3, 64, 64, 64).to(DEVICE)
    b = torch.randint(0, 3, (2, 1, 64, 64, 64)).to(DEVICE)

    t = time.time()
    _ = my_dice_loss(a, b)
    print("dice loss time:", time.time() - t)

    t = time.time()
    _ = soft_chamfer_pooled(a, pool=8)
    print("soft chamfer time:", time.time() - t)

    t = time.time()
    _ = dimension_soft(a)
    print("dimension soft time:", time.time() - t)

    t = time.time()
    _ = dimension(a)

    print("dimension time:", time.time() - t)

    t = time.time()
    _ = soft_nested(a)
    print("nested time:", time.time() - t)


if __name__ == "__main__":
    main()
