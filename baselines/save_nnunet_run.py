import wandb
import argparse

wandb.init(project="hippopotamus-project", entity="hippopotamus")

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, required=True, help='Path to the nnU-Net model checkpoint folder')
parser.add_argument('--fold', type=int, required=True, help='Fold number of the nnU-Net model')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name the model was trained on')
parser.add_argument('--seed', type=int, default=None, help='Optional split seed to include in artifact name')
args = parser.parse_args()

artifact_name = f"nnunet-model-{args.dataset}-fold{args.fold}"
if args.seed is not None:
    artifact_name = f"nnunet-model-{args.dataset}-seed{args.seed}-fold{args.fold}"

artifact = wandb.Artifact(
    name=artifact_name,
    type="model"
)

artifact.add_dir(args.model_path)

wandb.log_artifact(artifact)

wandb.finish()
