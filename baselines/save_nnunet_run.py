import wandb
import argparse

wandb.init(project="hippopotamus-project", entity="hippopotamus")

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, required=True, help='Path to the nnU-Net model checkpoint folder')
parser.add_argument('--fold', type=int, required=True, help='Fold number of the nnU-Net model')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name the model was trained on')
args = parser.parse_args()

artifact = wandb.Artifact(
    name=f"nnunet-model-{args.dataset}-fold{args.fold}",
    type="model"
)

artifact.add_dir(args.model_path)

wandb.log_artifact(artifact)

wandb.finish()
