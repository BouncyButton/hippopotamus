import wandb
import argparse

wandb.init(project="hippopotamus-project", entity="hippopotamus")

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, required=True, help='Path to the uNetr++ model checkpoint folder')
parser.add_argument('--fold', type=int, required=True, help='Fold number of the uNetr++ model')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name the model was trained on')
args = parser.parse_args()

artifact = wandb.Artifact(
    name=f"unetrpp-model-{args.dataset}-fold{args.fold}",
    type="model"
)

artifact.add_dir(args.model_path)

# save also plans.pkl in the upper folder
artifact.add_file(f"{args.model_path}/../plans.pkl", name="plans.pkl")

wandb.log_artifact(artifact)

wandb.finish()
