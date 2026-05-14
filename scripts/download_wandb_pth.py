import argparse
from pathlib import Path

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Download a .pth checkpoint artifact from Weights & Biases.")
    parser.add_argument("--name", required=True, help="Artifact name to download.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where the artifact will be downloaded.")
    parser.add_argument("--type", default="model", help="Artifact type. Defaults to 'model'.")
    parser.add_argument("--project", default="hippopotamus-project", help="W&B project name.")
    parser.add_argument("--entity", default="hippopotamus", help="W&B entity/account.")
    parser.add_argument("--run-name", default=None, help="Optional W&B run name.")
    parser.add_argument("--alias", default="latest", help="Artifact alias or version. Defaults to 'latest'.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifact_ref = f"{args.entity}/{args.project}/{args.name}:{args.alias}"

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, job_type="artifact-download")
    artifact = run.use_artifact(artifact_ref, type=args.type)
    artifact_dir = Path(artifact.download(root=str(output_dir))).resolve()
    run.finish()

    checkpoint_paths = sorted(artifact_dir.glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"downloaded artifact does not contain a .pth file: {artifact_dir}")

    print(f"downloaded artifact '{artifact_ref}' to {artifact_dir}")
    for checkpoint_path in checkpoint_paths:
        print(f"found checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
