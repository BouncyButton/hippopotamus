#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _list_nii_gz(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    return sorted(
        p.name for p in folder.iterdir()
        if p.is_file() and p.name.endswith(".nii.gz") and not p.name.startswith("._")
    )


def _normalize_labels(labels_obj) -> Dict[str, str]:
    # nnUNet style: {"background": 0, "classA": 1}
    # UNETR++/notebook style: {"0": "background", "1": "classA"}
    if not isinstance(labels_obj, dict):
        return {"0": "background"}

    keys_are_numeric = all(str(k).isdigit() for k in labels_obj.keys())
    values_are_strings = all(isinstance(v, str) for v in labels_obj.values())
    if keys_are_numeric and values_are_strings:
        return {str(k): str(v) for k, v in labels_obj.items()}

    out: Dict[str, str] = {}
    for label_name, label_id in labels_obj.items():
        out[str(label_id)] = str(label_name)
    return out


def _normalize_channels(payload: dict) -> Dict[str, str]:
    channel_names = payload.get("channel_names")
    if isinstance(channel_names, dict) and channel_names:
        return {str(k): str(v) for k, v in channel_names.items()}

    modality = payload.get("modality")
    if isinstance(modality, dict) and modality:
        return {str(k): str(v) for k, v in modality.items()}

    return {"0": "MRI"}


def _dataset_name_from_task(task_dir: Path) -> str:
    # Task102_MNI -> MNI
    name = task_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Convert nnUNet dataset.json to UNETR++-compatible dataset.json with explicit file listing."
    )
    parser.add_argument("--task-dir", required=True, help="Task directory (contains imagesTr/labelsTr[/imagesTs])")
    parser.add_argument("--source-json", required=True, help="Existing nnUNet-style dataset.json")
    parser.add_argument("--output", default=None, help="Output dataset.json (default: <task-dir>/dataset.json)")
    parser.add_argument("--name", default=None, help="Dataset name override (default: inferred from task dir)")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    source_json = Path(args.source_json)
    output = Path(args.output) if args.output else task_dir / "dataset.json"

    with open(source_json, "r") as f:
        src = json.load(f)

    images_tr_dir = task_dir / "imagesTr"
    labels_tr_dir = task_dir / "labelsTr"
    images_ts_dir = task_dir / "imagesTs"

    image_files_tr = _list_nii_gz(images_tr_dir)
    label_files_tr = _list_nii_gz(labels_tr_dir)
    test_image_files = _list_nii_gz(images_ts_dir)
    label_set = set(label_files_tr)

    training_cases = []
    for image_file in image_files_tr:
        case = image_file.replace(".nii.gz", "")
        case = case[:-5] if case.endswith("_0000") else case

        cand_with_suffix = f"{case}_0000.nii.gz"
        cand_without_suffix = f"{case}.nii.gz"
        label_name: Optional[str] = None
        if cand_with_suffix in label_set:
            label_name = cand_with_suffix
        elif cand_without_suffix in label_set:
            label_name = cand_without_suffix

        if label_name is None:
            continue

        training_cases.append({
            "image": f"./imagesTr/{case}.nii.gz",
            "label": f"./labelsTr/{label_name}",
        })

    test_cases = [f"./imagesTs/{f}" for f in test_image_files]

    channel_names = _normalize_channels(src)
    payload = {
        "labels": _normalize_labels(src.get("labels", {})),
        "channel_names": channel_names,
        "name": args.name or src.get("name") or _dataset_name_from_task(task_dir),
        "numTraining": len(training_cases),
        "numTest": len(test_cases),
        "file_ending": src.get("file_ending", ".nii.gz"),
        "description": src.get("description", "UNETR++ dataset.json generated from local files."),
        "licence": src.get("licence", "see challenge website"),
        "modality": channel_names,
        "reference": src.get("reference", "see challenge website"),
        "release": src.get("release", "0.0"),
        "tensorImageSize": src.get("tensorImageSize", "4D"),
        "training": training_cases,
        "test": test_cases,
    }

    with open(output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Generated UNETR++ dataset.json: {output}")
    print(f"training={len(training_cases)} test={len(test_cases)}")


if __name__ == "__main__":
    main()
