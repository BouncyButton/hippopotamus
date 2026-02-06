import argparse
import json
import pickle
from collections import OrderedDict

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input JSON path")
    parser.add_argument("-o", "--output", required=True, help="Output splits_final.pkl")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        payload = json.load(f)

    splits = payload["splits"]

    pkl_splits = []
    for fold in splits:
        pkl_splits.append(OrderedDict({
            "train": np.array(list(fold["train"])),
            "val": np.array(list(fold["val"])),
        }))

    with open(args.output, "wb") as f:
        pickle.dump(pkl_splits, f)
    print(f"Saved splits_final.pkl to {args.output}")


if __name__ == "__main__":
    main()
