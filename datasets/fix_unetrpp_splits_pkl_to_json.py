import argparse
import json
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Original splits_final.pkl")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        splits = pickle.load(f)

    json_splits = []
    for fold in splits:
        json_splits.append({
            "train": list(fold["train"]),
            "val": list(fold["val"]),
        })

    payload = {
        "splits": json_splits,
    }

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON splits to {args.output}")


if __name__ == "__main__":
    main()
