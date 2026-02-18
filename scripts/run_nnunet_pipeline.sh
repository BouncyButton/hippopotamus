#!/usr/bin/env bash
set -euo pipefail

# End-to-end nnUNet pipeline (updated split strategy):
# 1) ensure dataset exists locally
# 2) create deterministic holdout + train-only 5-fold CV splits
# 3) build a train-only nnUNet raw dataset copy (prevents train/test preprocessing leakage)
# 4) preprocess + train folds
# 5) upload fold models + metadata artifacts to W&B
# 6) run evaluation/evaluate.py (test split, multi-fold ensemble)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_nnunet_pipeline.sh [options]

Options:
  --dataset Dataset102_MNI          Target dataset folder name
  --mode sanity|full                sanity => nnUNetTrainer_1epoch, full => nnUNetTrainer_250epochs
  --trainer NAME                    Override trainer class
  --folds "0 1 2 3 4"               Space-separated fold list
  --test-size 0.2                   Holdout test proportion
  --seed 42                         Fixed random seed for deterministic splits
  --config 3d_fullres               nnUNet configuration
  --run-root PATH                   Working root for nnUNet raw/preprocessed/results
  --project NAME                    W&B project
  --entity NAME                     W&B entity
  --skip-dataset-create             Do not run dataset creation script
  --skip-train                      Do not train (still can evaluate existing artifacts)
  --skip-eval                       Skip evaluation/evaluate.py
  --max-cases N                     Optional evaluate.py --max-cases

Examples:
  scripts/run_nnunet_pipeline.sh --dataset Dataset102_MNI --mode sanity --folds "0 1"
  scripts/run_nnunet_pipeline.sh --dataset Dataset103_ADNI --mode full --folds "0 1 2 3 4"
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="Dataset102_MNI"
MODE="sanity"
TRAINER=""
FOLDS_STR="0"
TEST_SIZE="0.2"
SEED="42"
CONFIGURATION="3d_fullres"
RUN_ROOT=""
PROJECT="hippopotamus-project"
ENTITY="hippopotamus"
SKIP_DATASET_CREATE="1"
SKIP_TRAIN="0"
SKIP_EVAL="0"
MAX_CASES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --trainer) TRAINER="$2"; shift 2 ;;
    --folds) FOLDS_STR="$2"; shift 2 ;;
    --test-size) TEST_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --config) CONFIGURATION="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --entity) ENTITY="$2"; shift 2 ;;
    --skip-dataset-create) SKIP_DATASET_CREATE="1"; shift ;;
    --skip-train) SKIP_TRAIN="1"; shift ;;
    --skip-eval) SKIP_EVAL="1"; shift ;;
    --max-cases) MAX_CASES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$TRAINER" ]]; then
  if [[ "$MODE" == "sanity" ]]; then
    TRAINER="nnUNetTrainer_1epoch"
  else
    TRAINER="nnUNetTrainer_250epochs"
  fi
fi

if [[ -z "$RUN_ROOT" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_BASE="$(dirname "${REPO_ROOT}")/hippopotamus_runs"
  RUN_ROOT="${RUN_BASE}/nnunet_${DATASET}_${MODE}_${TS}"
fi

DATASET_ID="$(echo "$DATASET" | sed -E 's/^Dataset([0-9]+)_.+$/\1/')"
DATASET_CODE="$(echo "$DATASET" | cut -d'_' -f2)"
SPLITS_FILE="splits_final_${SEED}.json"
HOLDOUT_FILE="train_test_split_${SEED}.json"
IFS=' ' read -r -a FOLDS <<< "$FOLDS_STR"

FULL_DATASET_DIR="${REPO_ROOT}/datasets/${DATASET}"
FULL_WORK_ROOT="${RUN_ROOT}/datasets_full"
FULL_WORK_DATASET_DIR="${FULL_WORK_ROOT}/${DATASET}"
RAW_ROOT="${RUN_ROOT}/datasets"
PREPROC_ROOT="${RUN_ROOT}/preprocessed"
RESULTS_ROOT="${RUN_ROOT}/results"
TRAIN_DATASET_DIR="${RAW_ROOT}/${DATASET}"
EVAL_OUTPUT_DIR="${RUN_ROOT}/evaluation_output"

echo "[INFO] repo_root=${REPO_ROOT}"
echo "[INFO] dataset=${DATASET} (id=${DATASET_ID}, code=${DATASET_CODE})"
echo "[INFO] mode=${MODE}, trainer=${TRAINER}, folds=${FOLDS_STR}"
echo "[INFO] split_files: cv=${SPLITS_FILE}, holdout=${HOLDOUT_FILE}"
echo "[INFO] run_root=${RUN_ROOT}"

if ! command -v nnUNetv2_train >/dev/null 2>&1; then
  echo "[INFO] nnUNet CLI not found, installing local nnUNet baseline package..."
  python3 -m pip install -e "${REPO_ROOT}/baselines/nnUNet"
fi

mkdir -p "${RUN_ROOT}" "${FULL_WORK_ROOT}" "${RAW_ROOT}" "${PREPROC_ROOT}" "${RESULTS_ROOT}" "${EVAL_OUTPUT_DIR}"

if [[ "${SKIP_DATASET_CREATE}" != "1" ]]; then
  case "$DATASET" in
    Dataset101_MSD) CREATE_SCRIPT="${REPO_ROOT}/datasets/Dataset101_MSD/create_msd_dataset.py" ;;
    Dataset102_MNI) CREATE_SCRIPT="${REPO_ROOT}/datasets/Dataset102_MNI/create_mni_dataset.py" ;;
    Dataset103_ADNI) CREATE_SCRIPT="${REPO_ROOT}/datasets/Dataset103_ADNI/create_adni_dataset.py" ;;
    Dataset105_COBRA) CREATE_SCRIPT="${REPO_ROOT}/datasets/Dataset105_COBRA/create_cobra_dataset.py" ;;
    *) echo "[ERROR] Unsupported dataset: ${DATASET}"; exit 1 ;;
  esac
  echo "[INFO] Creating/syncing dataset into ${FULL_WORK_DATASET_DIR} using ${CREATE_SCRIPT}"
  (cd "${RUN_ROOT}" && python3 "${CREATE_SCRIPT}" --target "${FULL_WORK_DATASET_DIR}")
fi

resolve_dataset_dir() {
  local base="$1"
  local name="$2"
  if [[ -d "${base}/imagesTr" && -d "${base}/labelsTr" ]]; then
    echo "${base}"
    return
  fi
  if [[ -d "${base}/${name}/imagesTr" && -d "${base}/${name}/labelsTr" ]]; then
    echo "${base}/${name}"
    return
  fi
  local found
  found="$(find "${base}" -maxdepth 3 -type d -name imagesTr 2>/dev/null | head -n1 || true)"
  if [[ -n "${found}" ]]; then
    echo "$(dirname "${found}")"
    return
  fi
  echo "${base}"
}

REPO_DATASET_DIR_RESOLVED="$(resolve_dataset_dir "${FULL_DATASET_DIR}" "${DATASET}")"
if [[ "${SKIP_DATASET_CREATE}" != "1" ]]; then
  SOURCE_DATASET_DIR="${FULL_WORK_DATASET_DIR}"
else
  SOURCE_DATASET_DIR="${FULL_DATASET_DIR}"
fi

SOURCE_DATASET_DIR_RESOLVED="$(resolve_dataset_dir "${SOURCE_DATASET_DIR}" "${DATASET}")"
echo "[INFO] source_dataset_dir_resolved=${SOURCE_DATASET_DIR_RESOLVED}"
if [[ ! -d "${SOURCE_DATASET_DIR_RESOLVED}/imagesTr" || ! -d "${SOURCE_DATASET_DIR_RESOLVED}/labelsTr" ]]; then
  echo "[ERROR] Could not find imagesTr/labelsTr under ${SOURCE_DATASET_DIR}"
  exit 1
fi

# Copy full dataset outside repository so all subsequent writes happen in RUN_ROOT.
if [[ "${SOURCE_DATASET_DIR_RESOLVED}" != "${FULL_WORK_DATASET_DIR}" ]]; then
  echo "[INFO] Copying full dataset to writable working location: ${FULL_WORK_DATASET_DIR}"
  rm -rf "${FULL_WORK_DATASET_DIR}"
  mkdir -p "${FULL_WORK_DATASET_DIR}"
  cp -a "${SOURCE_DATASET_DIR_RESOLVED}/." "${FULL_WORK_DATASET_DIR}/"
fi

# Prefer existing seed-specific split files from repo dataset; otherwise generate.
if [[ -f "${REPO_DATASET_DIR_RESOLVED}/${SPLITS_FILE}" && -f "${REPO_DATASET_DIR_RESOLVED}/${HOLDOUT_FILE}" ]]; then
  echo "[INFO] Reusing existing seed-specific splits from repo dataset"
  cp "${REPO_DATASET_DIR_RESOLVED}/${SPLITS_FILE}" "${FULL_WORK_DATASET_DIR}/${SPLITS_FILE}"
  cp "${REPO_DATASET_DIR_RESOLVED}/${HOLDOUT_FILE}" "${FULL_WORK_DATASET_DIR}/${HOLDOUT_FILE}"
elif [[ -f "${FULL_WORK_DATASET_DIR}/${SPLITS_FILE}" && -f "${FULL_WORK_DATASET_DIR}/${HOLDOUT_FILE}" ]]; then
  echo "[INFO] Reusing existing seed-specific splits from working dataset"
else
  echo "[INFO] Generating deterministic holdout + train-only CV splits"
  python3 "${REPO_ROOT}/datasets/generate_holdout_cv_splits.py" \
    --dataset-dir "${FULL_WORK_DATASET_DIR}" \
    --seed "${SEED}" \
    --test-size "${TEST_SIZE}" \
    --n-folds 5 \
    --holdout-output "${HOLDOUT_FILE}" \
    --cv-output "${SPLITS_FILE}"
fi

# Keep canonical compatibility filenames as aliases/copies.
cp "${FULL_WORK_DATASET_DIR}/${SPLITS_FILE}" "${FULL_WORK_DATASET_DIR}/splits_final_train.json"
cp "${FULL_WORK_DATASET_DIR}/${HOLDOUT_FILE}" "${FULL_WORK_DATASET_DIR}/train_test_split.json"

echo "[INFO] Building train-only dataset copy for preprocessing/training"
mkdir -p "${TRAIN_DATASET_DIR}/imagesTr" "${TRAIN_DATASET_DIR}/labelsTr"
cp "${FULL_WORK_DATASET_DIR}/dataset.json" "${TRAIN_DATASET_DIR}/dataset.json"
python3 - <<PY
import json
import shutil
from pathlib import Path

full_dir = Path("${FULL_WORK_DATASET_DIR}")
train_dir = Path("${TRAIN_DATASET_DIR}")
holdout = json.load(open(full_dir / "train_test_split.json"))
train_cases = holdout["train"]

for case in train_cases:
    src_img = full_dir / "imagesTr" / f"{case}_0000.nii.gz"
    src_lbl = full_dir / "labelsTr" / f"{case}.nii.gz"
    if not src_img.exists() or not src_lbl.exists():
        raise FileNotFoundError(f"Missing file for case {case}: {src_img} / {src_lbl}")
    shutil.copy(src_img, train_dir / "imagesTr" / src_img.name)
    shutil.copy(src_lbl, train_dir / "labelsTr" / src_lbl.name)
print(f"Copied {len(train_cases)} train cases to {train_dir}")
PY

# Keep split files in RUN_ROOT/datasets/<DATASET> so evaluate.py can resolve them via --repo-root RUN_ROOT.
cp "${FULL_WORK_DATASET_DIR}/train_test_split.json" "${TRAIN_DATASET_DIR}/train_test_split.json"
cp "${FULL_WORK_DATASET_DIR}/splits_final_train.json" "${TRAIN_DATASET_DIR}/splits_final_train.json"
cp "${FULL_WORK_DATASET_DIR}/${HOLDOUT_FILE}" "${TRAIN_DATASET_DIR}/${HOLDOUT_FILE}"
cp "${FULL_WORK_DATASET_DIR}/${SPLITS_FILE}" "${TRAIN_DATASET_DIR}/${SPLITS_FILE}"

export nnUNet_raw="${RAW_ROOT}"
export nnUNet_preprocessed="${PREPROC_ROOT}"
export nnUNet_results="${RESULTS_ROOT}"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[INFO] Running nnUNet plan + preprocess (train-only dataset)"
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity

  echo "[INFO] Replacing preprocessed splits with train-only 5-fold splits"
  cp "${FULL_WORK_DATASET_DIR}/${SPLITS_FILE}" "${PREPROC_ROOT}/${DATASET}/splits_final.json"

  for fold in "${FOLDS[@]}"; do
    echo "[INFO] Training fold ${fold} with ${TRAINER}"
    nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${fold}" -tr "${TRAINER}"

    MODEL_PATH="${RESULTS_ROOT}/${DATASET}/${TRAINER}__nnUNetPlans__${CONFIGURATION}/fold_${fold}"
    echo "[INFO] Uploading fold ${fold} model artifact from ${MODEL_PATH}"
    python3 "${REPO_ROOT}/baselines/save_nnunet_run.py" \
      --model-path "${MODEL_PATH}" \
      --fold "${fold}" \
      --dataset "${DATASET_CODE}"
  done

  echo "[INFO] Packaging nnUNet metadata artifact"
  python3 "${REPO_ROOT}/evaluation/create_nnunet_metadata.py" \
    --dataset-id "${DATASET_ID}" \
    --dataset-name "${DATASET}" \
    --dataset-code "${DATASET_CODE}" \
    --root "${RUN_ROOT}" \
    --project "${PROJECT}" \
    --entity "${ENTITY}"
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  # Evaluate uses holdout test cases and fold-ensemble inference for nnUNet.
  export nnUNet_raw="${FULL_WORK_ROOT}"
  export nnUNet_preprocessed="${PREPROC_ROOT}"
  export nnUNet_results="${RESULTS_ROOT}"
  mkdir -p "${EVAL_OUTPUT_DIR}"

  EVAL_CMD=(
    python3 "${REPO_ROOT}/evaluation/evaluate.py"
    --project "${PROJECT}"
    --entity "${ENTITY}"
    --methods nnunet
    --datasets "${DATASET_CODE}"
    --folds "${FOLDS[@]}"
    --checkpoint final
    --output-dir "${EVAL_OUTPUT_DIR}"
    --eval-split test
    --cv-splits-name "${SPLITS_FILE}"
    --holdout-split-name "${HOLDOUT_FILE}"
    --repo-root "${RUN_ROOT}"
  )
  if [[ -n "${MAX_CASES}" ]]; then
    EVAL_CMD+=(--max-cases "${MAX_CASES}")
  fi

  echo "[INFO] Running evaluation: ${EVAL_CMD[*]}"
  "${EVAL_CMD[@]}"
fi

echo "[DONE] Pipeline completed."
