#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_unetrpp_pipeline.sh [options]

Options:
  --dataset Dataset102_MNI          Target dataset folder name
  --mode sanity|full                sanity => fold 0 only, full => provided --folds
  --trainer NAME                    UNETR++ trainer class (default: unetr_pp_trainer_general_purpose)
  --folds "0 1 2 3 4"               Space-separated fold list
  --test-size 0.2                   Holdout-only: test proportion
  --seed 42                         Random seed for deterministic splitting
  --outer_fold_idx 0                Nested mode: outer fold index to materialize (0..4)
  --split-scheme nested|holdout     Split strategy (default: nested)
  --inner-n-folds 5                 Nested mode: number of inner folds
  --inner-seed 42                   Nested mode: seed for inner-fold ordering
  --config 3d_fullres               UNETR++ configuration
  --crop-size "48 56 40"            Crop size for run_training
  --run-root PATH                   Working root for raw/preprocessed/results
  --project NAME                    W&B project
  --entity NAME                     W&B entity
  --env-name NAME                   Conda env name (default: nnunetrpp_py38)
  --skip-dataset-create             Do not run dataset creation script
  --skip-train                      Do not train
  --skip-eval                       Skip evaluation/evaluate.py
  --max-cases N                     Optional evaluate.py --max-cases

Examples:
  scripts/run_unetrpp_pipeline.sh --dataset Dataset102_MNI --mode sanity
  scripts/run_unetrpp_pipeline.sh --dataset Dataset103_ADNI --mode full --folds "0 1 2 3 4"
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="nnunetrpp_py38"
DATASET="Dataset102_MNI"
MODE="sanity"
TRAINER="unetr_pp_trainer_general_purpose"
FOLDS_STR="0"
TEST_SIZE="0.2"
SEED="42"
OUTER_FOLD_IDX="0"
SPLIT_SCHEME="nested"
INNER_N_FOLDS="5"
INNER_SEED="42"
CONFIGURATION="3d_fullres"
CROP_SIZE="48 56 40"
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
    --outer_fold_idx) OUTER_FOLD_IDX="$2"; shift 2 ;;
    --split-scheme) SPLIT_SCHEME="$2"; shift 2 ;;
    --inner-n-folds) INNER_N_FOLDS="$2"; shift 2 ;;
    --inner-seed) INNER_SEED="$2"; shift 2 ;;
    --config) CONFIGURATION="$2"; shift 2 ;;
    --crop-size) CROP_SIZE="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --entity) ENTITY="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --skip-dataset-create) SKIP_DATASET_CREATE="1"; shift ;;
    --skip-train) SKIP_TRAIN="1"; shift ;;
    --skip-eval) SKIP_EVAL="1"; shift ;;
    --max-cases) MAX_CASES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ "${MODE}" == "sanity" ]]; then
  FOLDS_STR="0"
fi

if [[ -z "${RUN_ROOT}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_BASE="$(dirname "${REPO_ROOT}")/hippopotamus_runs"
  RUN_ROOT="${RUN_BASE}/unetrpp_${DATASET}_${MODE}_${TS}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Run scripts/setup_unetrpp_env.sh first."
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

DATASET_ID="$(echo "${DATASET}" | sed -E 's/^Dataset([0-9]+)_.+$/\1/')"
DATASET_CODE="$(echo "${DATASET}" | cut -d'_' -f2)"
TASK_NAME="Task${DATASET_ID}_${DATASET_CODE}"

if [[ "${SPLIT_SCHEME}" == "nested" ]]; then
  SPLITS_FILE="splits_final_outer${OUTER_FOLD_IDX}.json"
  HOLDOUT_FILE="train_test_split_outer${OUTER_FOLD_IDX}.json"
else
  SPLITS_FILE="splits_final_${SEED}.json"
  HOLDOUT_FILE="train_test_split_${SEED}.json"
fi

IFS=' ' read -r -a FOLDS <<< "${FOLDS_STR}"

FULL_DATASET_DIR="${REPO_ROOT}/datasets/${DATASET}"
FULL_WORK_ROOT="${RUN_ROOT}/datasets_full"
FULL_WORK_DATASET_DIR="${FULL_WORK_ROOT}/${DATASET}"
RAW_BASE="${RUN_ROOT}/datasets"
RAW_DATA_DIR="${RAW_BASE}/unetr_pp_raw_data"
TASK_DATASET_DIR="${RAW_DATA_DIR}/${TASK_NAME}"
PREPROC_ROOT="${RUN_ROOT}/preprocessed"
RESULTS_ROOT="${RUN_ROOT}/results"
EVAL_OUTPUT_DIR="${RUN_ROOT}/evaluation_output"

echo "[INFO] repo_root=${REPO_ROOT}"
echo "[INFO] env_name=${ENV_NAME}"
echo "[INFO] dataset=${DATASET} (id=${DATASET_ID}, code=${DATASET_CODE}, task=${TASK_NAME})"
echo "[INFO] mode=${MODE}, trainer=${TRAINER}, folds=${FOLDS_STR}, crop_size=${CROP_SIZE}"
echo "[INFO] split_scheme=${SPLIT_SCHEME}, seed=${SEED}, outer_fold_idx=${OUTER_FOLD_IDX}, inner_n_folds=${INNER_N_FOLDS}, inner_seed=${INNER_SEED}"
echo "[INFO] split_files: cv=${SPLITS_FILE}, holdout=${HOLDOUT_FILE}"
echo "[INFO] run_root=${RUN_ROOT}"

mkdir -p "${RUN_ROOT}" "${FULL_WORK_ROOT}" "${RAW_DATA_DIR}" "${PREPROC_ROOT}" "${RESULTS_ROOT}" "${EVAL_OUTPUT_DIR}"

case "${DATASET}" in
  Dataset101_MSD) CREATE_SCRIPT="${REPO_ROOT}/datasets/Dataset101_MSD/create_msd_dataset.py" ;;
  Dataset102_MNI) CREATE_SCRIPT="${REPO_ROOT}/datasets/Dataset102_MNI/create_mni_dataset.py" ;;
  Dataset103_ADNI) CREATE_SCRIPT="${REPO_ROOT}/datasets/Dataset103_ADNI/create_adni_dataset.py" ;;
  Dataset105_COBRA) CREATE_SCRIPT="${REPO_ROOT}/datasets/Dataset105_COBRA/create_cobra_dataset.py" ;;
  *) echo "[ERROR] Unsupported dataset: ${DATASET}"; exit 1 ;;
esac

if [[ "${SKIP_DATASET_CREATE}" != "1" ]]; then
  echo "[INFO] Creating/syncing dataset into ${FULL_WORK_DATASET_DIR} using ${CREATE_SCRIPT}"
  (cd "${RUN_ROOT}" && python "${CREATE_SCRIPT}" --target "${FULL_WORK_DATASET_DIR}")
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

cleanup_macos_resource_forks() {
  local root="$1"
  if [[ -d "${root}" ]]; then
    find "${root}" -type f -name '._*' -delete || true
  fi
}

if [[ "${SKIP_DATASET_CREATE}" != "1" ]]; then
  SOURCE_DATASET_DIR="${FULL_WORK_DATASET_DIR}"
else
  SOURCE_DATASET_DIR="${FULL_DATASET_DIR}"
fi
SOURCE_DATASET_DIR_RESOLVED="$(resolve_dataset_dir "${SOURCE_DATASET_DIR}" "${DATASET}")"
echo "[INFO] source_dataset_dir_resolved=${SOURCE_DATASET_DIR_RESOLVED}"

if [[ ! -d "${SOURCE_DATASET_DIR_RESOLVED}/imagesTr" || ! -d "${SOURCE_DATASET_DIR_RESOLVED}/labelsTr" ]]; then
  if [[ "${SKIP_DATASET_CREATE}" == "1" ]]; then
    echo "[WARN] Source dataset has no imagesTr/labelsTr. Auto-creating dataset in ${FULL_WORK_DATASET_DIR}"
    (cd "${RUN_ROOT}" && python "${CREATE_SCRIPT}" --target "${FULL_WORK_DATASET_DIR}")
    SOURCE_DATASET_DIR_RESOLVED="$(resolve_dataset_dir "${FULL_WORK_DATASET_DIR}" "${DATASET}")"
    echo "[INFO] source_dataset_dir_resolved(after create)=${SOURCE_DATASET_DIR_RESOLVED}"
  fi
fi

if [[ ! -d "${SOURCE_DATASET_DIR_RESOLVED}/imagesTr" || ! -d "${SOURCE_DATASET_DIR_RESOLVED}/labelsTr" ]]; then
  echo "[ERROR] Could not find imagesTr/labelsTr under ${SOURCE_DATASET_DIR_RESOLVED}"
  exit 1
fi

if [[ "${SOURCE_DATASET_DIR_RESOLVED}" != "${FULL_WORK_DATASET_DIR}" ]]; then
  echo "[INFO] Copying full dataset to writable working location: ${FULL_WORK_DATASET_DIR}"
  rm -rf "${FULL_WORK_DATASET_DIR}"
  mkdir -p "${FULL_WORK_DATASET_DIR}"
  cp -a "${SOURCE_DATASET_DIR_RESOLVED}/." "${FULL_WORK_DATASET_DIR}/"
fi
echo "[INFO] Cleaning macOS resource-fork files under ${FULL_WORK_DATASET_DIR}"
cleanup_macos_resource_forks "${FULL_WORK_DATASET_DIR}"

for f in "${SPLITS_FILE}" "splits_final_train.json" "${HOLDOUT_FILE}" "train_test_split.json"; do
  rm -f "${FULL_WORK_DATASET_DIR}/${f}"
done

echo "[INFO] Generating deterministic splits"
if [[ "${SPLIT_SCHEME}" == "nested" ]]; then
  python "${REPO_ROOT}/datasets/generate_holdout_cv_splits.py" \
    --dataset-dir "${FULL_WORK_DATASET_DIR}" \
    --split-scheme nested \
    --outer-fold "${OUTER_FOLD_IDX}" \
    --outer-n-folds 5 \
    --n-folds "${INNER_N_FOLDS}" \
    --seed "${SEED}" \
    --inner-seed "${INNER_SEED}" \
    --holdout-output "${HOLDOUT_FILE}" \
    --cv-output "${SPLITS_FILE}"
else
  python "${REPO_ROOT}/datasets/generate_holdout_cv_splits.py" \
    --dataset-dir "${FULL_WORK_DATASET_DIR}" \
    --split-scheme holdout \
    --seed "${SEED}" \
    --test-size "${TEST_SIZE}" \
    --n-folds 5 \
    --holdout-output "${HOLDOUT_FILE}" \
    --cv-output "${SPLITS_FILE}"
fi

cp "${FULL_WORK_DATASET_DIR}/${SPLITS_FILE}" "${FULL_WORK_DATASET_DIR}/splits_final_train.json"
cp "${FULL_WORK_DATASET_DIR}/${HOLDOUT_FILE}" "${FULL_WORK_DATASET_DIR}/train_test_split.json"

echo "[INFO] Building train-only UNETR++ task dataset"
rm -rf "${TASK_DATASET_DIR}"
mkdir -p "${TASK_DATASET_DIR}/imagesTr" "${TASK_DATASET_DIR}/labelsTr"

DATASET_JSON_SRC=""
for c in "${FULL_WORK_DATASET_DIR}/dataset.json" "${SOURCE_DATASET_DIR_RESOLVED}/dataset.json"; do
  if [[ -f "${c}" ]]; then
    DATASET_JSON_SRC="${c}"
    break
  fi
done
if [[ -z "${DATASET_JSON_SRC}" ]]; then
  echo "[ERROR] dataset.json not found under ${FULL_WORK_DATASET_DIR} or ${SOURCE_DATASET_DIR_RESOLVED}"
  exit 1
fi

python - <<PY
import json
import shutil
from pathlib import Path

full_dir = Path("${FULL_WORK_DATASET_DIR}")
task_dir = Path("${TASK_DATASET_DIR}")
holdout = json.load(open(full_dir / "train_test_split.json"))
train_cases = [c for c in holdout["train"] if not str(c).startswith("._")]
dataset_json_src = Path("${DATASET_JSON_SRC}")

dataset_json = json.load(open(dataset_json_src))
dataset_json["numTraining"] = len(train_cases)
if "training" in dataset_json and isinstance(dataset_json["training"], list):
    allowed = set(train_cases)
    filtered = []
    for item in dataset_json["training"]:
        if isinstance(item, dict):
            image_path = str(item.get("image", ""))
            label_path = str(item.get("label", ""))
            image_case = Path(image_path).name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
            label_case = Path(label_path).name.replace(".nii.gz", "")
            if image_case in allowed or label_case in allowed:
                filtered.append(item)
    if filtered:
        dataset_json["training"] = filtered
with open(task_dir / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

for case in train_cases:
    src_img = full_dir / "imagesTr" / f"{case}_0000.nii.gz"
    src_lbl = full_dir / "labelsTr" / f"{case}.nii.gz"
    if not src_img.exists() or not src_lbl.exists():
        raise FileNotFoundError(f"Missing file for case {case}: {src_img} / {src_lbl}")
    shutil.copy(src_img, task_dir / "imagesTr" / src_img.name)
    shutil.copy(src_lbl, task_dir / "labelsTr" / src_lbl.name)
print(f"Copied {len(train_cases)} train cases to {task_dir}")
PY

mkdir -p "${RAW_BASE}/${DATASET}"
cp "${FULL_WORK_DATASET_DIR}/train_test_split.json" "${RAW_BASE}/${DATASET}/train_test_split.json"
cp "${FULL_WORK_DATASET_DIR}/splits_final_train.json" "${RAW_BASE}/${DATASET}/splits_final_train.json"
cp "${FULL_WORK_DATASET_DIR}/${HOLDOUT_FILE}" "${RAW_BASE}/${DATASET}/${HOLDOUT_FILE}"
cp "${FULL_WORK_DATASET_DIR}/${SPLITS_FILE}" "${RAW_BASE}/${DATASET}/${SPLITS_FILE}"

export RESULTS_FOLDER="${RESULTS_ROOT}"
export unetr_pp_preprocessed="${PREPROC_ROOT}"
export unetr_pp_raw_data_base="${RAW_BASE}"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  rm -rf "${PREPROC_ROOT:?}/${TASK_NAME}"

  echo "[INFO] Running UNETR++ plan + preprocess"
  (
    cd "${REPO_ROOT}/baselines/unetr_plus_plus"
    python -m unetr_pp.experiment_planning.nnFormer_plan_and_preprocess -t "${DATASET_ID}"
  )

  PREPROC_TASK_DIR="${PREPROC_ROOT}/${TASK_NAME}"
  if [[ ! -f "${PREPROC_TASK_DIR}/splits_final.pkl" ]]; then
    echo "[ERROR] Expected split file not found: ${PREPROC_TASK_DIR}/splits_final.pkl"
    exit 1
  fi

  echo "[INFO] Fixing UNETR++ split file to enforce patient-safe grouping"
  python "${REPO_ROOT}/datasets/fix_unetrpp_splits_pkl_to_json.py" \
    -i "${PREPROC_TASK_DIR}/splits_final.pkl" \
    -o "${PREPROC_TASK_DIR}/splits_final.json"
  python "${REPO_ROOT}/datasets/fix_unetrpp_splits_json.py" \
    -i "${PREPROC_TASK_DIR}/splits_final.json" \
    -o "${PREPROC_TASK_DIR}/splits_final_fixed.json" \
    --dataset "${DATASET_CODE}"
  python "${REPO_ROOT}/datasets/fix_unetrpp_splits_json_to_pkl.py" \
    -i "${PREPROC_TASK_DIR}/splits_final_fixed.json" \
    -o "${PREPROC_TASK_DIR}/splits_final.pkl"

  CROP_ARGS=()
  if [[ -n "${CROP_SIZE}" ]]; then
    IFS=' ' read -r -a CROP_ARR <<< "${CROP_SIZE}"
    CROP_ARGS=(--crop_size "${CROP_ARR[@]}")
  fi

  for fold in "${FOLDS[@]}"; do
    echo "[INFO] Training fold ${fold} with ${TRAINER}"
    (
      cd "${REPO_ROOT}/baselines/unetr_plus_plus"
      python -m unetr_pp.run.run_training "${CONFIGURATION}" "${TRAINER}" "${DATASET_ID}" "${fold}" "${CROP_ARGS[@]}"
    )

    MODEL_PATH="${RESULTS_ROOT}/unetr_pp/${CONFIGURATION}/${TASK_NAME}/${TRAINER}__unetr_pp_Plansv2.1/fold_${fold}"
    echo "[INFO] Uploading fold ${fold} model artifact from ${MODEL_PATH}"
    python "${REPO_ROOT}/baselines/save_unetrpp_run.py" \
      --model-path "${MODEL_PATH}" \
      --fold "${fold}" \
      --dataset "${DATASET_CODE}"
  done
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  mkdir -p "${EVAL_OUTPUT_DIR}"
  EVAL_CMD=(
    python "${REPO_ROOT}/evaluation/evaluate.py"
    --project "${PROJECT}"
    --entity "${ENTITY}"
    --methods unetrpp
    --datasets "${DATASET_CODE}"
    --folds "${FOLDS[@]}"
    --checkpoint final
    --output-dir "${EVAL_OUTPUT_DIR}"
    --eval-split test
    --cv-splits-name "${SPLITS_FILE}"
    --holdout-split-name "${HOLDOUT_FILE}"
    --repo-root "${RUN_ROOT}"
    --unetrpp-python "$(command -v python)"
  )
  if [[ -n "${MAX_CASES}" ]]; then
    EVAL_CMD+=(--max-cases "${MAX_CASES}")
  fi
  echo "[INFO] Running evaluation: ${EVAL_CMD[*]}"
  "${EVAL_CMD[@]}"
fi

echo "[DONE] UNETR++ pipeline completed."
