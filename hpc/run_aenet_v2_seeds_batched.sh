#!/bin/bash
# Run AENet v2 seeds in batches. There are three seeds, so the default
# batch size of 5 submits all three together.
# Override with:
#   BATCH_SIZE=2 bash run_aenet_v2_seeds_batched.sh

set -u

PROJECT_ROOT="${AENET_ROOT:-/path/to/aenet/AENet}"
JOB_DIR="${PROJECT_ROOT}/hpc"
BATCH_SIZE="${BATCH_SIZE:-5}"
SEEDS=(42 123 2025)

cd "${JOB_DIR}" || exit 1

submit_batch() {
  local batch=("$@")
  local job_ids=()
  local job_seeds=()
  local seed

  echo
  echo "============================================================"
  echo "Submitting AENet v2 batch of ${#batch[@]} seeds"
  echo "============================================================"

  for seed in "${batch[@]}"; do
    script="run_fastmri_knee_medium_aenet_v2_s${seed}.sh"
    output="$(jsub < "${script}")"
    echo "${output}"
    job_id="$(
      printf '%s\n' "${output}" |
        sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' |
        head -n 1
    )"
    if [[ -z "${job_id}" ]]; then
      echo "Could not parse job id for seed ${seed}. Stop."
      exit 1
    fi
    job_ids+=("${job_id}")
    job_seeds+=("${seed}")
  done

  echo "Waiting for ${#job_ids[@]} AENet v2 jobs..."
  for index in "${!job_ids[@]}"; do
    job_id="${job_ids[index]}"
    seed="${job_seeds[index]}"
    while jjobs 2>/dev/null | awk 'NR > 1 {print $1}' | grep -qx "${job_id}"; do
      sleep 60
    done

    out_file="${PROJECT_ROOT}/fastmri_knee_aenet_v2_s${seed}_${job_id}.out"
    err_file="${PROJECT_ROOT}/fastmri_knee_aenet_v2_s${seed}_${job_id}.err"
    echo "Finished AENet v2 seed ${seed}, job ${job_id}"
    if [[ -s "${out_file}" ]]; then
      tail -n 30 "${out_file}"
    fi
    if [[ -s "${err_file}" ]]; then
      echo "--- error tail ---"
      tail -n 50 "${err_file}"
    fi
  done
}

for ((start = 0; start < ${#SEEDS[@]}; start += BATCH_SIZE)); do
  batch=("${SEEDS[@]:start:BATCH_SIZE}")
  submit_batch "${batch[@]}"
done

echo
echo "AENet v2 seed batches finished."
