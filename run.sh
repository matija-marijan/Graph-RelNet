#!/bin/bash
set -euo pipefail

# ---- settings ----
export TQDM_DISABLE=1
MAX_JOBS=3                      # how many trainings in parallel after seed0
LOGDIR="logs"
SNRS=(100 30 5 0 -5 -10)
SEEDS_REST=(1 2 3 4)            # seed 0 is the "first" one we wait on
mkdir -p "$LOGDIR"

# Kill all background jobs on Ctrl-C/TERM
trap 'echo -e "\n[TRAP] Stopping..."; jobs -p | xargs -r kill 2>/dev/null; wait || true; exit 130' INT TERM

wait_for_slot () {
  while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
    # don't abort the whole script if a job fails (e.g., OOM)
    wait -n || echo "[WARN] a job exited non-zero; check logs."
  done
}

launch_train () {
  local model="$1" seed="$2" desc="$3"
  wait_for_slot
  echo "[LAUNCH] $model seed=$seed desc=$desc"
  # If you add more GPUs later, you can pin: export CUDA_VISIBLE_DEVICES=$((seed % 2))
  python training.py \
    --model "$model" \
    --edge_processing gcc_phat \
    --batch_size 64 \
    --validate --save \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --scheduler --wandb \
    --seed "$seed" \
    --description "$desc" \
    |& tee "${LOGDIR}/${model}_s${seed}_${desc}.log" &
}

wait_all () {
  while (( $(jobs -r | wc -l) > 0 )); do
    wait -n || echo "[WARN] a job exited non-zero; check logs."
  done
}

run_snr_block () {
  local snr="$1"
  local desc="SNR_${snr}_random"
  echo -e "\n================ SNR ${snr} ================\n"

  # 1) Build datasets in parallel, then wait only for them
  python create_dataset.py --num_arrays 10 --num_signals 20 --num_angles 50 \
    --signal_type random --num_samples 1024 --array_type random --seed 42 \
    --SNR "$snr" --sampling_frequency 16000 &
  python create_dataset.py --num_arrays 5 --num_signals 20 --num_angles 50 \
    --signal_type timit --num_samples 1024 --array_type random --seed 42 \
    --validation --SNR "$snr" --sampling_frequency 16000 &
  wait_all
  echo "[INFO] Datasets ready for SNR ${snr}."

  # 2) First training: RelNet seed 0 (foreground, block until fully done)
  echo "[FIRST] RelNet seed=0 desc=${desc}"
  python training.py \
    --model RelNet \
    --edge_processing gcc_phat \
    --batch_size 64 \
    --validate --save \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --scheduler --wandb \
    --seed 0 \
    --description "$desc" \
    |& tee "${LOGDIR}/RelNet_s0_${desc}.log"
  echo "[INFO] First training completed for SNR ${snr}."

  # 3) Remaining trainings with throttled parallelism
  # 3a) RelNet seeds 1–4
  for s in "${SEEDS_REST[@]}"; do launch_train "RelNet" "$s" "$desc"; done
  wait_all
  echo "[INFO] RelNet (seeds 1–4) finished for SNR ${snr}."

  # 3b) Graph_RelNet seeds 0–4
  for s in 0 1 2 3 4; do launch_train "Graph_RelNet" "$s" "$desc"; done
  wait_all
  echo "[INFO] Graph_RelNet finished for SNR ${snr}."

  # 4) Evaluation
  python -m evaluation.delay_and_sum_evaluation \
    --validate --wandb \
    --signal_processing raw --edge_processing gcc_phat \
    --fs 16000 --save --resolution 1 \
    --description "$desc"
}

# ---- run all SNRs ----
for snr in "${SNRS[@]}"; do
  run_snr_block "$snr"
done

echo -e "\n[DONE] All SNR blocks completed."
