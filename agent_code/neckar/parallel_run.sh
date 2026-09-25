#!/bin/bash
# parallel_confirm.sh — N confirmation runs in parallel, 
# only the small per-run results (checkpoint + episode CSV) are kept afterward.

 
RUN="train"           # "train" or "test"
N_RUNS=3
TASK_ROUNDS=2000
SCENARIO="classic"
AGENTS="rule_based_agent"
BEHAVIOR="peaceful"
MODEL="cnn"
TASK_LABEL="task4"   # change per task — names the results folder; test mode reuses whatever run_1..run_N a prior "train" pass under this same TASK_LABEL already produced


export NECKAR_BEHAVIOR=$BEHAVIOR
export NECKAR_MODEL_TYPE=$MODEL
export NECKAR_EPISODES_PER_UPDATE=4
export NECKAR_SCENARIO=$SCENARIO
#export NECKAR_WARM_START="$(pwd)\results\task2\run_1\actor-critic-cnn-peaceful-classic-2000-rounds-0-opponents-ep2000.pt"
CORES=$(python -c "import os; print(os.cpu_count())")
THREADS_PER_RUN=$(( CORES / N_RUNS ))
[ "$THREADS_PER_RUN" -lt 1 ] && THREADS_PER_RUN=1
 
PROJECT_ROOT="$(pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/${TASK_LABEL}"
mkdir -p "$RESULTS_DIR"
 
# Wait for each background job to finish and report which ones failed
wait_all() {
  local failed=0
  for k in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$k]}"; then
      echo "  run_${RUN_IDS[$k]} FAILED — see its log in $RESULTS_DIR/run_${RUN_IDS[$k]}/"
      failed=1
    fi
  done
  return $failed
}

PIDS=()
RUN_IDS=()


# Running parallel training runs
if [ "$RUN" = "train" ]; then

  for i in $(seq 1 $N_RUNS); do
    RUN_DIR="$RESULTS_DIR/run_$i"
    mkdir -p "$RUN_DIR"
 
    (
      # Each process gets its own output directory, which is read into by train.py/callback.py
      export NECKAR_RUN_DIR="$RUN_DIR"
      export OMP_NUM_THREADS=$THREADS_PER_RUN
      export MKL_NUM_THREADS=$THREADS_PER_RUN
 
      python ../../main.py play --agents neckar $AGENTS --train 1 \
        --n-rounds $TASK_ROUNDS --scenario "$SCENARIO" --no-gui \
        > "$RUN_DIR/run_${i}.log" 2>&1
    ) &
    sleep 2
  done
 
  echo "Launched $N_RUNS training runs, waiting..."
  wait
 
  echo "Training complete. Results are already in their final location:"
  for i in $(seq 1 $N_RUNS); do
    echo "  $RESULTS_DIR/run_$i  (checkpoint(s), logs/, plots/, run_${i}.log)"
  done
 
elif [ "$RUN" = "test" ]; then
 

  # Evaluates each run's checkpoints found in run's folder against random_agent opponents
  # and writes to checkpoint_comparison.csv
  for i in $(seq 1 $N_RUNS); do
    RUN_DIR="$RESULTS_DIR/run_$i"
 
    if [ ! -d "$RUN_DIR" ]; then
      echo "Skipping run_$i — $RUN_DIR doesn't exist yet (run with RUN=\"train\" first)"
      continue
    fi
 
    (
      export OMP_NUM_THREADS=$THREADS_PER_RUN
      export MKL_NUM_THREADS=$THREADS_PER_RUN

      python run_eval.py --run-dir "$RUN_DIR" --task "$TASK_LABEL"  \
        > "$RUN_DIR/eval_${i}.log" 2>&1
    ) &
    PIDS+=($!); RUN_IDS+=($i)
    sleep 2
  done
 
  echo "Launched checkpoint evaluation for $N_RUNS runs, waiting..."
  wait_all
 
  echo "Evaluation complete. Comparison CSVs:"
  for i in $(seq 1 $N_RUNS); do
    echo "  $RESULTS_DIR/run_$i/checkpoint_comparison.csv"
  done
 
else
  echo "Unknown RUN=\"$RUN\" — expected \"train\" or \"test\""
  exit 1
fi