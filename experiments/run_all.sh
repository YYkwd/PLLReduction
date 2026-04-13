#!/usr/bin/env bash
# ============================================================================
# PLL Reduction - Full Experiment Runner
#
# Usage:
#   bash experiments/run_all.sh --phase 1          # Phase 1 only (tuning)
#   bash experiments/run_all.sh --phase 2          # Phase 2 only (benchmark)
#   bash experiments/run_all.sh --phase 3          # Phase 3 only (optional)
#   bash experiments/run_all.sh --phase all        # All phases sequentially
#   bash experiments/run_all.sh --batch 1a         # Single batch
#   bash experiments/run_all.sh --dry-run          # Preview commands
#   bash experiments/run_all.sh --dry-run --batch 7a
#
# Phases:
#   1  Parameter tuning  (batches 1a,1b,2,3a,3b,4a,4b  - all parallel)
#   2  Benchmark + CIFAR10 (batches 5a,5b,5c,7a,7b     - needs phase 1 results)
#   3  Optional refinement (batches 6a,6b,7c)
# ============================================================================
set -uo pipefail

PHASE=""
BATCH=""
DRY_RUN=false
LOGDIR="logs/$(date +%Y%m%d_%H%M%S)"
PYTHON="${PYTHON:-python}"
FAILED=0

usage() {
    echo "Usage: $0 [--phase 1|2|3|all] [--batch ID] [--dry-run] [--python PATH]"
    echo ""
    echo "Options:"
    echo "  --phase PHASE   Run a full phase (1, 2, 3, or all)"
    echo "  --batch ID      Run a single batch (1a,1b,2,3a,3b,4a,4b,4c,5a,5b,5c,5s,6a,6b,7a,7b,7c,7e,7s)"
    echo "  --dry-run       Print commands without executing"
    echo "  --python PATH   Python interpreter (default: python)"
    echo ""
    echo "Batches:"
    echo "  Phase 1 (tuning, all parallel):"
    echo "    1a  SR x CB ablation on core datasets"
    echo "    1b  SR x CB ablation on large datasets"
    echo "    2   warmup diagnosis"
    echo "    3a  r_min sensitivity"
    echo "    3b  alpha sensitivity"
    echo "    4a  target_d sweep (sdlpp_sr_cb)"
    echo "    4b  miu sweep"
    echo "    4c  target_d sweep (sdlpp_baseline, 6 primary datasets; not in phase 1)"
    echo "  Phase 2 (benchmark, needs phase 1 results):"
    echo "    5a  Core SDLPP benchmark (baseline + sr_cb, 4 datasets)"
    echo "    5b  Large SDLPP benchmark (2 datasets)"
    echo "    5c  Main classifier comparison: 6 datasets x SDLPP x KNN/IPAL"
    echo "    5s  SURE clf: Lost + Soccer Player x SDLPP baseline vs SR+CB"
    echo "    7a  CIFAR10 SR x CB ablation"
    echo "    7b  CIFAR10 SDLPP-only benchmark"
    echo "    7e  CIFAR10 ResNet18 embedding x SDLPP x KNN/IPAL (needs .mat from cifar10_mat_to_resnet18.py)"
    echo "  Phase 3 (optional refinement):"
    echo "    6a  warmup x r_min joint sweep"
    echo "    6b  adaptive CB threshold sweep"
    echo "    7c  CIFAR10 multi-seed robustness"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)  PHASE="$2"; shift 2 ;;
        --batch)  BATCH="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --python) PYTHON="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$PHASE" && -z "$BATCH" ]]; then
    echo "Error: specify --phase or --batch"
    usage
fi

# ---------------------------------------------------------------------------
# Helper: run a single experiment command
# ---------------------------------------------------------------------------
run_cmd() {
    local name="$1"; shift
    if $DRY_RUN; then
        echo "[DRY-RUN] $name:"
        printf "  "
        printf '%q ' "$@"
        echo ""
        echo ""
        return 0
    fi
    mkdir -p "$LOGDIR"
    local logfile="${LOGDIR}/${name}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START  $name" | tee -a "$logfile"
    if "$@" >> "$logfile" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE   $name  (OK)" | tee -a "$logfile"
    else
        local rc=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL   $name  (exit=$rc)" | tee -a "$logfile"
        FAILED=$((FAILED + 1))
    fi
}

# ============================================================================
# Batch definitions
# ============================================================================

# Core = tabular PLL sets with existing .mat (slashdot f2/f3 removed — files absent in many setups)
CORE_DATASETS=(lost MSRCv2 Mirflickr slashdotpl-f1)
LARGE_DATASETS=("Soccer Player" "Yahoo! News")
# Main paper table: core + large (6 datasets)
PRIMARY_DATASETS=(lost MSRCv2 Mirflickr slashdotpl-f1 "Soccer Player" "Yahoo! News")
# Phase 1 sweeps run on all primary sets (long on server; use --batch to run subsets)
TUNE_DATASETS=(lost MSRCv2 Mirflickr slashdotpl-f1 "Soccer Player" "Yahoo! News")
CIFAR10_DATASETS=(
    cifar10-lt-g100-eta0.3-s42 cifar10-lt-g100-eta0.5-s42
    cifar10-lt-g150-eta0.3-s42 cifar10-lt-g150-eta0.5-s42
    cifar10-lt-g200-eta0.3-s42 cifar10-lt-g200-eta0.5-s42
    cifar10-lt-g250-eta0.3-s42 cifar10-lt-g250-eta0.5-s42
)
CIFAR10_FAST_DATASETS=(
    cifar10-lt-g100-eta0.3-fast-s42 cifar10-lt-g100-eta0.5-fast-s42
    cifar10-lt-g200-eta0.3-fast-s42 cifar10-lt-g200-eta0.5-fast-s42
)
CIFAR10_SEED_DATASETS=(
    cifar10-lt-g100-eta0.3-s43 cifar10-lt-g100-eta0.3-s44
    cifar10-lt-g100-eta0.5-s43 cifar10-lt-g100-eta0.5-s44
    cifar10-lt-g200-eta0.3-s43 cifar10-lt-g200-eta0.3-s44
    cifar10-lt-g200-eta0.5-s43 cifar10-lt-g200-eta0.5-s44
)

batch_1a() {
    run_cmd "1a_sr_cb_core" $PYTHON experiments/run.py \
        --datasets "${CORE_DATASETS[@]}" \
        --method sdlpp_sr_cb \
        --sweep disambig.params.use_sample_reliability=false,true \
        --sweep disambig.params.use_class_balance=false,true \
        --n-repeats 5 --campaign ablation_sr_cb_core_v2
}

batch_1b() {
    run_cmd "1b_sr_cb_large" $PYTHON experiments/run.py \
        --datasets "${LARGE_DATASETS[@]}" \
        --method sdlpp_sr_cb \
        --sweep disambig.params.use_sample_reliability=false,true \
        --sweep disambig.params.use_class_balance=false,true \
        --n-repeats 5 --campaign ablation_sr_cb_large_v2
}

batch_2() {
    run_cmd "2_warmup" $PYTHON experiments/run.py \
        --datasets "${TUNE_DATASETS[@]}" \
        --method sdlpp_sr_cb \
        --sweep disambig.params.warmup=0,5,10,20,30,50 \
        --n-repeats 5 --campaign diag_warmup_v1
}

batch_3a() {
    run_cmd "3a_rmin" $PYTHON experiments/run.py \
        --datasets "${TUNE_DATASETS[@]}" \
        --method sdlpp_sr_cb \
        --sweep disambig.params.r_min=0.0,0.05,0.1,0.2,0.3,0.5 \
        --n-repeats 5 --campaign sweep_rmin_v1
}

batch_3b() {
    run_cmd "3b_alpha" $PYTHON experiments/run.py \
        --datasets "${TUNE_DATASETS[@]}" \
        --method sdlpp_sr_cb \
        --sweep disambig.params.alpha=0.1,0.3,0.5,0.7,1.0 \
        --n-repeats 5 --campaign sweep_alpha_v1
}

batch_4a() {
    run_cmd "4a_target_d" $PYTHON experiments/run.py \
        --datasets "${TUNE_DATASETS[@]}" \
        --method sdlpp_sr_cb \
        --sweep model.params.target_d=5,8,13,20,30,50 \
        --n-repeats 5 --campaign sweep_target_d_v1
}

batch_4b() {
    run_cmd "4b_miu" $PYTHON experiments/run.py \
        --datasets "${TUNE_DATASETS[@]}" \
        --method sdlpp_sr_cb \
        --sweep model.params.miu=0.01,0.05,0.1,0.5,1.0 \
        --n-repeats 5 --campaign sweep_miu_v1
}

# SDLPP baseline vs reduced dimension target_d (same grid as 4a; run separately on server)
batch_4c() {
    run_cmd "4c_baseline_target_d" $PYTHON experiments/run.py \
        --datasets "${PRIMARY_DATASETS[@]}" \
        --method sdlpp_baseline \
        --sweep model.params.target_d=5,8,13,20,30,50 \
        --n-repeats 5 --campaign sweep_baseline_target_d_v1
}

batch_5a() {
    run_cmd "5a_bench_core" $PYTHON experiments/run.py \
        --datasets "${CORE_DATASETS[@]}" \
        --methods sdlpp_baseline sdlpp_sr_cb \
        --n-repeats 5 --campaign benchmark_core_sdlpp_v1
}

batch_5b() {
    run_cmd "5b_bench_large" $PYTHON experiments/run.py \
        --datasets "${LARGE_DATASETS[@]}" \
        --methods sdlpp_baseline sdlpp_sr_cb \
        --n-repeats 5 --campaign benchmark_large_sdlpp_v1
}

batch_5c() {
    run_cmd "5c_clf_main" $PYTHON experiments/run.py \
        --datasets "${PRIMARY_DATASETS[@]}" \
        --methods sdlpp_baseline sdlpp_sr_cb \
        --classifiers knn ipal \
        --n-repeats 5 --campaign benchmark_clf_main_v1
}

# SURE (AAAI'19) on tabular PLL sets requested for paper-style comparison
batch_5s() {
    run_cmd "5s_sure_lost_soccer" $PYTHON experiments/run.py \
        --datasets lost "Soccer Player" \
        --methods sdlpp_baseline sdlpp_sr_cb \
        --classifiers sure \
        --n-repeats 10 --campaign benchmark_sure_lost_soccer_v1
}

batch_6a() {
    run_cmd "6a_joint_warmup_rmin" $PYTHON experiments/run.py \
        --datasets MSRCv2 slashdotpl-f1 \
        --method sdlpp_sr_cb \
        --sweep disambig.params.warmup=10,20,30 \
        --sweep disambig.params.r_min=0.0,0.1,0.2 \
        --n-repeats 5 --campaign joint_warmup_rmin_v1
}

batch_6b() {
    run_cmd "6b_adaptive_cb" $PYTHON experiments/run.py \
        --datasets MSRCv2 slashdotpl-f1 \
        --method sdlpp_sr_cb \
        --sweep disambig.params.cb_cv0=0.05,0.1,0.2 \
        --sweep disambig.params.cb_cv1=0.3,0.5,0.7 \
        --n-repeats 5 --campaign adaptive_cb_v1
}

batch_7a() {
    run_cmd "7a_cifar10_ablation" $PYTHON experiments/run.py \
        --datasets "${CIFAR10_DATASETS[@]}" \
        --method sdlpp_sr_cb \
        --sweep disambig.params.use_sample_reliability=false,true \
        --sweep disambig.params.use_class_balance=false,true \
        --n-repeats 5 --campaign cifar10_ablation_sr_cb
}

batch_7b() {
    run_cmd "7b_cifar10_bench" $PYTHON experiments/run.py \
        --datasets "${CIFAR10_DATASETS[@]}" \
        --methods sdlpp_baseline sdlpp_sr_cb \
        --n-repeats 5 --campaign cifar10_benchmark_sdlpp_v1
}

# Requires: python experiments/cifar10_mat_to_resnet18.py (see experiments/REMOTE_EXPERIMENTS.md)
batch_7e() {
    run_cmd "7e_cifar10_resnet18_clf" $PYTHON experiments/run.py \
        --datasets cifar10-lt-g100-eta0.3-s42-resnet18 \
        --methods sdlpp_baseline sdlpp_sr_cb \
        --classifiers knn ipal \
        --n-repeats 5 --campaign cifar10_resnet18_clf_v1
}

batch_7c() {
    run_cmd "7c_cifar10_seeds" $PYTHON experiments/run.py \
        --datasets "${CIFAR10_SEED_DATASETS[@]}" \
        --method sdlpp_sr_cb \
        --n-repeats 5 --campaign cifar10_seed_robustness
}

batch_7s() {
    run_cmd "7s_cifar10_smoke" $PYTHON experiments/run.py \
        --datasets "${CIFAR10_FAST_DATASETS[@]}" \
        --methods sdlpp_baseline sdlpp_sr_cb \
        --fast --campaign cifar_smoke_test
}

# ============================================================================
# Phase orchestration
# ============================================================================

phase1() {
    echo "========== Phase 1: Parameter Tuning (7 batches, parallel) =========="
    batch_1a &
    batch_1b &
    batch_2  &
    batch_3a &
    batch_3b &
    batch_4a &
    batch_4b &
    wait
    echo "========== Phase 1 complete (failures: $FAILED) =========="
}

phase2() {
    echo "========== Phase 2: Benchmark + CIFAR10 (5 batches, parallel) =========="
    batch_5a &
    batch_5b &
    batch_5c &
    batch_7a &
    batch_7b &
    wait
    echo "========== Phase 2 complete (failures: $FAILED) =========="
}

phase3() {
    echo "========== Phase 3: Optional Refinement (3 batches, parallel) =========="
    batch_6a &
    batch_6b &
    batch_7c &
    wait
    echo "========== Phase 3 complete (failures: $FAILED) =========="
}

# ============================================================================
# Dispatch
# ============================================================================

if [[ -n "$BATCH" ]]; then
    case "$BATCH" in
        1a) batch_1a ;;
        1b) batch_1b ;;
        2)  batch_2  ;;
        3a) batch_3a ;;
        3b) batch_3b ;;
        4a) batch_4a ;;
        4b) batch_4b ;;
        4c) batch_4c ;;
        5a) batch_5a ;;
        5b) batch_5b ;;
        5c) batch_5c ;;
        5s) batch_5s ;;
        6a) batch_6a ;;
        6b) batch_6b ;;
        7a) batch_7a ;;
        7b) batch_7b ;;
        7c) batch_7c ;;
        7e) batch_7e ;;
        7s) batch_7s ;;
        *)  echo "Unknown batch: $BATCH"; exit 1 ;;
    esac
elif [[ -n "$PHASE" ]]; then
    case "$PHASE" in
        1)   phase1 ;;
        2)   phase2 ;;
        3)   phase3 ;;
        all)
            phase1
            echo ""
            echo "*** Phase 1 done. Review results in results/ before proceeding. ***"
            echo "*** Update configs/methods/sdlpp_sr_cb.yaml with optimal params if needed. ***"
            echo ""
            phase2
            echo ""
            phase3
            ;;
        *)  echo "Unknown phase: $PHASE"; exit 1 ;;
    esac
fi

if ! $DRY_RUN; then
    echo ""
    echo "============================================"
    echo "All requested experiments finished."
    echo "Total failures: $FAILED"
    [[ -d "$LOGDIR" ]] && echo "Logs: $LOGDIR"
    echo "Results: results/"
    echo "============================================"
fi

exit $FAILED
