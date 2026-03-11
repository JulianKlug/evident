#!/bin/bash
# Run benchmarks for all models sequentially, saving results separately
set -e
cd /home/klug/evident

MODELS=("mistral-small3.2:24b" "qwen3:30b-a3b" "gemma3:27b" "deepseek-r1:32b" "qwen3:14b")

for model in "${MODELS[@]}"; do
    safe_name=$(echo "$model" | tr ':.' '_')
    echo "=== Starting benchmark: $model ==="
    python run_benchmark.py --model "$model" --few-shot-only --normalize 2>&1 | tee "docs/benchmark_log_${safe_name}.txt"

    # Copy results with model-specific name
    if [ -f docs/benchmark_results.csv ]; then
        cp docs/benchmark_results.csv "docs/benchmark_results_${safe_name}.csv"
        echo "=== Saved results to docs/benchmark_results_${safe_name}.csv ==="
    fi
done

echo "=== All benchmarks complete ==="
