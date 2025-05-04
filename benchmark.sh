# python main_benchmark.py \
    # --explainer-model "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" \
    # --autograder-model "gpt-4o" \
    # --run-name "llama4_maverick_explainer_vs_gpt4o_grader"

# python main_benchmark.py \
#     --explainer-model "Qwen/Qwen2.5-72B-Instruct-Turbo" \
#     --autograder-model "gpt-4o" \
#     --run-name "qwen_explainer_vs_gpt4o_grader"

# python main_benchmark.py \
#     --explainer-model "o1" \
#     --autograder-model "gpt-4o" \
#     --run-name "o1_explainer_vs_gpt4o_grader" \
#     --n-workers 30 


# python main_benchmark.py \
#     --explainer-model "claude-3-7-sonnet-latest" \
#     --autograder-model "gpt-4o" \
#     --thinking-budget 4096 \
#     --run-name "claude_thinking_budget_4096"\
#     --n-workers 5

# python main_benchmark.py \
#     --explainer-model "grok-3-beta" \
#     --autograder-model "gpt-4o" \
#     --run-name "grok_3_beta" \
#     --n-workers 5 \

# Run the benchmark with TinyLlama using VLLM endpoint
# python main_benchmark.py \
#     --explainer-model "vllm:TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
#     --autograder-model "gpt-4o" \
#     --run-name "vllm_tinyllama_vs_gpt4o_grader" \
#     --n-workers 1 \
#     --limit 5

# Example for running with Qwen and reasoning effort parameter
python main_benchmark.py \
    --explainer-model "vllm:Qwen/QwQ-32B-AWQ" \
    --autograder-model "gpt-4o" \
    --run-name "vllm_qwen_qwq" \
    --n-workers 1 \
# OPTIONAL if we want to use the reasoning effort parameter
#     --vllm-reasoning-effort "medium" \


