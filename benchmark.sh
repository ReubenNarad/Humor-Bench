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
#     --thinking-budget 16384 \
#     --run-name "claude_thinking_budget_16384"\
#     --n-workers 5

# python main_benchmark.py \
#     --explainer-model "grok-3-beta" \
#     --autograder-model "gpt-4o" \
#     --run-name "grok_3_beta" \
#     --n-workers 5 \

# python main_benchmark.py \
#     --explainer-model "o4-mini" \
#     --autograder-model "gpt-4o" \
#     --run-name "o4-mini_medium" \
#     --n-workers 10 \
#     --reasoning-effort="medium"

# python main_benchmark.py \
#     --explainer-model "qwen-plus-2025-04-28" \
#     --autograder-model "gpt-4o" \
#     --run-name "qwen_plus_reasoning_50" \
#     --thinking-budget 600 \
#     --n-workers 3


python main_benchmark.py \
    --explainer-model "gemini-2.5-pro-preview-03-25" \
    --autograder-model "gpt-4o" \
    --run-name "gemini_2.5_rerun" \
    --n-workers 10
