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
#     --explainer-model "grok-4-0709" \
#     --autograder-model "gpt-4o" \
#     --run-name "grok_4_0709" \
#     --n-workers 2 \

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


# python main_benchmark.py \
#     --explainer-model "gemini-2.5-pro-preview-03-25" \
#     --autograder-model "gpt-4o" \
#     --run-name "gemini_2.5_rerun" \
#     --n-workers 10

# # Example command to run benchmark with OpenRouter model
# python main_benchmark.py \
#     --explainer-model "microsoft/phi-4-reasoning:free" \
#     --autograder-model "gpt-4o" \
#     --run-name "phi-4_base" \
#     --n-workers 2 \
#     --limit 10

# Gemini Thinking Budget Experiment


# Gemini Thinking Budget Experiment
# for model in "gemini-2.5-pro-preview-03-25" "gemini-2.5-flash-preview-04-17"; do
#     for budget in 100 200 500 1000 4000; do
#         echo "Running benchmark for $model with budget $budget"
#         # Run benchmark with current model and thinking budget
#         python main_benchmark.py \
#             --explainer-model "$model" \
#             --autograder-model "gpt-4o" \
#             --run-name "gemini_thinking_budget_${budget}_$(echo $model | tr '-' '_')" \
#             --thinking-budget $budget \
#             --n-workers 15
#     done
# done

python main_benchmark.py \
    --explainer-model "claude-opus-4-20250514" \
    --autograder-model "gpt-4o" \
    --run-name "claude-opus-4" \
    --n-workers 20 \