# LLM Agent Context for TSP Mech Interp Project

## Overall Goal

The primary objective of this project is to perform mechanistic interpretability (mech interp) on a neural network trained to solve the Traveling Salesperson Problem (TSP). This involves understanding the internal representations and computations of the network. Key techniques currently employed include:

*   Training Sparse Autoencoders (SAEs) on the network's activations.
*   Analyzing the features learned by these SAEs.
*   Experimenting with probes to test specific hypotheses about network function.

## Project Structure

The project is organized as follows:

*   **Core Python Scripts:**
    *   `main_benchmark.py`: Seems to be the main entry point for running benchmark experiments, likely involving the TSP NN, SAEs, or probes, potentially using explainer and autograder models.
    *   `benchmark.py`: Likely contains benchmarking logic called by `main_benchmark.py`.
    *   `autograder.py`: Contains logic for automatic grading or evaluation, potentially of explanations or model outputs.
    *   `autograder_eval.py`: Script focused on evaluating the autograder's performance or running evaluations using the autograder.
    *   `explainer.py`: Likely contains code related to generating explanations for the model's behavior, possibly using external LLMs.
    *   `generate_explanations.py`: Script specifically for generating explanations.
    *   `utils.py`: Utility functions used across the project.
    *   `prompts.py`: Stores prompts, likely used for interacting with LLMs (explainers/graders).
    *   *(Other scripts like `regrade.py`, `annotate.py`, `rubric_annotator.py` seem related to data annotation and evaluation refinement).*
*   **Shell Scripts:**
    *   `benchmark.sh`: Used to launch benchmark experiments by running `main_benchmark.py` with different configurations (models, parameters). Contains commented-out examples and an active command.
    *   `run_eval.sh`: Runs `run_eval.py`, likely for evaluating experiment results.
    *   `autograder_eval_analysis.sh`: Presumably runs `autograder_eval_analysis.py` for analyzing evaluation results.
    *   `generate_mixed_explanations.sh`: Likely orchestrates the generation of explanations using `generate_explanations.py` or similar scripts.
*   **Data/Configuration:**
    *   `model_prices.json`: Contains pricing information for different models.
    *   `.env`: Environment variables (ensure sensitive keys are handled securely).
    *   `comprehensive_annotations.csv`: Seems to contain annotated data, possibly for training or evaluation.
*   **Directories:**
    *   `runs/`: Stores results and artifacts from benchmark runs initiated via `main_benchmark.py`.
    *   `autograder_runs/`: Likely stores results specific to autograder evaluations.
    *   `analysis/`: Intended for storing analysis results and scripts.
    *   `rubric/`: Contains files related to evaluation rubrics.
    *   `legacy/`: Contains older or deprecated code.
    *   `__pycache__/`: Python bytecode cache (usually ignored).

## Running Experiments

1.  **Main Benchmarks:** Modify and run `benchmark.sh` or execute `python main_benchmark.py` directly with appropriate command-line arguments (e.g., `--explainer-model`, `--autograder-model`, `--run-name`). Results are typically saved in the `runs/` directory, organized by `run-name`.
2.  **Evaluation:** Run `run_eval.sh` (which executes `run_eval.py`) to perform evaluations, possibly on the results generated in step 1.
3.  **Analysis:** Execute analysis scripts like `autograder_eval_analysis.py` (potentially via `autograder_eval_analysis.sh`) to process results from `runs/` or `autograder_runs/`.
4.  **Explanation Generation:** Use `generate_mixed_explanations.sh` or run `generate_explanations.py` directly as needed.

*(Note: This is inferred from script names and contents. The exact flow might require examining the Python scripts in more detail).*

## Rules for LLM Agent Interaction

1.  **Report Changes:** Any modifications made to the codebase or significant findings/actions taken should be documented by updating **this file (`llms.md`)**. Add a new section or update relevant existing sections.
