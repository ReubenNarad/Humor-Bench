import os
import sys
import pandas as pd
import json
import re
# import matplotlib.pyplot as plt # No longer needed for static plots
# import matplotlib.patches as mpatches # No longer needed for static plots
from datetime import datetime
import numpy as np
# import argparse # No longer needed

# Import the new report generator function
from html_report_generator import create_interactive_report

# --- Configuration ---
# !! MODIFY THIS LIST to include the specific run files you want to compare !!
# These should be the ORIGINAL outputs from main_benchmark.py in the main runs/ dir
RUN_FILES_TO_ANALYZE = [
    "runs/main/20250410_093249_gpt4o_explainer_vs_gpt4o_grader_exp-gpt_4o_ag-gpt_4o.csv",
    "runs/main/20250409_234703_o3mini_explainer_vs_gpt4o_grader_exp-o3_mini_ag-gpt_4o.csv",
    "runs/main/20250409_231147_claude_explainer_vs_gpt4o_grader_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    "runs/main/20250409_220748_gemini_explainer_vs_gpt4o_grader_exp-gemini_2_5_pro_preview_03_25_ag-gpt_4o.csv",
    "runs/main/20250410_143220_gemini_explainer_vs_gpt4o_grader_exp-gemini_1_5_pro_ag-gpt_4o.csv",
    "runs/main/20250416_102802_llama4_maverick_explainer_vs_gpt4o_grader_exp-meta_llama_Llama_4_Maverick_17B_128E_Instruct_FP8_ag-gpt_4o.csv",
    "runs/main/20250416_111359_qwen_explainer_vs_gpt4o_grader_exp-Qwen_Qwen2_5_72B_Instruct_Turbo_ag-gpt_4o.csv",
    "runs/main/20250416_123510_llama4_scout_explainer_vs_gpt4o_grader_exp-meta_llama_Llama_4_Scout_17B_16E_Instruct_ag-gpt_4o.csv",
    "runs/main/20250416_152132_o4-mini_explainer_vs_gpt4o_grader_exp-o4_mini_ag-gpt_4o.csv",
    "runs/main/20250416_182853_o3_explainer_vs_gpt4o_grader_exp-o3_ag-gpt_4o.csv",
    "runs/main/20250416_184110_o1_explainer_vs_gpt4o_grader_exp-o1_ag-gpt_4o.csv",

    # Claude thinking budget experiment
    "runs/claude_thinking_experiment/20250416_193729_claude_thinking_budget_1024_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    # "runs/claude_thinking_experiment/20250416_200436_claude_thinking_budget_2048_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    "runs/claude_thinking_experiment/20250416_204832_claude_thinking_budget_4096_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",

    # OpenAI reasoning effort experiment
    # "runs/openai_effort_experiment/20250417_102543_o4-mini_low_exp-o4_mini_ag-gpt_4o.csv",
    # "runs/openai_effort_experiment/20250417_102740_o4-mini_medium_exp-o4_mini_ag-gpt_4o.csv",
    # "runs/openai_effort_experiment/20250417_103013_o4-mini_high_exp-o4_mini_ag-gpt_4o.csv",
]

# Map extracted model names (after replacing _) to desired nicknames for the plot
MODEL_NICKNAMES = {
    "gpt-4o": "gpt-4o",
    "o4-mini": "o4-mini", # Base nickname for o4-mini
    "o3-mini": "o3-mini",
    "claude-3-7-sonnet-latest": "Claude 3.7",  # Default Claude without thinking
    "gemini-2-5-pro-preview-03-25": "Gemini 2.5 pro",
    "gemini-1-5-pro": "Gemini 1.5 pro",
    "meta-llama-Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama 4 Maverick",
    "Qwen-Qwen2-5-72B-Instruct-Turbo": "Qwen 2.5 72B",
    "meta-llama-Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout",
    "o1": "o1", # Base nickname for o1
    "o3": "o3", # Base nickname for o3
    # Add specialized nicknames for Claude with thinkin  budgets
    # These will be generated dynamically below
}

MODEL_PRICES_PATH = "model_data/model_prices.json"
BASE_ANALYSIS_DIR = "analysis" # Base directory for all analysis output
RUNS_SUBDIR = "runs" # Subdirectory within BASE_ANALYSIS_DIR for individual runs

# Remove VIEW_MODE global variable
# VIEW_MODE = "per_row" # Default value (REMOVED)

# --- Add Model Family Inference Logic ---
# Simplified inference logic based on explainer.py, using string literals
def infer_family_from_model(model):
    """
    Infer the model family from the model name string.
    """
    model_lower = model.lower()
    # Check for specific prefixes first for Together API models
    # Updated list based on common providers and model_prices.json structure
    if any(prefix in model_lower for prefix in ['meta-llama/', 'mistralai/', 'qwen/', 'deepseek-ai/', 'togethercomputer/']):
        return 'together'
    # General keywords for Together (if prefix doesn't match)
    elif any(name in model_lower for name in ['llama', 'mistral', 'mixtral', 'falcon', 'qwen2', 'dbrx']): # Added qwen2
        return 'together'
    # Then check other families based on common naming conventions and model_prices.json
    elif any(name in model_lower for name in ['gpt', 'o1', 'o3', 'o4']): # Added o3, o4
        return 'openai'
    elif 'claude' in model_lower:
        return 'anthropic' # Use 'anthropic' to match user request color key
    elif 'gemini' in model_lower:
        return 'gemini'
    elif 'deepseek' in model_lower: # Check specific deepseek if not covered by together prefixes
         return 'deepseek' # Or map to 'together' if always used via their API
    else:
        # Fallback or raise error - Defaulting to 'unknown' for plotting
        print(f"Warning: Could not definitively infer family for {model}. Assigning 'unknown'.")
        return 'unknown'

# Define colors based on user request (using string family names)
FAMILY_COLORS = {
    'anthropic': '#e3c771',
    'openai': '#89e095',
    'gemini': '#8055e6', # Using mediumpurple as lightpurple can be very pale
    'together': 'lightblue',  # Using lightblue as blue can be dark
    # 'deepseek': 'orange',    # Adding a color for deepseek if inferred separately
    'unknown': 'grey'       # Fallback color
}

# --- Helper Function (calculate_cost) ---
def calculate_cost(row, model_prices):
    """Calculate the cost for a single row based on usage and prices."""
    cost = 0.0
    explainer_model = row.get('explainer_model')
    explainer_usage_str = row.get('explainer_usage', '{}')
    autograder_model = row.get('autograder_model')
    autograder_usage_str = row.get('autograder_usage', '{}')

    # Explainer cost
    if explainer_model in model_prices:
        try:
            usage = json.loads(explainer_usage_str)
            tokens_in = usage.get('tokens_in', 0)
            tokens_out = usage.get('tokens_out', 0)
            prices = model_prices[explainer_model]
            cost += (tokens_in / 1_000_000) * prices.get("input $/M", 0)
            cost += (tokens_out / 1_000_000) * prices.get("output $/M", 0)
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
            pass # Fail silently for summary

    # Autograder cost
    if autograder_model in model_prices:
         try:
            usage = json.loads(autograder_usage_str)
            tokens_in = usage.get('tokens_in', 0)
            tokens_out = usage.get('tokens_out', 0)
            prices = model_prices[autograder_model]
            cost += (tokens_in / 1_000_000) * prices.get("input $/M", 0)
            cost += (tokens_out / 1_000_000) * prices.get("output $/M", 0)
         except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
            pass # Fail silently for summary

    return cost


# --- Main Analysis Logic ---
# Remove view_mode parameter from function definition
def generate_benchmark_data(run_files, prices_path):
    """Analyzes multiple benchmark runs and generates data for interactive report."""
    run_summaries = []
    autograder_models_found = set()

    # Load prices
    model_prices = {}
    if os.path.exists(prices_path):
        try:
            with open(prices_path, 'r') as f:
                model_prices = json.load(f)
            print(f"Loaded model prices from {prices_path}")
        except Exception as e:
            print(f"Warning: Failed to load or parse {prices_path}: {e}. Costs will not be calculated.")
    else:
        print(f"Warning: {prices_path} not found. Costs will not be calculated.")


    # print(f"\nAnalyzing benchmark runs (View Mode: {view_mode})...") # Remove view mode from log
    print(f"\nAnalyzing benchmark runs...")
    print("-" * 30)

    for file_path in run_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}...")
        try:
            df = pd.read_csv(file_path)

            # Verify required columns - ensure 'caption' is always checked
            required_cols = ['explainer_model', 'autograder_model', 'autograder_judgment', 'explainer_usage', 'autograder_usage', 'caption']

            if not all(col in df.columns for col in required_cols):
                print(f"  Skipping: Missing required columns. Needed: {', '.join(required_cols)}")
                continue

            # Filter out rows where judgment is not PASS or FAIL before calculating rates
            valid_df = df[df['autograder_judgment'].isin(['PASS', 'FAIL'])].copy()
            if valid_df.empty:
                 print(f"  Skipping: No valid 'PASS' or 'FAIL' judgments found.")
                 continue

            # --- Calculate ALL PASS Rate Metrics ---
            num_rows = len(valid_df)
            passes_row = sum(valid_df['autograder_judgment'] == 'PASS')
            pass_rate_per_row = passes_row / num_rows if num_rows > 0 else 0
            print(f"  Metrics - Per Row: Items={num_rows}, Pass Rate={pass_rate_per_row:.2%}")

            # Per Caption (All)
            caption_groups = valid_df.groupby('caption')
            num_captions = len(caption_groups)
            perfect_captions = 0
            for _, group in caption_groups:
                if all(group['autograder_judgment'] == 'PASS'):
                    perfect_captions += 1
            pass_rate_per_caption_all = perfect_captions / num_captions if num_captions > 0 else 0
            print(f"  Metrics - Per Caption (All): Captions={num_captions}, Perfect={perfect_captions}, Pass Rate={pass_rate_per_caption_all:.2%}")

            # Per Caption (Some)
            total_partial_score = 0.0
            for _, group in caption_groups:
                 partial_score = (group['autograder_judgment'] == 'PASS').mean()
                 total_partial_score += partial_score
            pass_rate_per_caption_some = total_partial_score / num_captions if num_captions > 0 else 0
            print(f"  Metrics - Per Caption (Some): Captions={num_captions}, Total Score={total_partial_score:.2f}, Pass Rate={pass_rate_per_caption_some:.2%}")

            # --- Extract Metadata ---
            # Extract Explainer Model from filename
            match = re.search(r'exp-(.*?)_ag', filename)
            if not match:
                print(f"  Skipping: Could not extract explainer model name from filename.")
                continue
            explainer_model_from_filename = match.group(1).replace('_', '-')

            # Get autograder model used in this run
            autograder_model = df['autograder_model'].iloc[0] # Use original df for metadata
            autograder_models_found.add(autograder_model)

            # --- Calculate Cost ---
            total_cost = 0
            avg_cost_per_row = 0 # Keep this per-row for consistency
            if model_prices:
                cost_col_name = 'temp_cost'
                # Calculate cost on the original full dataframe (df) as errors might still cost
                df[cost_col_name] = df.apply(lambda row: calculate_cost(row, model_prices), axis=1)
                total_cost = df[cost_col_name].sum()
                # Use len(df) for avg cost calculation as errors still incur cost potentially
                avg_cost_per_row = total_cost / len(df) if len(df) > 0 else 0
                # df = df.drop(columns=[cost_col_name]) # Optional cleanup
            print(f"  Metrics - Cost: Avg/Row=${avg_cost_per_row:.6f}, Total=${total_cost:.4f}")

            # --- Append all data to summary ---
            run_summaries.append({
                'explainer_model': explainer_model_from_filename,
                'autograder_model': autograder_model,
                'num_rows': num_rows,
                'num_captions': num_captions,
                'pass_rate_per_row': pass_rate_per_row,
                'pass_rate_per_caption_all': pass_rate_per_caption_all,
                'pass_rate_per_caption_some': pass_rate_per_caption_some,
                'total_cost': total_cost,
                'avg_cost_per_row': avg_cost_per_row,
                'source_file': filename
                # Add other metrics like GPQA score here if available later
                # 'gpqa_score': df['gpqa_score'].mean() # Example
            })

        except FileNotFoundError:
            print(f"  Skipping: File not found.")
        except Exception as e:
            print(f"  Skipping: Error processing file - {e}")

    # --- Create and process summary DataFrame ---
    if not run_summaries:
        print("\nNo valid runs found or processed.")
        return

    summary_df = pd.DataFrame(run_summaries)
    # Sort by a default metric initially, e.g., per_row pass rate
    summary_df = summary_df.sort_values(by='pass_rate_per_row', ascending=True)

    # --- Generate Display Name based on model type and filename ---
    def generate_display_name(row):
        """Generate display name, checking for Claude budget or OpenAI effort"""
        model_name = row['explainer_model'] # Normalized name like 'o4-mini' or 'claude-3-7-sonnet-latest'
        filename = row['source_file']
        base_nickname = MODEL_NICKNAMES.get(model_name, model_name) # Get default nickname first

        # 1. Check for OpenAI 'o' models with effort in filename (using run_name convention)
        # Assumes run_name format like 'o4-mini_low', 'o4-mini_medium', etc.
        if model_name in ['o1', 'o4-mini', 'o3', 'o3-mini']: # Add other 'o' models if needed
            effort_match_low = re.search(rf"{model_name}_low", filename)
            effort_match_medium = re.search(rf"{model_name}_medium", filename)
            effort_match_high = re.search(rf"{model_name}_high", filename)

            if effort_match_low:
                return f"{base_nickname} low"
            elif effort_match_medium:
                # Optionally, don't append 'medium' as it's the default
                return f"{base_nickname} medium"
                # return base_nickname # Keep it clean if medium is default behavior - Previous logic
            elif effort_match_high:
                return f"{base_nickname} high"

        # 2. Check for Claude models with thinking budget in filename
        if "claude" in model_name.lower() and "thinking_budget" in filename:
            budget_match = re.search(r'thinking_budget_(\d+k?)', filename)
            if budget_match:
                budget_str = budget_match.group(1)
                # Use base_nickname here too, e.g., "Claude 3.7 (think: 4096)"
                return f"{base_nickname} (think: {budget_str})"

        # 3. Default case: Use the standard nickname mapping
        return base_nickname

    # Apply the new display name generation function
    summary_df['display_name'] = summary_df.apply(generate_display_name, axis=1)

    # --- Infer Family and Assign Colors ---
    summary_df['explainer_family'] = summary_df['explainer_model'].apply(infer_family_from_model)
    summary_df['plot_color'] = summary_df['explainer_family'].map(FAMILY_COLORS).fillna(FAMILY_COLORS['unknown'])

    # --- Determine Autograder and Set Up Output Directory ---
    report_autograder = "unknown" # Default if mixed or error
    safe_report_autograder = "unknown"
    if len(autograder_models_found) == 1:
        report_autograder = list(autograder_models_found)[0]
        safe_report_autograder = report_autograder.replace('/', '_').replace('-', '_').replace('.', '_')
        print(f"\n--- Benchmark Summary (Autograder: {report_autograder}) ---")
    elif len(autograder_models_found) > 1:
         report_autograder = "multiple"
         safe_report_autograder = "multiple"
         print("\n--- Benchmark Summary (Multiple Autograders Found) ---")
         print("Warning: Comparing runs with different autograders. Data generated, but use caution interpreting results.")
    else:
        print("\n--- Benchmark Summary (Could not determine Autograder) ---")


    # --- Create Timestamped Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # New directory structure: analysis/runs/TIMESTAMP_AUTOGRADER/
    run_output_dir = os.path.join(BASE_ANALYSIS_DIR, RUNS_SUBDIR, f"{timestamp}_{safe_report_autograder}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Created analysis run directory: {run_output_dir}")

    # --- Save Processed Data as JSON ---
    # 1. First copy only the columns that definitely exist
    plot_data_df = summary_df[[
        'display_name', 'explainer_model', 'autograder_model',
        'pass_rate_per_row', 'pass_rate_per_caption_all', 'pass_rate_per_caption_some',
        'total_cost', 'avg_cost_per_row',
        'plot_color', 'explainer_family',
        'num_rows', 'num_captions'
    ]]

    # --- Load GPQA scores first ---
    # Load GPQA scores from external file
    gpqa_file_path = "model_data/models_gpqa.json"
    try:
        with open(gpqa_file_path, 'r') as f:
            gpqa_data = json.load(f)
        
        # Add GPQA scores to the dataframe
        for i, row in plot_data_df.iterrows():
            model_name = row['explainer_model']
            # Try to find the model in the GPQA data
            if model_name in gpqa_data:
                plot_data_df.at[i, 'gpqa_score'] = gpqa_data[model_name].get('score', None)
            else:
                # Try with a more flexible matching approach
                for gpqa_model in gpqa_data:
                    # Convert both to lowercase and remove common separators for comparison
                    clean_gpqa_model = gpqa_model.lower().replace('-', '').replace('_', '').replace(' ', '')
                    clean_model_name = model_name.lower().replace('-', '').replace('_', '').replace(' ', '')
                    if clean_model_name in clean_gpqa_model or clean_gpqa_model in clean_model_name:
                        plot_data_df.at[i, 'gpqa_score'] = gpqa_data[gpqa_model].get('score', None)
                        break
        
        # Check if any GPQA scores were added
        has_gpqa = plot_data_df['gpqa_score'].notna().sum() > 0
        if has_gpqa:
            print(f"Added GPQA scores for {plot_data_df['gpqa_score'].notna().sum()} of {len(plot_data_df)} models")
        else:
            print("Warning: No GPQA scores could be matched to the models in the analysis")
        
    except Exception as e:
        print(f"Warning: Could not load GPQA scores from {gpqa_file_path}: {e}")
        print("GPQA scores will not be available in the report")

    # --- Load ARC-AGI scores ---
    arc_agi_file_path = "model_data/models_ARC_AGI.json"
    try:
        with open(arc_agi_file_path, 'r') as f:
            arc_agi_data = json.load(f)
        
        # Add ARC-AGI scores to the dataframe
        for i, row in plot_data_df.iterrows():
            model_name = row['explainer_model']
            # Try to find the model in the ARC-AGI data
            if model_name in arc_agi_data:
                plot_data_df.at[i, 'arc_agi_score'] = arc_agi_data[model_name].get('score', None)
            else:
                # Try with a more flexible matching approach
                for arc_model in arc_agi_data:
                    # Convert both to lowercase and remove common separators for comparison
                    clean_arc_model = arc_model.lower().replace('-', '').replace('_', '').replace(' ', '')
                    clean_model_name = model_name.lower().replace('-', '').replace('_', '').replace(' ', '')
                    if clean_model_name in clean_arc_model or clean_arc_model in clean_model_name:
                        plot_data_df.at[i, 'arc_agi_score'] = arc_agi_data[arc_model].get('score', None)
                        break
        
        # Check if any ARC-AGI scores were added
        has_arc = plot_data_df['arc_agi_score'].notna().sum() > 0
        if has_arc:
            print(f"Added ARC-AGI scores for {plot_data_df['arc_agi_score'].notna().sum()} of {len(plot_data_df)} models")
        else:
            print("Warning: No ARC-AGI scores could be matched to the models in the analysis")
        
    except Exception as e:
        print(f"Warning: Could not load ARC-AGI scores from {arc_agi_file_path}: {e}")
        print("ARC-AGI scores will not be available in the report")

    # --- Load LM Arena ELO scores ---
    lmarena_file_path = "model_data/lmarena_elo_4_22_25.json"
    try:
        with open(lmarena_file_path, 'r') as f:
            lmarena_data = json.load(f)
        
        # Add LM Arena ELO scores to the dataframe
        for i, row in plot_data_df.iterrows():
            model_name = row['explainer_model']
            # Try to find the model in the LM Arena data
            if model_name in lmarena_data:
                plot_data_df.at[i, 'lmarena_elo_score'] = lmarena_data[model_name].get('score', None)
            else:
                # Try with a more flexible matching approach
                for arena_model in lmarena_data:
                    # Convert both to lowercase and remove common separators for comparison
                    clean_arena_model = arena_model.lower().replace('-', '').replace('_', '').replace(' ', '')
                    clean_model_name = model_name.lower().replace('-', '').replace('_', '').replace(' ', '')
                    if clean_model_name in clean_arena_model or clean_arena_model in clean_model_name:
                        plot_data_df.at[i, 'lmarena_elo_score'] = lmarena_data[arena_model].get('score', None)
                        break
        
        # Check if any LM Arena ELO scores were added
        has_lmarena = plot_data_df['lmarena_elo_score'].notna().sum() > 0
        if has_lmarena:
            print(f"Added LM Arena ELO scores for {plot_data_df['lmarena_elo_score'].notna().sum()} of {len(plot_data_df)} models")
        else:
            print("Warning: No LM Arena ELO scores could be matched to the models in the analysis")
        
    except Exception as e:
        print(f"Warning: Could not load LM Arena ELO scores from {lmarena_file_path}: {e}")
        print("LM Arena ELO scores will not be available in the report")

    # --- THEN convert to list and save ---
    # Convert DataFrame to list of dictionaries for JSON
    plot_data_list = plot_data_df.to_dict(orient='records')

    data_json_path = os.path.join(run_output_dir, "data.json")

    try:
        with open(data_json_path, 'w') as f:
            json.dump(plot_data_list, f, indent=4)
        print(f"Saved plot data to: {data_json_path}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")
        return # Cannot proceed without data

    # --- Generate Interactive HTML Report ---
    # Call the function from the other file
    create_interactive_report(run_output_dir, data_json_path, report_autograder)

    # --- Remove Static Plot Generation Code ---
    # (All plt / matplotlib code related to bar and scatter plots is removed)

    # --- Remove Per-Question Difficulty Analysis ---
    # (This section is removed as it's less suitable for the interactive overview)
    # (Consider making it a separate script if still needed)


if __name__ == "__main__":
    # Remove argument parsing
    # No longer need to set VIEW_MODE

    # Run the data generation and report creation
    generate_benchmark_data(RUN_FILES_TO_ANALYZE, MODEL_PRICES_PATH)
