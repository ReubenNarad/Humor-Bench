import os
import sys
import pandas as pd
import json
import re
import matplotlib.pyplot as plt # Re-enable for static plots
# import matplotlib.patches as mpatches # No longer needed for static plots
from datetime import datetime
import numpy as np
# import argparse # No longer needed

# Import adjustText for preventing label overlaps
try:
    from adjustText import adjust_text
except ImportError:
    print("Warning: adjustText library not found. Install with 'pip install adjustText' for better label placement")

# Import the new report generator function
from html_report_generator import create_interactive_report

# --- Configuration ---
# !! MODIFY THIS LIST to include the specific run files you want to compare !!
# These should be the ORIGINAL outputs from main_benchmark.py in the main runs/ dir
RUN_FILES_TO_ANALYZE = [
    # "runs/main/20250410_093249_gpt4o_explainer_vs_gpt4o_grader_exp-gpt_4o_ag-gpt_4o.csv",
    # "runs/main/20250409_234703_o3mini_explainer_vs_gpt4o_grader_exp-o3_mini_ag-gpt_4o.csv",
    "../runs/main/20250409_231147_claude_explainer_vs_gpt4o_grader_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    # "runs/main/20250512_100811_gemini_2.5_rerun_exp-gemini_2_5_pro_preview_03_25_ag-gpt_4o.csv",
    # "runs/main/20250410_143220_gemini_explainer_vs_gpt4o_grader_exp-gemini_1_5_pro_ag-gpt_4o.csv",
    # "runs/main/20250416_102802_llama4_maverick_explainer_vs_gpt4o_grader_exp-meta_llama_Llama_4_Maverick_17B_128E_Instruct_FP8_ag-gpt_4o.csv",
    # "runs/main/20250416_111359_qwen_explainer_vs_gpt4o_grader_exp-Qwen_Qwen2_5_72B_Instruct_Turbo_ag-gpt_4o.csv",
    # "runs/main/20250416_123510_llama4_scout_explainer_vs_gpt4o_grader_exp-meta_llama_Llama_4_Scout_17B_16E_Instruct_ag-gpt_4o.csv",
    # "runs/main/20250416_152132_o4-mini_explainer_vs_gpt4o_grader_exp-o4_mini_ag-gpt_4o.csv",
    # "runs/main/20250416_182853_o3_explainer_vs_gpt4o_grader_exp-o3_ag-gpt_4o.csv",
    # "runs/main/20250416_184110_o1_explainer_vs_gpt4o_grader_exp-o1_ag-gpt_4o.csv",
    # "runs/main/20250428_145241_deepseek_v3_exp-deepseek_ai_DeepSeek_V3_ag-gpt_4o.csv",
    # "runs/main/20250428_151728_deepseek_r1_exp-deepseek_ai_DeepSeek_R1_ag-gpt_4o.csv",
    # "runs/main/20250502_222737_grok_3_beta_exp-grok_3_beta_ag-gpt_4o.csv",

    # Claude thinking budget experiment
    "../runs/claude_thinking_experiment/20250416_193729_claude_thinking_budget_1024_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    "../runs/claude_thinking_experiment/20250416_200436_claude_thinking_budget_2048_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    "../runs/claude_thinking_experiment/20250416_204832_claude_thinking_budget_4096_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    "../runs/claude_thinking_experiment/20250506_171410_claude_thinking_budget_8192_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    "../runs/claude_thinking_experiment/20250506_174903_claude_thinking_budget_16384_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",

    # OpenAI reasoning effort experiment
    # "runs/openai_effort_experiment/20250506_180913_o3_high_exp-o3_ag-gpt_4o.csv",
    # "runs/openai_effort_experiment/20250506_185500_o3_low_exp-o3_ag-gpt_4o.csv",
    # "runs/openai_effort_experiment/20250506_190750_o4-mini_low_exp-o4_mini_ag-gpt_4o.csv",
    # "runs/openai_effort_experiment/20250506_191011_o4-mini_medium_exp-o4_mini_ag-gpt_4o.csv",
    # "runs/openai_effort_experiment/20250506_191022_o4-mini_high_exp-o4_mini_ag-gpt_4o.csv",
    # "runs/openai_effort_experiment/20250506_191209_o3_medium_exp-o3_ag-gpt_4o.csv"

    # Qwen reasoning experiment
    # "runs/qwen_thinking_experiment/20250511_151057_qwen_plus_reasoning_50_exp-qwen_plus_2025_04_28_ag-gpt_4o.csv",
    # "runs/qwen_thinking_experiment/20250511_155221_qwen_plus_reasoning_1000_exp-qwen_plus_2025_04_28_ag-gpt_4o.csv",
    # "runs/qwen_thinking_experiment/20250511_164036_qwen_plus_reasoning_2000_exp-qwen_plus_2025_04_28_ag-gpt_4o.csv",
    # "runs/qwen_thinking_experiment/20250511_173236_qwen_plus_reasoning_200_exp-qwen_plus_2025_04_28_ag-gpt_4o.csv",
    # "runs/qwen_thinking_experiment/20250511_180928_qwen_plus_reasoning_400_exp-qwen_plus_2025_04_28_ag-gpt_4o.csv",
    # "runs/qwen_thinking_experiment/20250511_184731_qwen_plus_reasoning_600_exp-qwen_plus_2025_04_28_ag-gpt_4o.csv"
]

# Path to categorization files
CATEGORY_FILES = {
    "wordplay": "../categorized_output_wordplay.csv",
    "cultural_reference": "../categorized_output_cultural_reference.csv",
    "toxic_or_shocking": "../categorized_output_toxic_or_shocking.csv"
}

# Number of hard questions to select (kept for backward compatibility)
HARD_SUBSET_SIZE = 100

# Map extracted model names (after replacing _) to desired nicknames for the plot
MODEL_NICKNAMES = {
    "gpt-4o": "gpt-4o",
    "o4-mini": "o4-mini", # Base nickname for o4-mini
    "o3-mini": "o3-mini",
    "claude-3-7-sonnet-latest": "Claude 3.7 Sonnet",  # Updated default Claude nickname
    "gemini-2-5-pro-preview-03-25": "Gemini 2.5 pro",
    "gemini-1-5-pro": "Gemini 1.5 pro",
    "meta-llama-Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama 4 Maverick",
    "Qwen-Qwen2-5-72B-Instruct-Turbo": "Qwen 2.5 72B",
    "qwen-plus-2025-04-28": "Qwen Plus 04-28", # Added for Qwen thinking experiment
    "meta-llama-Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout",
    "o1": "o1", # Base nickname for o1
    "o3": "o3", # Base nickname for o3
    "deepseek-ai-DeepSeek-V3": "DeepSeek V3", # Add nickname for DeepSeek V3
    "deepseek-ai-DeepSeek-R1": "DeepSeek R1", # Add nickname for DeepSeek R1
    "grok-3-beta": "Grok 3", # Add nickname for grok-3-beta
    # Add specialized nicknames for Claude with thinkin  budgets
    # These will be generated dynamically below
}

MODEL_PRICES_PATH = "../model_data/model_prices.json"
BASE_ANALYSIS_DIR = "../analysis" # Base directory for all analysis output
RUNS_SUBDIR = "runs" # Subdirectory within BASE_ANALYSIS_DIR for individual runs

# Remove VIEW_MODE global variable
# VIEW_MODE = "per_row" # Default value (REMOVED)

# --- Update FAMILY_COLORS to rename families ---
FAMILY_COLORS = {
    'anthropic': '#e3c771',
    'openai': '#89e095',
    'google': '#8055e6',  # Renamed from gemini
    'open_source': 'lightblue',  # Consolidated category for open source models
    'XAI': '#333333',  # Changed to all caps
    'unknown': 'grey'  # Fallback color
}

# --- Update the infer_family_from_model function ---
def infer_family_from_model(model):
    """
    Infer the model family from the model name string.
    """
    model_lower = model.lower()
    # Check for specific prefixes first for Together API models
    # Updated list based on common providers and model_prices.json structure
    if 'qwen' in model_lower: # Specific check for Qwen models first
        return 'open_source'
    if any(prefix in model_lower for prefix in ['meta-llama/', 'mistralai/', 'deepseek-ai/', 'togethercomputer/']): # Removed 'qwen/'
        return 'open_source'
    # General keywords for Together (if prefix doesn't match)
    elif any(name in model_lower for name in ['llama', 'mistral', 'mixtral', 'falcon', 'dbrx']): # Removed 'qwen2'
        return 'open_source'
    # Then check other families based on common naming conventions and model_prices.json
    elif any(name in model_lower for name in ['gpt', 'o1', 'o3', 'o4']): # Added o3, o4
        return 'openai'
    elif 'claude' in model_lower:
        return 'anthropic' # Use 'anthropic' to match user request color key
    elif 'gemini' in model_lower:
        return 'google'  # Changed from 'gemini' to 'google'
    elif 'grok' in model_lower: # Added check for grok models
        return 'XAI'  # Changed to all caps
    elif 'deepseek' in model_lower: # Check specific deepseek if not covered by together prefixes
         return 'open_source' # Change to open_source
    else:
        # Fallback or raise error - Defaulting to 'unknown' for plotting
        print(f"Warning: Could not definitively infer family for {model}. Assigning 'unknown'.")
        return 'unknown'

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

# --- Add a function to calculate explainer-only cost ---
def calculate_explainer_cost(row, model_prices):
    """Calculate the cost for just the explainer model (no autograder cost)."""
    explainer_model = row.get('explainer_model')
    explainer_usage_str = row.get('explainer_usage', '{}')
    
    # Default cost to 0
    cost = 0.0
    
    # Explainer cost only
    if explainer_model in model_prices:
        try:
            usage = json.loads(explainer_usage_str) if isinstance(explainer_usage_str, str) else explainer_usage_str
            tokens_in = usage.get('tokens_in', 0)
            tokens_out = usage.get('tokens_out', 0)
            prices = model_prices[explainer_model]
            cost += (tokens_in / 1_000_000) * prices.get("input $/M", 0)
            cost += (tokens_out / 1_000_000) * prices.get("output $/M", 0)
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
            pass  # Fail silently for summary
    
    return cost

# --- Main Analysis Logic ---
# Remove view_mode parameter from function definition
def generate_benchmark_data(run_files, prices_path, paper_plots=False):
    """Analyzes multiple benchmark runs and generates data for interactive report."""
    run_summaries = []
    autograder_models_found = set()
    
    # Dictionary to track passing rate per caption across all models
    caption_pass_counts = {}
    caption_appearance_counts = {}
    all_data_frames = []  # Store all valid dataframes for hard subset determination

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
                 
            # Store original filename in the dataframe for easier identification 
            valid_df = valid_df.copy()
            if 'source_file' not in valid_df.columns:
                valid_df['source_file'] = filename  # Add filename to help with identification later
                
            # Store valid dataframe for hard subset determination with source information
            all_data_frames.append(valid_df)

            # Update caption pass counts for determining hard questions
            for caption, group in valid_df.groupby('caption'):
                passes = sum(group['autograder_judgment'] == 'PASS')
                if caption not in caption_pass_counts:
                    caption_pass_counts[caption] = 0
                    caption_appearance_counts[caption] = 0
                caption_pass_counts[caption] += passes
                caption_appearance_counts[caption] += len(group)
                
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
            explainer_cost = 0  # Add this line
            avg_cost_per_row = 0 
            if model_prices:
                cost_col_name = 'temp_cost'
                # Calculate cost on the original full dataframe (df) as errors might still cost
                df[cost_col_name] = df.apply(lambda row: calculate_cost(row, model_prices), axis=1)
                total_cost = df[cost_col_name].sum()
                
                # Calculate explainer-only cost (add this)
                df['explainer_cost'] = df.apply(lambda row: calculate_explainer_cost(row, model_prices), axis=1)
                explainer_cost = df['explainer_cost'].sum()
                
                # Use len(df) for avg cost calculation as errors still incur cost potentially
                avg_cost_per_row = total_cost / len(df) if len(df) > 0 else 0
                # df = df.drop(columns=[cost_col_name]) # Optional cleanup
            print(f"  Metrics - Cost: Avg/Row=${avg_cost_per_row:.6f}, Total=${total_cost:.4f}, Explainer Only=${explainer_cost:.4f}")

            # --- Calculate mean explainer output tokens ---
            mean_output_tokens = 0
            explainer_output_tokens = []
            if 'explainer_usage' in df.columns:
                for usage_str in df['explainer_usage']:
                    try:
                        if pd.notna(usage_str):
                            usage_json = json.loads(str(usage_str))
                            tokens_out = usage_json.get('tokens_out')
                            if tokens_out is not None:
                                explainer_output_tokens.append(int(tokens_out))
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
                
                if explainer_output_tokens:
                    mean_output_tokens = sum(explainer_output_tokens) / len(explainer_output_tokens)
                    print(f"  Metrics - Mean Output Tokens: {mean_output_tokens:.2f}")
                else:
                    print("  Warning: No valid explainer output token data found.")
            else:
                print("  Warning: 'explainer_usage' column not found in the run data.")

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
                'explainer_cost': explainer_cost,  # Add this line to include the explainer-only cost
                'avg_cost_per_row': avg_cost_per_row,
                'mean_output_tokens': mean_output_tokens,  # Add mean token count to the summary
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
    
    # --- Find Hard Question Subset ---
    try:
        # --- Load category data and create subsets ---
        if all_data_frames:
            # First, combine all dataframes to create category subsets later
            combined_df = pd.concat(all_data_frames)

            # Dictionary to hold all category datasets
            category_data = {}
            category_elements = {}
            
            # Load each category file
            for category_name, file_path in CATEGORY_FILES.items():
                try:
                    category_df = pd.read_csv(file_path)
                    print(f"\nLoaded {len(category_df)} rows from {file_path}")
                    
                    # Determine the actual column name from the CSV (may contain spaces)
                    # First get all columns that aren't 'idx', 'prompt_tokens', or 'completion_tokens'
                    category_columns = [col for col in category_df.columns 
                                       if col not in ['idx', 'prompt_tokens', 'completion_tokens']]
                    
                    if not category_columns:
                        print(f"Warning: Could not find category column in {file_path}")
                        continue
                        
                    # Use the first remaining column as the category column name
                    actual_column_name = category_columns[0]
                    print(f"Using column '{actual_column_name}' for category '{category_name}'")
                    
                    # Extract TRUE rows for this category using the actual column name
                    true_items = category_df[category_df[actual_column_name] == True]
                    true_idx = set(true_items['idx'].values)
                    
                    print(f"Found {len(true_idx)} items with TRUE for category '{category_name}'")
                    if len(true_idx) == 0:
                        continue
                        
                    # Store category data for later use
                    category_data[category_name] = true_idx
                    
                    # Create subset of unique caption-element pairs for this category
                    category_rows = combined_df[combined_df['idx'].isin(true_idx)]
                    if not category_rows.empty:
                        # Store unique caption-element pairs
                        category_elements[category_name] = set(
                            (row['caption'], row['element']) 
                            for _, row in category_rows.iterrows()
                        )
                        
                        # Save category subset for reference
                        category_subset_dir = os.path.join(BASE_ANALYSIS_DIR, "category_subsets")
                        os.makedirs(category_subset_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        category_subset_path = os.path.join(
                            category_subset_dir, 
                            f"{timestamp}_{category_name}_subset.csv"
                        )
                        category_rows.to_csv(category_subset_path, index=False)
                        print(f"Saved {category_name} subset ({len(category_elements[category_name])} elements) to: {category_subset_path}")
                    else:
                        print(f"Warning: No rows found for category '{category_name}' in the benchmark data")
                        
                except FileNotFoundError:
                    print(f"Warning: Category file {file_path} not found. Skipping {category_name} category.")
                except Exception as e:
                    print(f"Error processing category '{category_name}': {e}")
            
            print(f"\nProcessed {len(category_data)} category files successfully")
        else:
            print("\nNo valid dataframes found. Cannot create category subsets.")
            category_elements = {}
        
    except Exception as e:
        print(f"\nError calculating category subsets: {e}")
        category_elements = {}  # Safety fallback
    
    # Now calculate metrics for each model on each category subset
    for i, summary in enumerate(run_summaries):
        model_name = summary['explainer_model']
        model_file = summary['source_file']
        
        # Find matching dataframe from all_data_frames
        file_basename = os.path.basename(model_file)
        matching_df = None
        
        for df in all_data_frames:
            # First check if there's a source_file column we can use
            if 'source_file' in df.columns:
                if file_basename in df['source_file'].iloc[0]:
                    matching_df = df
                    break
            # If not, check if the explainer model matches
            elif 'explainer_model' in df.columns:
                if df['explainer_model'].iloc[0] == model_name:
                    matching_df = df
                    break
        
        if matching_df is not None:
            # Calculate pass rate metrics for each category subset
            for category_name, elements in category_elements.items():
                # Filter model data for category elements
                category_df = matching_df[matching_df.apply(
                    lambda row: (row['caption'], row['element']) in elements, 
                    axis=1
                )]
                
                # Calculate per-row pass rate on category subset
                if not category_df.empty:
                    category_passes = sum(category_df['autograder_judgment'] == 'PASS')
                    category_total = len(category_df)
                    category_pass_rate = category_passes / category_total if category_total > 0 else 0
                    
                    # Add this category's metric to the summary
                    metric_name = f"{category_name}_pass_rate_per_row"
                    summary_df.at[i, metric_name] = category_pass_rate
                    
                    print(f"  Model {model_name}: {category_name} pass rate = {category_pass_rate:.2%} ({category_passes}/{category_total})")

    # --- Generate Display Name based on model type and filename ---
    def generate_display_names(row):
        """Generate plain text display name and HTML plot label."""
        model_name = row['explainer_model']
        filename = row['source_file']
        base_nickname = MODEL_NICKNAMES.get(model_name, model_name)

        plain_name = base_nickname
        html_label = base_nickname # Default to plain name

        # 1. Check for OpenAI 'o' models with effort
        if model_name in ['o1', 'o4-mini', 'o3', 'o3-mini']:
            effort_match_low = re.search(rf"{model_name}_low", filename)
            effort_match_medium = re.search(rf"{model_name}_medium", filename)
            effort_match_high = re.search(rf"{model_name}_high", filename)

            if effort_match_low:
                plain_name = f"{base_nickname} low"
            elif effort_match_medium:
                plain_name = f"{base_nickname} medium"
            elif effort_match_high:
                plain_name = f"{base_nickname} high"
            html_label = plain_name # HTML label is the same here

        # 2. Check for Claude models with thinking budget
        elif "claude" in model_name.lower():
            # For thinking budget runs
            if "thinking_budget" in filename:
                budget_match = re.search(r'thinking_budget_(\d+k?)', filename)
                if budget_match:
                    budget_str = budget_match.group(1)
                    # Simplified display format - just "Budget: {value}"
                    plain_name = f"Budget: {budget_str}"
                    html_label = f"Budget: {budget_str}"
            # For base Claude model (no thinking budget specified)
            else:
                plain_name = f"Budget: 0"
                html_label = f"Budget: 0"

        # 3. Check for Qwen Plus 2025-04-28 models with reasoning value
        elif model_name == "qwen-plus-2025-04-28" and "reasoning" in filename:
            reasoning_match = re.search(r'reasoning_(\d+)', filename)
            if reasoning_match:
                reasoning_val = reasoning_match.group(1)
                # base_nickname here is "Qwen Plus 04-28" from MODEL_NICKNAMES
                plain_name = f"{base_nickname} Reasoning {reasoning_val}"
                html_label = f"{base_nickname}<br><span style='color:grey;'>Reasoning: {reasoning_val}</span>"

        return pd.Series([plain_name, html_label], index=['display_name', 'plot_label'])

    # Apply the new display name generation function
    summary_df[['display_name', 'plot_label']] = summary_df.apply(generate_display_names, axis=1)

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
    run_output_dir = os.path.join(BASE_ANALYSIS_DIR, RUNS_SUBDIR, "claude_experiment")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Created analysis run directory: {run_output_dir}")

    # --- Save Processed Data as JSON ---
    # 1. First copy only the columns that definitely exist
    plot_data_df = summary_df[[
        'display_name', 'explainer_model', 'autograder_model',
        'pass_rate_per_row', 'pass_rate_per_caption_all', 'pass_rate_per_caption_some',
        'total_cost', 'explainer_cost', 'avg_cost_per_row',
        'plot_color', 'explainer_family',
        'num_rows', 'num_captions'
    ]]
    
    # Add mean_output_tokens field if it exists
    if 'mean_output_tokens' in summary_df.columns:
        plot_data_df['mean_output_tokens'] = summary_df['mean_output_tokens']
        print(f"Added mean output token data for {plot_data_df['mean_output_tokens'].notna().sum()} of {len(plot_data_df)} models")
    
    # Add plot_label column if it exists in summary_df
    if 'plot_label' in summary_df.columns:
        plot_data_df['plot_label'] = summary_df['plot_label']
    else: # Ensure the column exists even if empty (e.g., if no special names were generated)
        plot_data_df['plot_label'] = plot_data_df['display_name'] # Default to display_name

    # Add category-specific pass rates if they exist
    for category_name in CATEGORY_FILES.keys():
        metric_name = f"{category_name}_pass_rate_per_row"
        if metric_name in summary_df.columns:
            plot_data_df[metric_name] = summary_df[metric_name]

    # --- Load GPQA scores first ---
    # Load GPQA scores from external file
    gpqa_file_path = "../model_data/models_gpqa.json"
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
    arc_agi_file_path = "../model_data/models_ARC_AGI.json"
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
    lmarena_file_path = "../model_data/lmarena_elo_4_22_25.json"
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

    # --- Generate reports based on options ---
    if paper_plots:
        print("\nGenerating paper-ready plots only...")
        # Generate Static Scatter Plots for Academic Paper
        generate_static_scatter_plots(run_output_dir, data_json_path)
    else:
        # Generate Interactive HTML Report
        create_interactive_report(run_output_dir, data_json_path, report_autograder)
        
        # Generate Static Scatter Plots for Academic Paper
        generate_static_scatter_plots(run_output_dir, data_json_path)
        
        # Generate Static Bar Chart
        try:
            print("Generating static bar chart...")
            plt.figure(figsize=(14, 8)) # Adjusted size for potentially many labels

            # Sort data for the bar chart
            chart_df = summary_df.copy()

            def extract_experiment_sort_key(display_name_series):
                display_name = str(display_name_series) # Ensure it's a string
                # Try Claude thinking budget first (e.g., "Claude 3.7 Sonnet 1024")
                claude_match = re.search(r'Claude.*?Sonnet\s+(\d+k?)', display_name)
                if claude_match:
                    val_str = claude_match.group(1)
                    if 'k' in val_str.lower(): 
                        return int(val_str.lower().replace('k', '')) 
                    return int(val_str)

                # Try Qwen reasoning value (e.g., "Qwen Plus 04-28 Reasoning 50")
                qwen_match = re.search(r"Qwen Plus 04-28 Reasoning (\d+)", display_name) # Made Qwen match more specific
                if qwen_match:
                    return int(qwen_match.group(1))
                
                # Check for base models to assign a sort key of 0
                if display_name == MODEL_NICKNAMES.get("claude-3-7-sonnet-latest"): # Base Claude Sonnet
                    return 0
                if display_name == MODEL_NICKNAMES.get("qwen-plus-2025-04-28"):      # Base Qwen Plus 04-28
                    return 0
                
                return None # For other models without a specific numeric experiment value or known base name

            chart_df['experiment_sort_key'] = chart_df['display_name'].apply(extract_experiment_sort_key)
            
            # Separate models with and without experiment keys for clearer sorting
            models_with_key = chart_df[chart_df['experiment_sort_key'].notna()].copy()
            models_without_key = chart_df[chart_df['experiment_sort_key'].isna()].copy()
            
            # Ensure experiment_sort_key is numeric for proper sorting in models_with_key
            if not models_with_key.empty:
                models_with_key['experiment_sort_key'] = pd.to_numeric(models_with_key['experiment_sort_key'])
                
            models_with_key.sort_values(by=['experiment_sort_key', 'display_name'], ascending=[True, True], inplace=True)
            models_without_key.sort_values(by=['display_name'], ascending=[True], inplace=True)
            
            chart_df = pd.concat([models_with_key, models_without_key], ignore_index=True)
            
            # Ensure 'pass_rate_per_row' is numeric
            chart_df['pass_rate_per_row'] = pd.to_numeric(chart_df['pass_rate_per_row'], errors='coerce')
            chart_df.dropna(subset=['pass_rate_per_row'], inplace=True)

            # Extract just the thinking budget value for x-axis labels
            def extract_budget_label(display_name):
                display_name = str(display_name)
                # Try Claude thinking budget first (e.g., "Claude 3.7 Sonnet 1024")
                claude_match = re.search(r'Claude.*?Sonnet\s+(\d+k?)', display_name)
                if claude_match:
                    return claude_match.group(1)  # Return the budget value
                
                # Check for base models to assign "0"
                if display_name == MODEL_NICKNAMES.get("claude-3-7-sonnet-latest"):
                    return "0"
                    
                return display_name  # Fallback to display name for other models

            # Create budget labels for x-axis
            chart_df['budget_label'] = chart_df['display_name'].apply(extract_budget_label)

            bars = plt.bar(chart_df['budget_label'], chart_df['pass_rate_per_row'] * 100, color=chart_df['plot_color'])

            plt.xlabel("Thinking budget", fontsize=22)
            plt.ylabel("HumorBench Score (%)", fontsize=22)
            # plt.title(f"Claude 3.7 thinking budget experiment", fontsize=24)
            plt.xticks(rotation=45, ha="right", fontsize=16)
            
            # Rescale Y-axis
            pass_rates_percent = chart_df['pass_rate_per_row'].dropna() * 100
            if not pass_rates_percent.empty:
                min_rate = pass_rates_percent.min()
                max_rate = pass_rates_percent.max()
                
                y_axis_min = min_rate * 0.95
                y_axis_max = max_rate * 1.05
                
                y_axis_min = max(0, y_axis_min) # Ensure min is not less than 0
                y_axis_max = min(105, y_axis_max) # Ensure max is not excessively over 100
                
                # If min and max are very close, add some padding
                if abs(y_axis_max - y_axis_min) < 5: # e.g. less than 5 percentage points range
                    y_axis_min = max(0, y_axis_min - 2.5)
                    y_axis_max = min(105, y_axis_max + 2.5)

                plt.ylim(y_axis_min, y_axis_max)
                
                # Adjust y-ticks based on new scale
                # Generate around 10 ticks within this new range
                tick_values = np.linspace(y_axis_min, y_axis_max, num=11)
                plt.yticks(tick_values, fontsize=16)
                
                # Format y-ticks as percentages
                ax = plt.gca()
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
            else:
                plt.yticks(np.arange(0, 101, 10), fontsize=16) # Fallback if no data

            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on top of bars
            for bar in bars:
                yval = bar.get_height()
                if pd.notnull(yval):
                     plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=15)

            plt.tight_layout() # Adjust layout to make room for labels

            bar_chart_path = os.path.join(run_output_dir, "pass_rate_bar_chart.png")
            # plt.savefig(bar_chart_path)
            # print(f"Saved static bar chart to: {bar_chart_path}")
            # plt.close() # Close the plot to free memory
        except Exception as e:
            print(f"Error generating or saving static bar chart: {e}")

    # --- Generate Histogram of Explainer Output Tokens if only one run file --- (NEW SECTION)
    if len(run_files) == 1 and all_data_frames:
        try:
            print("\\nGenerating histogram of explainer output tokens for the single run...")
            single_run_df = all_data_frames[0] # Get the DataFrame for the single processed run
            explainer_output_tokens = []

            if 'explainer_usage' in single_run_df.columns:
                for usage_str in single_run_df['explainer_usage']:
                    try:
                        if pd.notna(usage_str):
                            usage_json = json.loads(str(usage_str))
                            tokens_out = usage_json.get('tokens_out')
                            if tokens_out is not None:
                                explainer_output_tokens.append(int(tokens_out))
                        # else: handle cases where usage_str might be NaN/None if necessary
                    except json.JSONDecodeError:
                        # Potentially log this or just skip if some rows have bad JSON
                        pass # Or print(f"Warning: Could not parse explainer_usage JSON: {usage_str}")
                    except (TypeError, AttributeError):
                        # Handle cases where usage_json is not a dict or tokens_out is not found as expected
                        pass
            else:
                print("Warning: 'explainer_usage' column not found in the run data. Cannot generate token histogram.")

            if explainer_output_tokens:
                plt.figure(figsize=(10, 6))
                plt.hist(explainer_output_tokens, bins='auto', color='mediumpurple', rwidth=0.85)
                plt.xlabel("Explainer Output Tokens per Row", fontsize=22)
                plt.ylabel("Frequency (Number of Rows)", fontsize=22)
                run_filename = os.path.basename(run_files[0]) # Get the original filename for the title
                # plt.title(f"Distribution of Explainer Output Tokens\\nRun: {run_filename}", fontsize=24)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()

                histogram_path = os.path.join(run_output_dir, "explainer_output_tokens_histogram.png")
                plt.savefig(histogram_path)
                print(f"Saved explainer output tokens histogram to: {histogram_path}")
                plt.close()
            elif 'explainer_usage' in single_run_df.columns: # Only print if column was there but no data extracted
                print("Warning: No valid explainer output token data found to generate histogram.")

            # --- Generate Histograms for each Category (if single run) --- (NEW SUB-SECTION)
            if category_elements: # Check if category_elements were loaded
                print("\\nGenerating explainer output token histograms for each category...")
                for category_name, elements_in_category in category_elements.items():
                    category_explainer_output_tokens = []
                    
                    # Filter the single_run_df for the current category
                    # Ensure 'caption' and 'element' columns are present in single_run_df
                    if not all(col in single_run_df.columns for col in ['caption', 'element']):
                        print(f"Warning: 'caption' or 'element' column missing. Cannot filter for category '{category_name}'.")
                        continue

                    category_specific_df = single_run_df[single_run_df.apply(
                        lambda row: (row.get('caption'), row.get('element')) in elements_in_category, 
                        axis=1
                    )]

                    if not category_specific_df.empty and 'explainer_usage' in category_specific_df.columns:
                        for usage_str in category_specific_df['explainer_usage']:
                            try:
                                if pd.notna(usage_str):
                                    usage_json = json.loads(str(usage_str))
                                    tokens_out = usage_json.get('tokens_out')
                                    if tokens_out is not None:
                                        category_explainer_output_tokens.append(int(tokens_out))
                            except (json.JSONDecodeError, TypeError, AttributeError):
                                pass # Silently skip rows with bad data for category histograms
                        
                        if category_explainer_output_tokens:
                            plt.figure(figsize=(10, 6))
                            plt.hist(category_explainer_output_tokens, bins='auto', color='mediumseagreen', rwidth=0.85)
                            plt.xlabel(f"Explainer Output Tokens per Row ({category_name})", fontsize=22)
                            plt.ylabel("Frequency (Number of Rows)", fontsize=22)
                            run_filename = os.path.basename(run_files[0])
                            # plt.title(f"Distribution of Explainer Output Tokens - Category: {category_name}\\nRun: {run_filename}", fontsize=24)
                            plt.grid(axis='y', linestyle='--', alpha=0.7)
                            plt.tight_layout()

                            cat_histogram_path = os.path.join(run_output_dir, f"explainer_output_tokens_histogram_{category_name}.png")
                            plt.savefig(cat_histogram_path)
                            print(f"Saved {category_name} token histogram to: {cat_histogram_path}")
                            plt.close()
                        else:
                            print(f"Warning: No valid explainer output token data found for category '{category_name}' to generate histogram.")
                    elif category_specific_df.empty:
                        print(f"Info: No data rows found for category '{category_name}' in this run.")
            else:
                print("Info: No category data loaded, skipping category-specific histograms.")

        except Exception as e:
            print(f"Error generating or saving explainer output tokens histogram: {e}")

    # --- Add after analyze_benchmark_results function, before if __name__ == "__main__" ---

def generate_static_scatter_plots(run_output_dir, data_json_path):
    """Generates high-quality static scatter plots for academic papers with accessibility features."""
    print("\nGenerating static scatter plots for academic papers...")
    
    # Load data from JSON
    try:
        with open(data_json_path, 'r') as f:
            plot_data = json.load(f)
        if not plot_data:
            print("Error: No data found in JSON file.")
            return
    except Exception as e:
        print(f"Error loading data from {data_json_path}: {e}")
        return
    
    # Define unique shapes for each family for colorblind accessibility
    # Matplotlib marker styles: 'o' (circle), 's' (square), '^' (triangle up), 
    # 'D' (diamond), 'v' (triangle down), '>' (triangle right), '*' (star)
    family_markers = {
        'anthropic': '^',     # Circle
        'openai': 'o',        # Square
        'google': 's',        # Triangle up (renamed from gemini)
        'open_source': 'D',   # Diamond (consolidated)
        'XAI': 'X',           # Thick X mark (capital X for thicker marker)
        'unknown': 'P'        # Plus
    }
    
    # --- Adding custom sizes for markers ---
    family_marker_sizes = {
        'anthropic': 100,     
        'openai': 100,        
        'google': 100,        
        'open_source': 100,   
        'XAI': 100,
        'unknown': 100        
    }
    
    # --- Adding custom linewidths for markers ---
    family_marker_linewidths = {
        'anthropic': 1,     
        'openai': 1,        
        'google': 1,        
        'open_source': 1,   
        'XAI': 1,
        'unknown': 1        
    }
    
    # Placeholder offsets for mean response tokens plot (Claude thinking budget experiment)
    mean_token_offsets = {
        "Budget: 0": (15, -10),  # Placeholder - adjust as needed
        "Budget: 1024": (-150, 0), 
        "Budget: 2048": (-150, 0),
        "Budget: 4096": (-150, 0),
        "Budget: 8192": (-150, 0),
        "Budget: 16384": (-150, -30)
    }
    
    # Map x_axis field to the appropriate offset dictionary
    axis_to_offsets = {
        "mean_output_tokens": mean_token_offsets,
        # Add other axes if needed
    }
    
    # Set up figure style for academic paper with bigger text/labels
    plt.rcParams.update({
        'font.size': 16,             
        'axes.labelsize': 22,        # Increased from 18 to 22
        'axes.titlesize': 24,        # Increased from 20 to 24
        'xtick.labelsize': 16,       # Increased from 14 to 16
        'ytick.labelsize': 16,       # Increased from 14 to 16
        'legend.fontsize': 14,       
        'legend.title_fontsize': 16, 
        'lines.linewidth': 2.5,      
        'lines.markersize': 12,      
        'lines.markeredgewidth': 2,  
        'figure.dpi': 300,
    })
    
    # Recategorize data to use consolidated family
    for item in plot_data:
        if item['explainer_family'] in ['qwen', 'together', 'deepseek']:
            item['explainer_family'] = 'open_source'
            item['plot_color'] = FAMILY_COLORS['open_source']
    
    # Create a DataFrame for easier filtering and plotting
    df = pd.DataFrame(plot_data)
    
    # Define X-axis options to generate multiple plots
    x_axis_options = [
        # {"field": "explainer_cost", "label": "Run Cost ($)", "log_scale": False},
        {"field": "mean_output_tokens", "label": "Mean Response Tokens", "log_scale": False},
        # {"field": "gpqa_score", "label": "GPQA Score (%)", "log_scale": False},
        # {"field": "arc_agi_score", "label": "ARC-AGI Score (%)", "log_scale": False},
        # {"field": "lmarena_elo_score", "label": "LM Arena ELO Score", "log_scale": False}
    ]
    
    # Only keep x-axis options that have data
    x_axis_options = [opt for opt in x_axis_options 
                     if opt["field"] in df.columns and df[opt["field"]].notna().any()]
    
    # Define Y-axis options (we'll use 'pass_rate_per_row' by default)
    y_axis_options = [
        {"field": "pass_rate_per_row", "label": "HumorBench Score (%)"},
    ]
    
    # Only keep y-axis options that have data
    y_axis_options = [opt for opt in y_axis_options 
                     if opt["field"] in df.columns and df[opt["field"]].notna().any()]
    
    # If no data is available for specific categories, use the main pass rate
    if len(y_axis_options) <= 1:
        y_axis_options = [{"field": "pass_rate_per_row", "label": "HumorBench Score (%)"}]
    
    # Generate plots for each X-axis option using the primary Y-axis (toxic/shocking)
    for y_option in y_axis_options:
        for x_option in x_axis_options:
            try:
                # Filter data for this x-y combination (remove NaN values)
                plot_df = df.dropna(subset=[x_option["field"], y_option["field"]])
                
                if plot_df.empty:
                    print(f"No data available for {x_option['label']} vs {y_option['label']}. Skipping.")
                    continue
                
                # Create figure with appropriate size for academic paper (larger for readability)
                plt.figure(figsize=(10, 8))
                
                # Lists to store annotations for adjustText
                texts = []
                
                # Group by family to apply different markers
                for family, group in plot_df.groupby('explainer_family'):
                    # Default values if family is not in our dictionaries
                    marker = family_markers.get(family, 'o')
                    color = FAMILY_COLORS.get(family, 'gray')
                    marker_size = family_marker_sizes.get(family, 180)
                    line_width = family_marker_linewidths.get(family, 2)
                    
                    # Plot points with both color and shape coding
                    plt.scatter(
                        group[x_option["field"]] * (100 if x_option["field"] in ["gpqa_score", "arc_agi_score"] else 1),
                        group[y_option["field"]] * 100,  # Convert to percentage
                        marker=marker,
                        s=marker_size,  # Use family-specific size
                        color=color,
                        edgecolors='black',
                        linewidths=line_width,  # Use family-specific linewidth
                        alpha=0.8,
                        label=family.replace('_', ' ').title()  # Format family name for legend
                    )
                
                    # Add model names as annotations with larger font
                    for _, row in group.iterrows():
                        x_val = row[x_option["field"]] * (100 if x_option["field"] in ["gpqa_score", "arc_agi_score"] else 1)
                        y_val = row[y_option["field"]] * 100
                        model_name = row['display_name']
                        
                        # Default offset if not specified for this model in the current plot type
                        offset_x, offset_y = 7, 7
                        
                        # Get the current offset dictionary based on x-axis field
                        current_offsets = axis_to_offsets.get(x_option["field"], {})
                        
                        # Apply model-specific offset if available for this plot type
                        if model_name in current_offsets:
                            offset_x, offset_y = current_offsets[model_name]
                            
                        # Apply the offsets
                        plt.annotate(
                            model_name,
                            xy=(x_val, y_val),
                            xytext=(offset_x, offset_y),
                            textcoords='offset points',
                            fontsize=20,  # Increased from 12 to 16
                        )
                
                # Add dotted line connecting Claude thinking budget points in mean_output_tokens plot
                if x_option["field"] == "mean_output_tokens":
                    # Create a separate dataframe for Claude models
                    claude_df = plot_df[plot_df['explainer_family'] == 'anthropic'].copy()
                    
                    # Sort dataframes by thinking budget size
                    # Extract budget value and add as column for sorting
                    def get_budget_size(name):
                        budget_match = re.search(r'Budget: (\d+k?)', name)
                        if budget_match:
                            budget_str = budget_match.group(1)
                            if 'k' in budget_str:
                                return int(budget_str.replace('k', '000'))
                            return int(budget_str)
                        return 0
                    
                    claude_df['budget'] = claude_df['display_name'].apply(get_budget_size)
                    claude_df = claude_df.sort_values('budget')
                    
                    # Draw the dotted line if we have at least 2 points
                    if len(claude_df) >= 2:
                        plt.plot(
                            claude_df[x_option["field"]],
                            claude_df[y_option["field"]] * 100,
                            '--',  # Dashed line
                            color=FAMILY_COLORS['anthropic'],  # Use anthropic color (beige)
                            linewidth=2,
                            alpha=0.85  # Make it faint
                        )
                        print(f"Added dotted line connecting Claude thinking budget points")
                
                # Set axis labels with larger font
                plt.xlabel(x_option["label"], fontsize=22)
                plt.ylabel(y_option["label"], fontsize=22)
                
                # Set title
                plot_title = f"Claude 3.7 Sonnet, Thinking Budget"
                if x_option["field"] == "explainer_cost":
                    plot_title = f"HumorBench"
                # plt.title(plot_title, fontsize=24)
                
                # Set x-axis limits only for specific plots, not for mean tokens
                if x_option["field"] == "explainer_cost":
                    plt.xlim(3.5, 6)
                elif x_option["field"] == "mean_output_tokens":
                    # Set x-axis limit to 1000 as requested
                    plt.xlim(160, 600)
                    plt.ylim(80,84)
                
                # Apply log scale with more tick marks if specified
                if x_option["log_scale"]:
                    plt.xscale('log')
                    plt.xlabel(f"{x_option['label']} (Log Scale)", fontsize=22)
                    
                    # Add more tick marks for log scale
                    if x_option["field"] == "explainer_cost":
                        # Define exact positions for log scale ticks
                        major_ticks = [0.5, 1, 2, 5, 10, 20]
                        plt.xticks(major_ticks)
                        
                        # Format tick labels
                        ax = plt.gca()
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}' if x < 1 else f'${int(x)}'))
                        
                        # Add minor ticks
                        import matplotlib.ticker as ticker
                        ax.xaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(0.1, 1.0, 0.1), numticks=20))
                
                # Add grid for better readability
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add legend showing both color and shape for each family
                # legend = plt.legend(title="Model Family", title_fontsize=16, 
                #                   loc='lower right', frameon=True, framealpha=0.9)
                # legend.get_frame().set_linewidth(2)
                
                # Adjust layout
                plt.tight_layout()
                
                # Create descriptive filename
                y_field = y_option["field"].replace("_", "-")
                x_field = x_option["field"].replace("_", "-")
                log_suffix = "_log" if x_option["log_scale"] else ""
                
                filename = f"claude_reasoning_exp.png"
                filepath = os.path.join(run_output_dir, filename)
                
                # Save with high DPI for print quality
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved {y_option['label']} vs {x_option['label']} plot to: {filepath}")
                
            except Exception as e:
                print(f"Error generating plot for {x_option['label']} vs {y_option['label']}: {e}")
    
    # Generate a single plot with a legend showing all families with their colors and markers
    try:
        plt.figure(figsize=(8, 6))
        
        # Plot dummy points for each family just for the legend
        for family in FAMILY_COLORS.keys():
            if family != 'unknown':  # Skip unknown in the legend
                marker = family_markers.get(family, 'o')
                color = FAMILY_COLORS.get(family, 'gray')
                marker_size = family_marker_sizes.get(family, 180)
                line_width = family_marker_linewidths.get(family, 2)
                
                plt.scatter(
                    [], [], 
                    marker=marker, 
                    s=marker_size, 
                    color=color, 
                    edgecolors='black', 
                    linewidths=line_width, 
                    label=family if family == 'XAI' else family.replace('_', ' ').title()
                )
        
        plt.axis('off')  # Hide axes
        # Create legend
        plt.legend(title="Model Families", title_fontsize=16, fontsize=14,
                loc='center', frameon=True, ncol=2)
        plt.tight_layout()
        
        # Save legend as separate file
        # legend_path = os.path.join(run_output_dir, "paper_plot_legend.png")
        # plt.savefig(legend_path, dpi=300, bbox_inches='tight')
        # plt.close()
        
        print(f"Saved family legend to: {legend_path}")
        
    except Exception as e:
        print(f"Error generating legend: {e}")
    
    print("\nStatic scatter plots generation complete.")


if __name__ == "__main__":
    # Remove argument parsing
    # No longer need to set VIEW_MODE

    # Add argument parsing for paper_plots option
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark results")
    parser.add_argument("--paper-plots", action="store_true", 
                       help="Generate only paper-ready scatter plots without interactive HTML")
    args = parser.parse_args()

    # Run the data generation and report creation
    generate_benchmark_data(RUN_FILES_TO_ANALYZE, MODEL_PRICES_PATH, paper_plots=args.paper_plots)
