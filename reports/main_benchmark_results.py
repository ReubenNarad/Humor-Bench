import os
import sys
import pandas as pd
import json
import re
import matplotlib.pyplot as plt # Re-enable for static plots
# import matplotlib.patches as mpatches # No longer needed for static plots
from datetime import datetime
import numpy as np
from scipy.stats import spearmanr  # Add import for Spearman correlation
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
    "../runs/main/20250410_093249_gpt4o_explainer_vs_gpt4o_grader_exp-gpt_4o_ag-gpt_4o.csv",
    # "../runs/main/20250409_234703_o3mini_explainer_vs_gpt4o_grader_exp-o3_mini_ag-gpt_4o.csv",
    "../runs/main/20250409_231147_claude_explainer_vs_gpt4o_grader_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    "../runs/20250522_173648_claude-4-sonnet_exp-claude_sonnet_4_20250514_ag-gpt_4o.csv",
    "../runs/main/20250710_195633_claude-opus-4_exp-claude_opus_4_20250514_ag-gpt_4o.csv",
    "../runs/main/20250512_100811_gemini_2.5_rerun_exp-gemini_2_5_pro_preview_03_25_ag-gpt_4o.csv",
    # "../runs/main/20250410_143220_gemini_explainer_vs_gpt4o_grader_exp-gemini_1_5_pro_ag-gpt_4o.csv",
    "../runs/main/20250416_102802_llama4_maverick_explainer_vs_gpt4o_grader_exp-meta_llama_Llama_4_Maverick_17B_128E_Instruct_FP8_ag-gpt_4o.csv",
    "../runs/main/20250416_111359_qwen_explainer_vs_gpt4o_grader_exp-Qwen_Qwen2_5_72B_Instruct_Turbo_ag-gpt_4o.csv",
    # "../runs/main/20250416_123510_llama4_scout_explainer_vs_gpt4o_grader_exp-meta_llama_Llama_4_Scout_17B_16E_Instruct_ag-gpt_4o.csv",
    "../runs/main/20250416_152132_o4-mini_explainer_vs_gpt4o_grader_exp-o4_mini_ag-gpt_4o.csv",
    "../runs/main/20250416_182853_o3_explainer_vs_gpt4o_grader_exp-o3_ag-gpt_4o.csv",
    "../runs/main/20250416_184110_o1_explainer_vs_gpt4o_grader_exp-o1_ag-gpt_4o.csv",
    # "../runs/main/20250428_145241_deepseek_v3_exp-deepseek_ai_DeepSeek_V3_ag-gpt_4o.csv",
    "../runs/main/20250428_151728_deepseek_r1_exp-deepseek_ai_DeepSeek_R1_ag-gpt_4o.csv",
    "../runs/main/20250502_222737_grok_3_beta_exp-grok_3_beta_ag-gpt_4o.csv",
    "../runs/main/20250710_095959_grok_4_0709_exp-grok_4_0709_ag-gpt_4o.csv",
    # "../runs/main/20250513_114914_openrouter_deepseek_r1_zero_exp-deepseek_deepseek_r1_zero:free_ag-gpt_4o.csv"
    

    # Claude thinking budget experiment
    # "runs/claude_thinking_experiment/20250416_193729_claude_thinking_budget_1024_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    # "runs/claude_thinking_experiment/20250416_200436_claude_thinking_budget_2048_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    # "runs/claude_thinking_experiment/20250416_204832_claude_thinking_budget_4096_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    # "runs/claude_thinking_experiment/20250506_171410_claude_thinking_budget_8192_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    # "runs/claude_thinking_experiment/20250506_174903_claude_thinking_budget_16384_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",

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
    "../runs/qwen_thinking_experiment/20250511_173236_qwen_plus_reasoning_200_exp-qwen_plus_2025_04_28_ag-gpt_4o.csv",
    # "runs/qwen_thinking_experiment/20250511_180928_qwen_plus_reasoning_400_exp-qwen_plus_2025_04_28_ag-gpt_4o.csv",
    # "runs/qwen_thinking_experiment/20250511_184731_qwen_plus_reasoning_600_exp-qwen_plus_2025_04_28_ag-gpt_4o.csv"
    "../runs/vllm/20250508_034834_vllm_phi4plus_exp-vllm_microsoft_Phi_4_reasoning_plus_ag-gpt_4o.csv",

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
    "claude-3-7-sonnet-latest": "Claude 3.7 Sonnet", 
    "claude-sonnet-4-20250514": "Claude Sonnet 4", 
    "claude-opus-4-20250514": "Claude Opus 4", 
    "gemini-2-5-pro-preview-03-25": "Gemini 2.5 pro",
    "gemini-2-5-flash-preview-04-17": "Gemini 2.5 flash",
    "gemini-1-5-pro": "Gemini 1.5 pro",
    "meta-llama-Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama 4 Maverick",
    "Qwen-Qwen2-5-72B-Instruct-Turbo": "Qwen 2.5 72B",
    "qwen-plus-2025-04-28": "Qwen 3", # Added for Qwen thinking experiment
    "meta-llama-Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout",
    "o1": "o1", # Base nickname for o1
    "o3": "o3", # Base nickname for o3
    "deepseek-ai-DeepSeek-V3": "DeepSeek V3", # Add nickname for DeepSeek V3
    "deepseek-ai-DeepSeek-R1": "DeepSeek R1", # Add nickname for DeepSeek R1
    "grok-3-beta": "Grok 3", # Add nickname for grok-3-beta
    "grok-4-0709": "Grok 4", # Add nickname for grok-4-0709
    "deepseek-deepseek-r1-zero:free": "DeepSeek R1 Zero", # Corrected OpenRouter model name with colon
    "vllm:microsoft/Phi-4-reasoning-plus": "Phi 4 Reasoning Plus", # Add nickname for vllm VLLM Phi-4 model
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
    
    # Check for OpenRouter models (models with :free or :pro suffix)
    if '/' in model_lower and (':free' in model_lower or ':pro' in model_lower):
        return 'open_source'
        
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
def generate_benchmark_data(run_files, prices_path, paper_plots=False, leaderboard_only=False):
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
        # --- Create Output Directory EARLY (needed for hard subset analysis) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(BASE_ANALYSIS_DIR, RUNS_SUBDIR, f"full benchmark")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Created analysis run directory: {run_output_dir}")
    
    # --- Find Hard Question Subset (only if leaderboard_only is True) ---
    if leaderboard_only:
        try:
            # Calculate pass rates for individual elements, not captions
            if all_data_frames:
                print(f"\nCalculating hard subset from {len(all_data_frames)} model runs...")
                
                # Track pass rates for individual (caption, element) pairs
                element_pass_counts = {}
                element_appearance_counts = {}
                
                # Count passes and appearances for each individual element
                for df in all_data_frames:
                    for _, row in df.iterrows():
                        element_key = (row['caption'], row['element'])
                        
                        if element_key not in element_pass_counts:
                            element_pass_counts[element_key] = 0
                            element_appearance_counts[element_key] = 0
                            
                        if row['autograder_judgment'] == 'PASS':
                            element_pass_counts[element_key] += 1
                        element_appearance_counts[element_key] += 1
                
                # Calculate pass rates for each element
                element_pass_rates = {}
                for element_key, appearances in element_appearance_counts.items():
                    passes = element_pass_counts.get(element_key, 0)
                    pass_rate = passes / appearances if appearances > 0 else 0
                    element_pass_rates[element_key] = pass_rate
                
                # Create list of (element_key, pass_rate, appearances) tuples
                element_data = [(element_key, element_pass_rates[element_key], element_appearance_counts[element_key]) 
                              for element_key in element_pass_rates.keys()]
                
                # Filter to only include elements attempted by at least 3 models
                eligible_elements = [item for item in element_data if item[2] >= 3]
                
                # Sort by pass rate (ascending) and take the top 100 individual elements
                eligible_elements.sort(key=lambda x: x[1])
                hard_elements = [item[0] for item in eligible_elements[:100]]  # 100 individual elements
                
                if hard_elements:
                    print(f"\nSelected the 100 hardest individual elements (lowest pass rates, at least 3 attempts)")
                else:
                    print("\nNo hard elements found with at least 3 attempts.")
                    return  # Exit early if no hard elements found
                
                # Evaluate each model's performance on the hard subset
                if hard_elements and all_data_frames:
                    print("\n--- Model Performance on Hard Subset ---")
                    
                    hard_subset_results = []
                    
                    for df in all_data_frames:
                        # Extract model identifier
                        if 'explainer_model' in df.columns and len(df['explainer_model'].unique()) == 1:
                            model_name = df['explainer_model'].iloc[0]
                        elif 'source_file' in df.columns and len(df['source_file'].unique()) == 1:
                            # Extract model name from filename if available
                            filename = df['source_file'].iloc[0]
                            match = re.search(r'exp-(.*?)_ag', filename)
                            model_name = match.group(1).replace('_', '-') if match else "unknown"
                        else:
                            model_name = "unknown"
                            
                        # Get display name based on model nicknames
                        display_name = MODEL_NICKNAMES.get(model_name, model_name)
                        
                                                # Filter dataframe to only include hard elements
                        hard_df = df[df.apply(lambda row: (row['caption'], row['element']) in hard_elements, axis=1)]
                        
                        if not hard_df.empty:
                            # Calculate pass rate on hard subset
                            total_hard = len(hard_df)
                            passed_hard = sum(hard_df['autograder_judgment'] == 'PASS')
                            hard_pass_rate = passed_hard / total_hard if total_hard > 0 else 0
                            
                            # Calculate regular pass rate for comparison
                            total_all = len(df)
                            passed_all = sum(df['autograder_judgment'] == 'PASS')
                            all_pass_rate = passed_all / total_all if total_all > 0 else 0
                            
                            # Store results
                            hard_subset_results.append({
                                'model': model_name,
                                'display_name': display_name,
                                'hard_pass_rate': hard_pass_rate,
                                'overall_pass_rate': all_pass_rate,
                                'hard_questions_attempted': total_hard,
                                'hard_questions_passed': passed_hard
                            })
                            
                            # Print result
                            print(f"  {display_name}: {passed_hard}/{total_hard} hard questions passed ({hard_pass_rate:.2%})")
                
                if hard_subset_results:
                    # Convert results to DataFrame
                    hard_perf_df = pd.DataFrame(hard_subset_results)
                    
                    # Sort by overall pass rate (descending) for main leaderboard
                    hard_perf_df = hard_perf_df.sort_values('overall_pass_rate', ascending=False)
                    
                                    # Save to CSV
                hard_perf_path = os.path.join(run_output_dir, "model_performance_on_hard_subset.csv")
                hard_perf_df.to_csv(hard_perf_path, index=False)
                print(f"\nSaved model performance on hard subset to: {hard_perf_path}")
                
                # Create a clean leaderboard format
                leaderboard_df = hard_perf_df.copy()
                leaderboard_df['rank'] = range(1, len(leaderboard_df) + 1)
                leaderboard_df['hard_score_pct'] = (leaderboard_df['hard_pass_rate'] * 100).round(1)
                leaderboard_df['overall_score_pct'] = (leaderboard_df['overall_pass_rate'] * 100).round(1)
                
                # Select and rename columns for clean leaderboard
                leaderboard_clean = leaderboard_df[['rank', 'display_name', 'overall_score_pct', 'hard_score_pct']].copy()
                leaderboard_clean.columns = ['Rank', 'Model', 'Overall (%)', 'Hard Subset (%)']
                
                # Save leaderboard
                leaderboard_path = os.path.join(run_output_dir, "leaderboard.csv")
                leaderboard_clean.to_csv(leaderboard_path, index=False)
                print(f"Saved leaderboard to: {leaderboard_path}")
                
                # Print complete leaderboard in a pretty table format
                print(f"\n🏆 HumorBench Leaderboard (All Models):")
                print("=" * 80)
                print(f"{'Rank':<4} {'Model':<35} {'Overall':<12} {'Hard Subset':<12}")
                print("-" * 80)
                
                for _, row in leaderboard_clean.iterrows():
                    rank = f"{int(row['Rank'])}."
                    model = row['Model'][:34]  # Truncate long model names
                    overall = f"{row['Overall (%)']}%"
                    hard_subset = f"{row['Hard Subset (%)']}%"
                    
                    print(f"{rank:<4} {model:<35} {overall:<12} {hard_subset:<12}")
                
                print("=" * 80)
                print(f"📊 Key Insights:")
                print(f"   • Best Overall Performance: {leaderboard_clean.iloc[0]['Model']} ({leaderboard_clean.iloc[0]['Overall (%)']}%)")
                print(f"   • Best Hard Subset Performance: {leaderboard_clean.loc[leaderboard_clean['Hard Subset (%)'].idxmax(), 'Model']} ({leaderboard_clean['Hard Subset (%)'].max()}%)")
                print(f"   • Average Overall Score: {leaderboard_clean['Overall (%)'].mean():.1f}%")
                print(f"   • Average Hard Subset Score: {leaderboard_clean['Hard Subset (%)'].mean():.1f}%")
                print("=" * 80)
                
                # Exit early when doing leaderboard only
                return
                
            else:
                print("\nNo valid dataframes found. Cannot create hard subset.")
                return
                
        except Exception as e:
            print(f"\nError calculating hard subset: {e}")
            return
    
    # --- Rest of analysis (only runs when leaderboard_only=False) ---
    
    # --- Load category data and create subsets ---
    try:
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
        elif "claude" in model_name.lower() and "thinking_budget" in filename:
            budget_match = re.search(r'thinking_budget_(\d+k?)', filename)
            if budget_match:
                budget_str = budget_match.group(1)
                # Simplified plain name for UI/dropdown
                plain_name = f"{base_nickname} {budget_str}"
                # HTML label for plot hover/text with newline and grey styling
                html_label = f"{base_nickname}<br><span style='color:grey;'>Thinking budget: {budget_str}</span>"

        # 3. Check for Gemini models with thinking budget
        elif ("gemini" in model_name.lower()) and "thinking_budget" in filename:
            budget_match = re.search(r'thinking_budget_(\d+k?)', filename)
            if budget_match:
                budget_str = budget_match.group(1)
                # Simplified plain name for UI/dropdown
                plain_name = f"{base_nickname} {budget_str}"
                # HTML label for plot hover/text with newline and grey styling
                html_label = f"{base_nickname}<br><span style='color:grey;'>Thinking budget: {budget_str}</span>"

        # 4. Check for Qwen Plus 2025-04-28 models with reasoning value
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

    # --- Determine Autograder ---
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
    
    # Save the hard subset to a CSV file if we identified any hard captions
    if 'hard_captions' in locals() and hard_captions:
        try:
            # Read the original comprehensive annotations file to keep its structure
            annotations_path = "../comprehensive_annotations.csv"
            if os.path.exists(annotations_path):
                annotations_df = pd.read_csv(annotations_path)
                
                # Filter for only hard captions
                hard_subset_df = annotations_df[annotations_df['caption'].isin(hard_captions)]
                
                # Add pass rate information
                hard_subset_df['pass_rate'] = hard_subset_df['caption'].map(
                    lambda caption: caption_pass_rates.get(caption, 0)
                )
                hard_subset_df['num_attempts'] = hard_subset_df['caption'].map(
                    lambda caption: caption_appearance_counts.get(caption, 0)
                )
                
                # Sort by pass rate in ascending order
                hard_subset_df = hard_subset_df.sort_values('pass_rate', ascending=True)
                
                # Save to file
                hard_subset_path = os.path.join(run_output_dir, "hard_subset.csv")
                hard_subset_df.to_csv(hard_subset_path, index=False)
                print(f"Saved hard subset ({len(hard_subset_df)} rows) to: {hard_subset_path}")
                
                # Now analyze category representation in the hard subset
                if 'category_data' in locals() and category_data:
                    print("\n--- Category Analysis of Hard Subset ---")
                    
                    # Create a mapping from idx to caption for easier lookup
                    idx_to_caption = {row['idx']: row['caption'] for _, row in annotations_df.iterrows() if not pd.isna(row['idx'])}
                    
                    # Count total number of items per category across all data
                    total_items_per_category = {cat: len(idx_set) for cat, idx_set in category_data.items()}
                    
                    # Count total number of unique items in all categories
                    total_dataset_size = len(annotations_df)
                    
                    # Print overall category distribution first
                    print("\nOverall Category Distribution in Full Dataset:")
                    for category_name, idx_set in category_data.items():
                        total_in_category = total_items_per_category[category_name]
                        percent_of_total = total_in_category / total_dataset_size * 100
                        print(f"  {category_name}: {total_in_category}/{total_dataset_size} items ({percent_of_total:.1f}%)")
                    
                    print("\nHard Subset Category Distribution:")
                    
                    # Count how many items from each category are in the hard subset
                    hard_subset_by_category = {}
                    hard_subset_idx = hard_subset_df['idx'].tolist()
                    
                    for category_name, idx_set in category_data.items():
                        # Find the intersection of hard subset indices and category indices
                        category_hard_items = set(hard_subset_idx).intersection(idx_set)
                        hard_subset_by_category[category_name] = len(category_hard_items)
                        
                        # Calculate and report the percentage of this category in hard subset vs. overall
                        total_in_category = total_items_per_category[category_name]
                        if total_in_category > 0:
                            hard_pct = len(category_hard_items) / total_in_category * 100
                            representation = len(category_hard_items) / len(hard_subset_idx) * 100
                            
                            print(f"Category: {category_name}")
                            print(f"  {len(category_hard_items)}/{total_in_category} ({hard_pct:.1f}%) items in this category are in the hard subset")
                            print(f"  {representation:.1f}% of the hard subset consists of this category")
                            
                            # Determine if over/under-represented
                            overall_pct = total_in_category / total_dataset_size * 100
                            print(f"  Base presence in full dataset: {overall_pct:.1f}% vs {representation:.1f}% in hard subset")
                            
                            if representation > (overall_pct * 1.2):  # 20% higher than expected
                                print(f"  ⚠️ OVER-REPRESENTED: This category appears more frequently in hard subset than overall")
                            elif representation < (overall_pct * 0.8):  # 20% lower than expected
                                print(f"  ⚠️ UNDER-REPRESENTED: This category appears less frequently in hard subset than overall")
                            
                    # Add category flags to the hard subset CSV
                    hard_subset_with_categories = hard_subset_df.copy()
                    
                    # Add category columns
                    for category_name in category_data.keys():
                        hard_subset_with_categories[f"is_{category_name}"] = hard_subset_with_categories['idx'].apply(
                            lambda idx: idx in category_data[category_name]
                        )
                    
                    # Add pass rate information to the enhanced version as well
                    hard_subset_with_categories['pass_rate'] = hard_subset_with_categories['caption'].map(
                        lambda caption: caption_pass_rates.get(caption, 0)
                    )
                    hard_subset_with_categories['num_attempts'] = hard_subset_with_categories['caption'].map(
                        lambda caption: caption_appearance_counts.get(caption, 0)
                    )
                    
                    # Sort by pass rate in ascending order
                    hard_subset_with_categories = hard_subset_with_categories.sort_values('pass_rate', ascending=True)
                    
                    # Save enhanced version with category flags
                    enhanced_path = os.path.join(run_output_dir, "hard_subset_with_categories.csv")
                    hard_subset_with_categories.to_csv(enhanced_path, index=False)
                    print(f"\nSaved enhanced hard subset with category flags to: {enhanced_path}")
        except Exception as e:
            print(f"Error saving or analyzing hard subset: {e}")
            import traceback
            traceback.print_exc()

    # --- Save Processed Data as JSON ---
    # 1. First copy only the columns that definitely exist
    plot_data_df = summary_df[[
        'display_name', 'explainer_model', 'autograder_model',
        'pass_rate_per_row', 'pass_rate_per_caption_all', 'pass_rate_per_caption_some',
        'total_cost', 'explainer_cost', 'avg_cost_per_row',
        'plot_color', 'explainer_family',
        'num_rows', 'num_captions'
    ]]
    
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
                # Try with a more precise matching approach
                for gpqa_model in gpqa_data:
                    # Convert both to lowercase and remove common separators for comparison
                    clean_gpqa_model = gpqa_model.lower().replace('-', '').replace('_', '').replace(' ', '')
                    clean_model_name = model_name.lower().replace('-', '').replace('_', '').replace(' ', '')
                    
                    # Exact match - preferred
                    if clean_model_name == clean_gpqa_model:
                        plot_data_df.at[i, 'gpqa_score'] = gpqa_data[gpqa_model].get('score', None)
                        break
                        
                    # For models like "o3" vs "o3-mini", ensure we don't match substrings 
                    # unless they are the complete model name (preventing o3 matching o3-mini)
                    if (clean_gpqa_model in clean_model_name and 
                        (len(clean_gpqa_model) == len(clean_model_name) or 
                         not any(c.isalnum() for c in clean_model_name[len(clean_gpqa_model):]))):
                        plot_data_df.at[i, 'gpqa_score'] = gpqa_data[gpqa_model].get('score', None)
                        break
                    
                    # Similar logic for reverse case
                    if (clean_model_name in clean_gpqa_model and 
                        (len(clean_model_name) == len(clean_gpqa_model) or 
                         not any(c.isalnum() for c in clean_gpqa_model[len(clean_model_name):]))):
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
    arc_agi_file_path = "../model_data/models_ARC_AGI_with_o.json"
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
                # Try with a more precise matching approach
                for arc_model in arc_agi_data:
                    # Convert both to lowercase and remove common separators for comparison
                    clean_arc_model = arc_model.lower().replace('-', '').replace('_', '').replace(' ', '')
                    clean_model_name = model_name.lower().replace('-', '').replace('_', '').replace(' ', '')
                    
                    # Exact match - preferred
                    if clean_model_name == clean_arc_model:
                        plot_data_df.at[i, 'arc_agi_score'] = arc_agi_data[arc_model].get('score', None)
                        break
                        
                    # For models like "o3" vs "o3-mini", ensure we don't match substrings 
                    # unless they are the complete model name (preventing o3 matching o3-mini)
                    if (clean_arc_model in clean_model_name and 
                        (len(clean_arc_model) == len(clean_model_name) or 
                         not any(c.isalnum() for c in clean_model_name[len(clean_arc_model):]))):
                        plot_data_df.at[i, 'arc_agi_score'] = arc_agi_data[arc_model].get('score', None)
                        break
                    
                    # Similar logic for reverse case
                    if (clean_model_name in clean_arc_model and 
                        (len(clean_model_name) == len(clean_arc_model) or 
                         not any(c.isalnum() for c in clean_arc_model[len(clean_model_name):]))):
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
                # Try with a more precise matching approach
                for arena_model in lmarena_data:
                    # Convert both to lowercase and remove common separators for comparison
                    clean_arena_model = arena_model.lower().replace('-', '').replace('_', '').replace(' ', '')
                    clean_model_name = model_name.lower().replace('-', '').replace('_', '').replace(' ', '')
                    
                    # Exact match - preferred
                    if clean_model_name == clean_arena_model:
                        plot_data_df.at[i, 'lmarena_elo_score'] = lmarena_data[arena_model].get('score', None)
                        break
                        
                    # For models like "o3" vs "o3-mini", ensure we don't match substrings 
                    # unless they are the complete model name (preventing o3 matching o3-mini)
                    if (clean_arena_model in clean_model_name and 
                        (len(clean_arena_model) == len(clean_model_name) or 
                         not any(c.isalnum() for c in clean_model_name[len(clean_arena_model):]))):
                        plot_data_df.at[i, 'lmarena_elo_score'] = lmarena_data[arena_model].get('score', None)
                        break
                    
                    # Similar logic for reverse case
                    if (clean_model_name in clean_arena_model and 
                        (len(clean_model_name) == len(clean_arena_model) or 
                         not any(c.isalnum() for c in clean_arena_model[len(clean_model_name):]))):
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

    # --- Calculate and print benchmark correlations (add this section) ---
    try:
        # Calculate correlations with the main HumorBench metric (pass_rate_per_row) only
        calculate_benchmark_correlations(plot_data_df, "pass_rate_per_row")
    except Exception as e:
        print(f"Error calculating benchmark correlations: {e}")
        import traceback
        traceback.print_exc()

    # --- Generate reports based on options ---
    if paper_plots:
        print("\nGenerating paper-ready plots only...")
        # Generate Static Scatter Plots for Academic Paper
        generate_static_scatter_plots(run_output_dir, data_json_path)
        # Create combined benchmark plots image
        create_combined_benchmark_plots(run_output_dir)
    else:
        # Generate Interactive HTML Report
        create_interactive_report(run_output_dir, data_json_path, report_autograder)
        
        # Generate Static Scatter Plots for Academic Paper
        generate_static_scatter_plots(run_output_dir, data_json_path)
        
        # Create combined benchmark plots image
        create_combined_benchmark_plots(run_output_dir)


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

def create_combined_benchmark_plots(run_output_dir):
    """Creates a single image with all three benchmark plots in a row."""
    try:
        print("\nCreating combined benchmark plots image...")
        
        # Only include the 3 benchmark plots (exclude cost plot)
        expected_filenames = [
            "full_bench_gpqa.png",       # GPQA plot
            "full_bench_arc.png",        # ARC-AGI plot
            "full_bench_lmarena.png"     # LM Arena ELO plot
        ]
        
        plot_files = []
        for filename in expected_filenames:
            filepath = os.path.join(run_output_dir, filename)
            if os.path.exists(filepath):
                plot_files.append(filepath)
                print(f"Found plot file: {filename}")
        
        if not plot_files:
            print("Error: Could not find any benchmark plot images")
            return
        
        print(f"Found {len(plot_files)} plot files to combine")
        
        # Create figure with subplots in a row - make it very wide
        # Adjust width based on number of plots
        width_per_plot = 8  # inches per plot
        fig_width = max(width_per_plot * len(plot_files), 20)  # minimum 20 inches
        fig, axes = plt.subplots(1, len(plot_files), figsize=(fig_width, 8))
        
        # If only one plot was found, convert axes to list for consistent handling
        if len(plot_files) == 1:
            axes = [axes]
        
        # Load and display each plot in its subplot
        for i, plot_path in enumerate(plot_files):
            try:
                img = plt.imread(plot_path)
                axes[i].imshow(img)
                axes[i].axis('off')  # Hide axes
                
                # Add plot type as title
                basename = os.path.basename(plot_path)
                if "gpqa" in basename:
                    axes[i].set_title("GPQA vs. Performance", fontsize=22)
                elif "arc" in basename:
                    axes[i].set_title("ARC-AGI vs. Performance", fontsize=22)
                elif "lmarena" in basename:
                    axes[i].set_title("LM Arena ELO vs. Performance", fontsize=22)
            except Exception as e:
                print(f"Error loading plot {plot_path}: {e}")
                # Show error message in this subplot
                axes[i].text(0.5, 0.5, f"Error loading image", 
                           ha='center', va='center', fontsize=16)
                axes[i].axis('on')

        # Remove extra spacing between subplots and add overall title
        plt.subplots_adjust(wspace=0.05)
        plt.suptitle("HumorBench vs. Reference Benchmarks", fontsize=24, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Save the combined figure with a different name to avoid overwriting
        combined_path = os.path.join(run_output_dir, "combined_benchmark_comparisons.png")
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined benchmark plots to: {combined_path}")
        
    except Exception as e:
        print(f"Error creating combined benchmark plots: {e}")
        import traceback
        traceback.print_exc()


def generate_static_scatter_plots(run_output_dir, data_json_path):
    """Generates high-quality static scatter plots for academic papers with accessibility features."""
    print("\nGenerating static scatter plots for academic papers...")
    
    # Import necessary libraries for broken axis
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    
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
    
    # --- Define offsets for model labels based on x-axis ---
    # Offsets for the cost plot (these are the existing values)
    cost_offsets = {
        "DeepSeek R1": (15, -5),
        "Grok 3": (-15, -25),
        "Grok 4": (-70, -25),
        "o4-mini": (-85, 0),
        "Claude 3.7 Sonnet": (10, 10),
        "Claude Sonnet 4": (5, -25),
        "Claude Opus 4": (-80, 10),
        "gpt-4o": (5, -20),
        "Gemini 2.5 pro": (-160, 10),
        "o3": (10, -5),
        "o1": (-10, 12),
        "o3-mini": (-30, 15),
        "DeepSeek R1 Zero": (5, -25),

    }
    
    # Placeholder offsets for GPQA plot (only one example model set, others at default 0,0)
    gpqa_offsets = {
        "gpt-4o": (15, -20),
        "Gemini 2.5 pro": (-160, 25),
        "Gemini 1.5 pro": (-60, 10),
        "Claude 3.7 Sonnet": (-195, 0),
        "o4-mini": (-90, -5),
        "o1": (-40, -10),
        "o3": (-35, -10),
        "Grok 3": (-80, -20),
        "DeepSeek R1": (-90, -30),
    }
    
    # Placeholder offsets for ARC-AGI plot
    arc_agi_offsets = {
        "gpt-4o": (15, 10),  # Example offset - user will fill in the rest
        "Gemini 2.5 pro": (-100, 15),
        "Claude 3.7 Sonnet": (-90, -30),
        "o3": (-30, -20),
        "o3-mini": (10, -15),  # Add specific offset for o3-mini
        "Grok 4": (-60, -25)  # Add offset for Grok 4
    }
    
    # Placeholder offsets for LM Arena ELO plot
    lmarena_elo_offsets = {
        "Gemini 2.5 pro": (-135, 15),
        "Gemini 1.5 pro": (-55, 20),
        "o3": (10, -10),
        "o1": (-40, -30),

    }
    
    # Map x_axis field to the appropriate offset dictionary
    axis_to_offsets = {
        "explainer_cost": cost_offsets,
        "gpqa_score": gpqa_offsets,
        "arc_agi_score": arc_agi_offsets,
        "lmarena_elo_score": lmarena_elo_offsets
    }
    
    # Set up figure style for academic paper with bigger text/labels
    plt.rcParams.update({
        'font.size': 100,             
        'axes.labelsize': 22,        # Increased from 18 to 22
        'axes.titlesize': 24,        # Increased from 20 to 24
        'xtick.labelsize': 20,       # Increased from 14 to 16
        'ytick.labelsize': 20,       # Increased from 14 to 16
        'legend.fontsize': 15,       # Increased from 14 to 22
        'legend.title_fontsize': 15, # Increased from 16 to 24
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
        {"field": "explainer_cost", "label": "Run Cost ($)", "log_scale": True},
        {"field": "gpqa_score", "label": "GPQA Diamond Score (%)", "log_scale": False},
        {"field": "arc_agi_score", "label": "ARC-AGI Score (%)", "log_scale": False},
        {"field": "lmarena_elo_score", "label": "LM Arena ELO Score", "log_scale": False}
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
                
                # Special handling for ARC-AGI plot with ellipsis in x-axis
                if x_option["field"] == "arc_agi_score":
                    # Create a single figure for a simpler broken axis approach
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Identify high-scoring models (ARC-AGI score > 60%) that should go on the right side
                    threshold = 0.60  # 60% threshold
                    high_scoring_data = plot_df[plot_df[x_option["field"]] > threshold]
                    regular_data = plot_df[plot_df[x_option["field"]] <= threshold]
                    
                    # Get the offset dictionary for this x-axis
                    current_offsets = axis_to_offsets.get(x_option["field"], {})
                    
                    # Plot regular models (ARC-AGI score <= 60%)
                    for family, group in regular_data.groupby('explainer_family'):
                        marker = family_markers.get(family, 'o')
                        color = FAMILY_COLORS.get(family, 'gray')
                        marker_size = family_marker_sizes.get(family, 180)
                        line_width = family_marker_linewidths.get(family, 2)
                        
                        ax.scatter(
                            group[x_option["field"]] * 100,  # Convert to percentage
                            group[y_option["field"]] * 100,  # Convert to percentage
                            marker=marker,
                            s=marker_size,
                            color=color,
                            edgecolors='black',
                            linewidths=line_width,
                            alpha=0.8,
                            label=family if family == 'XAI' else 'OpenAI' if family == 'openai' else family.replace('_', ' ').title()
                        )
                    
                    # Plot high-scoring models on the right side of the break
                    special_x_positions = [42, 47]  # Positions for high-scoring models (sorted by ARC-AGI score)
                    # Sort high scoring data by ARC-AGI score to match label order
                    high_scoring_sorted = high_scoring_data.sort_values(x_option["field"])
                    for i, (_, row) in enumerate(high_scoring_sorted.iterrows()):
                        family = row['explainer_family']
                        marker = family_markers.get(family, 'o')
                        color = FAMILY_COLORS.get(family, 'gray')
                        marker_size = family_marker_sizes.get(family, 180)
                        line_width = family_marker_linewidths.get(family, 2)
                        
                        # Use a different x position for each high-scoring model
                        x_pos = special_x_positions[i % len(special_x_positions)]
                        
                        # Check if we need to add family label 
                        family_label = family if family == 'XAI' else 'OpenAI' if family == 'openai' else family.replace('_', ' ').title()
                        # For simplicity, only label the first high-scoring model in each family
                        is_first_in_family = i == 0 or high_scoring_sorted.iloc[i-1]['explainer_family'] != family
                        label = family_label if is_first_in_family else None
                        
                        ax.scatter(
                            x_pos,
                            row[y_option["field"]] * 100,  # y-value as percentage
                            marker=marker,
                            s=marker_size,
                            color=color,
                            edgecolors='black',
                            linewidths=line_width,
                            alpha=0.8,
                            label=label
                        )
                    
                    # Add model name annotations for regular models
                    for _, row in regular_data.iterrows():
                        x_val = row[x_option["field"]] * 100  # Convert to percentage
                        y_val = row[y_option["field"]] * 100  # Convert to percentage
                        model_name = row['display_name']
                        
                        offset_x, offset_y = 7, 7  # Default offset
                        if model_name in current_offsets:
                            offset_x, offset_y = current_offsets[model_name]

                        ax.annotate(
                            model_name,
                            xy=(x_val, y_val),
                            xytext=(offset_x, offset_y),
                            textcoords='offset points',
                            fontsize=20,
                        )
                    
                    # Add annotations for high-scoring models
                    special_x_positions = [42, 47]  # Same positions as above
                    # Use the same sorted order as plotting and labels
                    for i, (_, row) in enumerate(high_scoring_sorted.iterrows()):
                        model_name = row['display_name']
                        x_pos = special_x_positions[i % len(special_x_positions)]
                        y_val = row[y_option["field"]] * 100
                        
                        offset_x, offset_y = 7, 7  # Default offset
                        if model_name in current_offsets:
                            offset_x, offset_y = current_offsets[model_name]
                        
                        ax.annotate(
                            model_name,
                            xy=(x_pos, y_val),
                            xytext=(offset_x, offset_y),
                            textcoords='offset points',
                            fontsize=20,
                        )
                    
                    # Set axis limits to accommodate both regular and special positions
                    ax.set_xlim(0, 50)
                    
                    # Set up clean x-ticks: regular scale from 0-35%, then ellipsis, then high scores
                    regular_ticks = list(range(0, 36, 5))  # [0, 5, 10, 15, 20, 25, 30, 35]
                    ellipsis_pos = 38
                    high_score_ticks = [42, 47]  # Positions for high-scoring models (Grok 4, o3) - more spread out
                    
                    all_ticks = regular_ticks + [ellipsis_pos] + high_score_ticks
                    ax.set_xticks(all_ticks)
                    
                    # Create corresponding labels
                    regular_labels = [str(x) for x in regular_ticks]
                    
                    # Create dynamic labels based on actual model scores (sorted by ARC-AGI score)
                    high_score_labels = []
                    if not high_scoring_data.empty:
                        # Sort high scoring data by ARC-AGI score to ensure correct order
                        high_scoring_sorted = high_scoring_data.sort_values(x_option["field"])
                        for _, row in high_scoring_sorted.iterrows():
                            score_pct = int(row[x_option["field"]] * 100)  # Convert to integer percentage
                            high_score_labels.append(str(score_pct))
                    
                    all_labels = regular_labels + ["..."] + high_score_labels
                    ax.set_xticklabels(all_labels)
                    
                    # Add grid for readability
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Add axis labels
                    ax.set_xlabel(x_option["label"], fontsize=22)
                    ax.set_ylabel(y_option["label"], fontsize=22)
                    
                    # Add legend in bottom right corner
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    legend = ax.legend(by_label.values(), by_label.keys(), 
                                      title="Model Family", title_fontsize=16,
                                      loc='lower right', frameon=True, framealpha=0.9)
                    legend.get_frame().set_linewidth(2)
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save figure
                    filename = f"full_bench_{x_option['field'].split('_')[0]}.png"
                    filepath = os.path.join(run_output_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Saved simplified ARC-AGI plot with ellipsis to: {filepath}")
                
                # Regular handling for other plots
                else:
                    # Create figure with appropriate size for academic paper
                    plt.figure(figsize=(10, 8))
                    
                    # Lists to store annotations for adjustText
                    texts = []
                    
                    # Get the offset dictionary for this x-axis
                    current_offsets = axis_to_offsets.get(x_option["field"], {})
                    
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
                            label=family if family == 'XAI' else 'OpenAI' if family == 'openai' else family.replace('_', ' ').title()  # Special format for XAI and OpenAI
                        )
                    
                        # Add model names as annotations with larger font
                        for _, row in group.iterrows():
                            x_val = row[x_option["field"]] * (100 if x_option["field"] in ["gpqa_score", "arc_agi_score"] else 1)
                            y_val = row[y_option["field"]] * 100
                            model_name = row['display_name']
                            
                            # Default offset if not specified for this model in the current plot type
                            offset_x, offset_y = 7, 7
                            
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
                    
                    # Set axis labels with larger font
                    plt.xlabel(x_option["label"], fontsize=22)
                    plt.ylabel(y_option["label"], fontsize=22)
                    
                    if x_option["field"] == "explainer_cost":
                        plot_title = f"HumorBench"
                    # plt.title(plot_title, fontsize=24)
                    
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
                    legend_position = 'lower right'  # Consistently use lower right for standard plots
                    legend = plt.legend(title="Model Family", title_fontsize=16, 
                                      loc=legend_position, frameon=True, framealpha=0.9)
                    legend.get_frame().set_linewidth(2)
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Create different filenames for each plot type
                    if x_option["field"] == "explainer_cost":
                        plot_type = "cost"
                    else:
                        plot_type = x_option["field"].split("_")[0]  # Get first part of field name
                        
                    filename = f"full_bench_{plot_type}.png"
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
                    label=family if family == 'XAI' else 'OpenAI' if family == 'openai' else family.replace('_', ' ').title()
                )
        
        plt.axis('off')  # Hide axes
        # Create legend
        plt.legend(title="Model Families", title_fontsize=16, fontsize=22,
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


# --- Add a function to calculate benchmark correlations ---
def calculate_benchmark_correlations(df, y_field="pass_rate_per_row"):
    """Calculate and print Spearman rank correlations between benchmark metrics.
    
    Args:
        df: DataFrame with benchmark data
        y_field: The HumorBench metric field to use (default: pass_rate_per_row)
    """
    print("\n--- Benchmark Correlation Analysis (Spearman's Rank Correlation) ---")
    
    # List of benchmark fields to correlate with HumorBench
    benchmark_fields = [
        ("gpqa_score", "GPQA Diamond Score"),
        ("arc_agi_score", "ARC-AGI Score"),
        ("lmarena_elo_score", "LM Arena ELO Score")
    ]
    
    # Filter for only benchmarks with data
    available_benchmarks = [(field, name) for field, name in benchmark_fields 
                           if field in df.columns and df[field].notna().any()]
    
    if not available_benchmarks:
        print("  No other benchmark data available for correlation analysis.")
        return
    
    # Print header
    print(f"  Correlating with HumorBench metric: {y_field}")
    print("  " + "-" * 60)
    print(f"  {'Benchmark':<25} | {'Correlation':<12} | {'p-value':<12} | {'# Models'}")
    print("  " + "-" * 60)
    
    # Calculate correlations
    for field, name in available_benchmarks:
        # Filter rows with valid data for both metrics
        valid_data = df.dropna(subset=[field, y_field])
        n_models = len(valid_data)
        
        if n_models < 3:
            print(f"  {name:<25} | {'N/A':<12} | {'N/A':<12} | {n_models} (too few for correlation)")
            continue
            
        # Calculate Spearman correlation
        corr, p_value = spearmanr(valid_data[field], valid_data[y_field])
        
        # Format significance markers
        sig_marker = ""
        if p_value < 0.001:
            sig_marker = "***"
        elif p_value < 0.01:
            sig_marker = "**"
        elif p_value < 0.05:
            sig_marker = "*"
            
        # Print result
        print(f"  {name:<25} | {corr:.3f}{sig_marker:<9} | {p_value:.3f}{'ᵃ':<9} | {n_models}")
    
    # Print footer with significance explanation
    print("  " + "-" * 60)
    print("  * p<0.05, ** p<0.01, *** p<0.001, ᵃ p-value precision may be limited for small sample sizes")
    print("  Note: Spearman's correlation measures monotonic rank associations\n")


if __name__ == "__main__":
    # Add argument parsing for paper_plots and leaderboard options
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark results")
    parser.add_argument("--paper-plots", action="store_true", 
                      help="Generate only paper-ready scatter plots without interactive HTML")
    parser.add_argument("--leaderboard", action="store_true",
                      help="Generate only leaderboard analysis with overall and hard subset scores (no plots or other reports)")
    args = parser.parse_args()

    # Run the data generation and report creation
    generate_benchmark_data(RUN_FILES_TO_ANALYZE, MODEL_PRICES_PATH, 
                          paper_plots=args.paper_plots, 
                          leaderboard_only=args.leaderboard)
