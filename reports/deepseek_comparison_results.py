import os
import sys
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
try:
    # Import DeepSeek tokenizer for accurate token counting
    from deepseek_tokenizer import ds_token
    HAS_DEEPSEEK_TOKENIZER = True
except ImportError:
    print("Warning: DeepSeek tokenizer not found. Will use API usage token counts instead.")
    print("Install with: pip install deepseek-tokenizer")
    HAS_DEEPSEEK_TOKENIZER = False

# --- Configuration ---
# Files to analyze for DeepSeek models
RUN_FILES_TO_ANALYZE = [
    # "../runs/vllm/20250508_041347_vllm_phi4reason_exp-vllm_microsoft_Phi_4_reasoning_ag-gpt_4o.csv",
    "../runs/main/20250428_145241_deepseek_v3_exp-deepseek_ai_DeepSeek_V3_ag-gpt_4o.csv",
    "../runs/main/20250428_151728_deepseek_r1_exp-deepseek_ai_DeepSeek_R1_ag-gpt_4o.csv",
    "../runs/main/20250513_114914_openrouter_deepseek_r1_zero_exp-deepseek_deepseek_r1_zero:free_ag-gpt_4o.csv",
    # "../runs/vllm/summarized/summarized_regraded_20250508_041347_vllm_phi4reason_exp-vllm_microsoft_Phi_4_reasoning_ag-gpt_4o.csv",
    # "../runs/vllm/summarized/summarized_regraded_20250508_034834_vllm_phi4plus_exp-vllm_microsoft_Phi_4_reasoning_plus_ag-gpt_4o.csv"
    # "../runs/truncated/truncated_500_tokens_20250513_134008.csv"
]

# Map extracted model names to display names
MODEL_NICKNAMES = {
    "deepseek-ai-DeepSeek-V3": "DeepSeek V3",
    "deepseek-ai-DeepSeek-R1": "DeepSeek R1", 
    "deepseek-deepseek-r1-zero:free": "DeepSeek R1 Zero"
}

MODEL_PRICES_PATH = "../model_data/model_prices.json"
BASE_ANALYSIS_DIR = "../analysis"
RUNS_SUBDIR = "runs"

# Update FAMILY_COLORS to rename families
FAMILY_COLORS = {
    'open_source': 'lightblue',
    'unknown': 'grey'
}

# --- Helper Functions ---
def count_tokens(text):
    """Count tokens in text using DeepSeek tokenizer or estimate if not available."""
    if not text:
        return 0
    
    if HAS_DEEPSEEK_TOKENIZER:
        try:
            tokens = ds_token.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            # Fall back to rough estimation
            return len(text.split())
    else:
        # Rough estimation if DeepSeek tokenizer is not available
        return len(text.split())

def infer_family_from_model(model):
    """Infer the model family from the model name string."""
    model_lower = model.lower()
    
    if 'deepseek' in model_lower:
        return 'open_source'
    else:
        print(f"Warning: Could not definitively infer family for {model}. Assigning 'unknown'.")
        return 'unknown'

def calculate_explainer_cost(row, model_prices):
    """Calculate the cost for just the explainer model (no autograder cost)."""
    explainer_model = row.get('explainer_model')
    explainer_usage_str = row.get('explainer_usage', '{}')
    
    cost = 0.0
    
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
def generate_benchmark_data(run_files, prices_path):
    """Analyzes benchmark runs and generates data for DeepSeek models."""
    run_summaries = []
    autograder_models_found = set()
    all_data_frames = []

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

    print(f"\nAnalyzing DeepSeek model runs...")
    print("-" * 30)

    for file_path in run_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}...")
        
        try:
            df = pd.read_csv(file_path)

            # Verify required columns
            required_cols = ['explainer_model', 'autograder_model', 'autograder_judgment', 'explainer_usage', 'autograder_usage', 'caption']
            if not all(col in df.columns for col in required_cols):
                print(f"  Skipping: Missing required columns. Needed: {', '.join(required_cols)}")
                continue

            # Filter out rows where judgment is not PASS or FAIL
            valid_df = df[df['autograder_judgment'].isin(['PASS', 'FAIL'])].copy()
            if valid_df.empty:
                 print(f"  Skipping: No valid 'PASS' or 'FAIL' judgments found.")
                 continue
                 
            # Store filename in dataframe
            valid_df['source_file'] = filename
            all_data_frames.append(valid_df)

            # Calculate metrics
            num_rows = len(valid_df)
            passes_row = sum(valid_df['autograder_judgment'] == 'PASS')
            pass_rate_per_row = passes_row / num_rows if num_rows > 0 else 0
            print(f"  Metrics - Per Row: Items={num_rows}, Pass Rate={pass_rate_per_row:.2%}")

            # Extract Explainer Model from filename
            match = re.search(r'exp-(.*?)_ag', filename)
            if not match:
                print(f"  Skipping: Could not extract explainer model name from filename.")
                continue
            explainer_model_from_filename = match.group(1).replace('_', '-')

            # Get autograder model
            autograder_model = df['autograder_model'].iloc[0]
            autograder_models_found.add(autograder_model)

            # Calculate costs
            total_cost = 0
            explainer_cost = 0
            if model_prices:
                df['temp_cost'] = df.apply(lambda row: calculate_cost(row, model_prices), axis=1)
                total_cost = df['temp_cost'].sum()
                
                df['explainer_cost'] = df.apply(lambda row: calculate_explainer_cost(row, model_prices), axis=1)
                explainer_cost = df['explainer_cost'].sum()
                
                avg_cost_per_row = total_cost / len(df) if len(df) > 0 else 0
            print(f"  Metrics - Cost: Total=${total_cost:.4f}, Explainer Only=${explainer_cost:.4f}")

            # Calculate mean explainer output tokens
            mean_output_tokens = 0
            explainer_output_tokens = []
            if 'explanation' in df.columns:
                for explanation in df['explanation']:
                    if pd.notna(explanation):
                        token_count = count_tokens(explanation)
                        explainer_output_tokens.append(token_count)
                
                if explainer_output_tokens:
                    mean_output_tokens = sum(explainer_output_tokens) / len(explainer_output_tokens)
                    print(f"  Metrics - Mean Explanation Tokens: {mean_output_tokens:.2f}")
                else:
                    print("  Warning: No valid explanation text found for token counting.")
            else:
                print("  Warning: 'explanation' column not found in the data. Cannot count tokens.")

            # Append to summary
            run_summaries.append({
                'explainer_model': explainer_model_from_filename,
                'autograder_model': autograder_model,
                'num_rows': num_rows,
                'pass_rate_per_row': pass_rate_per_row,
                'explainer_cost': explainer_cost,
                'mean_output_tokens': mean_output_tokens,
                'source_file': filename
            })

        except FileNotFoundError:
            print(f"  Skipping: File not found.")
        except Exception as e:
            print(f"  Skipping: Error processing file - {e}")

    # Create summary DataFrame
    if not run_summaries:
        print("\nNo valid runs found or processed.")
        return

    summary_df = pd.DataFrame(run_summaries)
    summary_df = summary_df.sort_values(by='pass_rate_per_row', ascending=True)
    
    # Generate display names
    def generate_display_names(model_name):
        """Generate display names for the models."""
        return MODEL_NICKNAMES.get(model_name, model_name)

    summary_df['display_name'] = summary_df['explainer_model'].apply(generate_display_names)
    
    # Infer family and assign colors
    summary_df['explainer_family'] = summary_df['explainer_model'].apply(infer_family_from_model)
    summary_df['plot_color'] = summary_df['explainer_family'].map(FAMILY_COLORS).fillna(FAMILY_COLORS['unknown'])

    # Create output directory
    run_output_dir = os.path.join(BASE_ANALYSIS_DIR, RUNS_SUBDIR, "deepseek_comparison")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Created analysis run directory: {run_output_dir}")

    # Prepare data for plotting
    plot_data_df = summary_df[['display_name', 'explainer_model', 'pass_rate_per_row', 'explainer_cost', 'plot_color', 'explainer_family', 'mean_output_tokens']]
    
    # Only generate the scatter plot when there are multiple models
    if len(run_files) > 1:
        # Generate the plot
        generate_deepseek_plot(run_output_dir, plot_data_df)
    else:
        print("Skipping scatter plot generation as there's only one model.")
    
    # --- Generate Histogram of Explainer Output Tokens if only one run file ---
    if len(run_files) == 1 and all_data_frames:
        try:
            print("\nGenerating histogram of explanation tokens for the single run...")
            single_run_df = all_data_frames[0]  # Get the DataFrame for the single processed run
            explainer_output_tokens = []

            if 'explanation' in single_run_df.columns:
                for explanation in single_run_df['explanation']:
                    if pd.notna(explanation):
                        token_count = count_tokens(explanation)
                        explainer_output_tokens.append(token_count)
            else:
                print("Warning: 'explanation' column not found in the run data. Cannot generate token histogram.")

            if explainer_output_tokens:
                plt.figure(figsize=(10, 6))
                plt.hist(explainer_output_tokens, bins='auto', color='mediumpurple', rwidth=0.85)
                plt.xlabel("Explanation Tokens (DeepSeek Tokenizer)", fontsize=22)
                plt.ylabel("Frequency (Number of Rows)", fontsize=22)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()

                model_name = summary_df['display_name'].iloc[0].replace(" ", "_").lower()
                histogram_path = os.path.join(run_output_dir, f"{model_name}_tokens_histogram.png")
                plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved explainer output tokens histogram to: {histogram_path}")
                # Add run-specific histogram statistics to the summary information
                print(f"  Histogram Statistics for {model_name}:")
                print(f"  Min Tokens: {min(explainer_output_tokens)}")
                print(f"  Max Tokens: {max(explainer_output_tokens)}")
                print(f"  Mean Tokens: {sum(explainer_output_tokens)/len(explainer_output_tokens):.2f}")
                print(f"  Median Tokens: {sorted(explainer_output_tokens)[len(explainer_output_tokens)//2]}")
            elif 'explanation' in single_run_df.columns:
                print("Warning: No valid explanation text found to generate histogram.")

        except Exception as e:
            print(f"Error generating or saving explainer output tokens histogram: {e}")
    
    return summary_df

def generate_deepseek_plot(run_output_dir, plot_data_df):
    """Generates the performance vs mean tokens plot for DeepSeek models."""
    print("\nGenerating DeepSeek performance vs mean tokens plot...")
    
    # Set up figure style
    plt.rcParams.update({
        'font.size': 150,             
        'axes.labelsize': 16,
        'axes.titlesize': 24,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 22,
        'legend.title_fontsize': 22,
        'lines.linewidth': 2.5,
        'lines.markersize': 12,
        'lines.markeredgewidth': 2,
        'figure.dpi': 300,
    })
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Define marker shape and size
    marker = 'D'  # Diamond for open source models
    marker_size = 100
    
    # Define label position offsets for each model
    offsets = {
        "DeepSeek V3": (10, 10),
        "DeepSeek R1": (-5, 15),
        "DeepSeek R1 Zero": (-180, -50),
    }
    
    # Plot scatter points
    plt.scatter(
        plot_data_df['mean_output_tokens'],
        plot_data_df['pass_rate_per_row'] * 100,  # Convert to percentage
        marker=marker,
        s=marker_size,
        color='lightblue',
        edgecolors='black',
        linewidths=1,
        alpha=0.8
    )
    
    # Add model labels
    for _, row in plot_data_df.iterrows():
        display_name = row['display_name']
        x_val = row['mean_output_tokens']
        y_val = row['pass_rate_per_row'] * 100
        
        # Default offset if not specified
        offset_x, offset_y = 7, 7
        
        # Apply model-specific offset if available
        if display_name in offsets:
            offset_x, offset_y = offsets[display_name]
        
        # Add annotation
        if display_name == "DeepSeek R1 Zero":
            plt.annotate(
                "DeepSeek R1 Zero\n(clipped output)",
                xy=(x_val, y_val),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=20,
            )
        else:
            plt.annotate(
                display_name,
                xy=(x_val, y_val),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=20,
            )
    
    # Set axis labels
    plt.xlabel("Mean Output Tokens", fontsize=20)
    plt.ylabel("HumorBench Score (%)", fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust y-axis range for better visibility
    plt.ylim(70, 82)
    
    # Apply tight layout
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(run_output_dir, "deepseek_comparison_plot.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved DeepSeek comparison plot to: {filepath}")

if __name__ == "__main__":
    generate_benchmark_data(RUN_FILES_TO_ANALYZE, MODEL_PRICES_PATH) 