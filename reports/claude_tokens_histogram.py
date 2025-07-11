import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Set style for better visualization
plt.style.use('ggplot')
sns.set_palette("muted")
sns.set_context("notebook", font_scale=1.2)

# Define paths to Claude experiment files
CLAUDE_EXPERIMENT_FILES = [
    "../runs/claude_thinking_experiment/20250416_193729_claude_thinking_budget_1024_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    # "../runs/claude_thinking_experiment/20250416_200436_claude_thinking_budget_2048_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    "../runs/claude_thinking_experiment/20250416_204832_claude_thinking_budget_4096_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    # "../runs/claude_thinking_experiment/20250506_171410_claude_thinking_budget_8192_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv",
    "../runs/claude_thinking_experiment/20250506_174903_claude_thinking_budget_16384_exp-claude_3_7_sonnet_latest_ag-gpt_4o.csv"
]

# Function to extract budget from filename
def extract_budget(filename):
    match = re.search(r'budget_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# Function to extract tokens from usage data
def extract_tokens(usage_str):
    try:
        if pd.notna(usage_str):
            usage_json = json.loads(str(usage_str))
            tokens_out = usage_json.get('tokens_out')
            if tokens_out is not None:
                return int(tokens_out)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None

def create_overlay_histogram(all_data, output_dir):
    """Create a simplified overlay histogram of the active budgets"""
    # Create clean plot with white background
    plt.figure(figsize=(12, 8))
    plt.style.use('default')
    
    # Define attractive colors for each budget
    colors = {
        1024: '#7EB6D9',  # light blue
        4096: '#F8A65D',  # orange
        16384: '#78C47A'  # green
    }
    
    # Common bins for consistent comparison
    all_tokens = []
    for tokens in all_data.values():
        all_tokens.extend(tokens)
    
    # Create bins that cover the range of all data
    min_val = min(all_tokens)
    max_val = max(all_tokens)
    bins = np.linspace(min_val, max_val, 25)
    
    # Sort budgets for correct layering (largest first)
    budgets = sorted([(int(name.split(": ")[1]), name, tokens) 
                     for name, tokens in all_data.items()], 
                     key=lambda x: x[0], reverse=True)
    
    # Plot each histogram with transparency in reverse order (largest first)
    for budget_num, budget_name, tokens in budgets:
        plt.hist(tokens, bins=bins, alpha=0.5, label=f'Thinking Budget: {budget_num}',
                color=colors.get(budget_num), edgecolor='white', linewidth=0.5)
    
    # Add labels and title
    plt.xlabel('Output Tokens', fontsize=35)
    plt.ylabel('Count', fontsize=35)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    
    # Clean up the plot
    plt.tight_layout()
    
    # Save the combined overlay
    plt.savefig(os.path.join(output_dir, "claude_tokens_overlay.png"), dpi=300, bbox_inches='tight')
    print(f"Saved overlay histogram to claude_tokens_overlay.png")
    plt.close()

def generate_token_histograms():
    print("Generating token histograms for Claude reasoning budgets...")
    
    # Create a directory for the plots
    output_dir = "../analysis/claude_token_histograms"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to hold data for combined plot
    all_budgets_data = {}
    
    # Process each experiment file
    for file_path in CLAUDE_EXPERIMENT_FILES:
        try:
            # Extract budget size from filename
            budget = extract_budget(file_path)
            if not budget:
                print(f"Couldn't extract budget from {file_path}, skipping...")
                continue
                
            print(f"Processing file for budget {budget}...")
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Extract output tokens
            output_tokens = []
            for usage_str in df['explainer_usage']:
                tokens = extract_tokens(usage_str)
                if tokens is not None:
                    output_tokens.append(tokens)
            
            if not output_tokens:
                print(f"No valid token data found in {file_path}, skipping...")
                continue
                
            # Calculate statistics
            mean_tokens = np.mean(output_tokens)
            median_tokens = np.median(output_tokens)
            min_tokens = min(output_tokens)
            max_tokens = max(output_tokens)
            
            print(f"Budget {budget}: Mean={mean_tokens:.1f}, Median={median_tokens:.1f}, Min={min_tokens}, Max={max_tokens}")
            
            # Store data for combined plot
            all_budgets_data[f"Budget: {budget}"] = output_tokens
            
            # Create histogram for this budget
            plt.figure(figsize=(10, 6))
            sns.histplot(output_tokens, kde=True, bins=25, color='steelblue')
            
            # Add vertical line for mean
            plt.axvline(mean_tokens, color='crimson', linestyle='--', linewidth=2, label=f'Mean: {mean_tokens:.1f}')
            plt.axvline(median_tokens, color='darkgreen', linestyle='-.', linewidth=2, label=f'Median: {median_tokens:.1f}')
            
            # Labels and title
            plt.title(f'Claude 3.7 Output Tokens - Thinking Budget: {budget}', fontsize=16)
            plt.xlabel('Output Tokens', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            
            # Save to file
            file_name = f"claude_tokens_budget_{budget}.png"
            plt.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches='tight')
            print(f"Saved histogram to {file_name}")
            plt.close()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create a combined box plot of all budgets
    if all_budgets_data:
        # Create the overlay histogram of just the active budgets
        create_overlay_histogram(all_budgets_data, output_dir)
        
        plt.figure(figsize=(12, 7))
        
        # Convert to DataFrame for easier plotting
        combined_df = pd.DataFrame({budget: pd.Series(tokens) for budget, tokens in all_budgets_data.items()})
        
        # Create violin plot with embedded box plot
        sns.violinplot(data=combined_df, palette="muted", inner="box")
        
        # Add data points
        sns.stripplot(data=combined_df, palette="dark:black", size=3, alpha=0.3, jitter=True)
        
        # Add labels and title
        plt.title('Claude 3.7 Output Tokens by Thinking Budget', fontsize=16)
        plt.xlabel('Thinking Budget', fontsize=12)
        plt.ylabel('Output Tokens', fontsize=12)
        plt.xticks(rotation=45)
        
        # Save to file
        plt.savefig(os.path.join(output_dir, "claude_tokens_all_budgets.png"), dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to claude_tokens_all_budgets.png")
        plt.close()
        
        # Create a box plot for an alternative visualization
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=combined_df, palette="muted")
        
        # Add labels and title
        plt.title('Claude 3.7 Output Tokens by Thinking Budget (Box Plot)', fontsize=16)
        plt.xlabel('Thinking Budget', fontsize=12)
        plt.ylabel('Output Tokens', fontsize=12)
        plt.xticks(rotation=45)
        
        # Save to file
        plt.savefig(os.path.join(output_dir, "claude_tokens_all_budgets_boxplot.png"), dpi=300, bbox_inches='tight')
        print(f"Saved box plot to claude_tokens_all_budgets_boxplot.png")
        plt.close()

if __name__ == "__main__":
    generate_token_histograms() 