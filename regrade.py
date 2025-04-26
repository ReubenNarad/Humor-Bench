# regrade.py
import os
import sys
import pandas as pd
import argparse
import asyncio
from tqdm import tqdm
import json
from datetime import datetime
import re

# Add parent directory to path so we can import the clients
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autograder import AutograderClient # Only need autograder
from main_benchmark import analyze_benchmark_results # Reuse analysis function

# --- Constants ---
BASE_OUTPUT_DIR = "runs"
MODEL_PRICES_PATH = "model_prices.json" # Needed for analysis


async def regrade_single_row(autograder, row):
    """Regrade a single row using the new autograder."""
    try:
        # Extract needed data from row
        description = row['description']
        caption = row['caption']
        element = row['element']
        explanation = row['explanation'] # Use existing explanation

        # Grade explanation using the NEW autograder client
        grade_result = await autograder.grade_explanation(
            description, caption, element, explanation
        )

        # Return results focused on the new grading
        return {
            'autograder_judgment': grade_result['judgment'],
            'autograder_reasoning': grade_result['reasoning'],
            'autograder_usage': grade_result['usage']
            # Keep track of match against original label if present for potential analysis
            # 'match_original_human': row.get('label') == grade_result['judgment'] if 'label' in row else None
        }
    except Exception as e:
        print(f"Error regrading row {row.name}: {str(e)}")
        return {
            'autograder_judgment': "ERROR",
            'autograder_reasoning': f"Processing error: {str(e)}",
            'autograder_usage': {}
            # 'match_original_human': None
        }

async def run_regrade(input_run_file, new_autograder_model, n_threads=10):
    """Regrades explanations in an existing run file with a new autograder."""
    print(f"--- Starting Regrade Process ---")
    print(f"Input Run File: {input_run_file}")
    print(f"New Autograder Model: {new_autograder_model}")

    # Load the input run CSV
    try:
        df = pd.read_csv(input_run_file)
        print(f"Loaded {len(df)} rows from {input_run_file}")
    except FileNotFoundError:
         print(f"Error: Input file not found at {input_run_file}")
         return None # Indicate failure

    # Check required columns for regrading
    required_input_cols = ['description', 'caption', 'element', 'explanation']
    missing_input_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_input_cols:
        print(f"Error: Missing required columns in input CSV for regrading: {missing_input_cols}")
        return None

    # Initialize the NEW autograder client
    try:
        autograder = AutograderClient(model=new_autograder_model)
        print(f"Initialized new autograder: {new_autograder_model} (Family: {autograder.family})")
    except Exception as e:
        print(f"Error initializing new autograder client: {e}")
        return None

    # Process rows concurrently
    results = []
    with tqdm(total=len(df), desc=f"Regrading with {new_autograder_model}") as pbar:
        for i in range(0, len(df), n_threads):
            batch = df.iloc[i:i+n_threads]
            tasks = [regrade_single_row(autograder, row) for _, row in batch.iterrows()]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            pbar.update(len(batch))

    # --- Combine results with original data ---
    # Create a copy to avoid modifying the original DataFrame in memory if loaded elsewhere
    regraded_df = df.copy()

    # Overwrite or add the autograder columns with the NEW results
    regraded_df['autograder_judgment'] = [r['autograder_judgment'] for r in results]
    regraded_df['autograder_reasoning'] = [r['autograder_reasoning'] for r in results]
    regraded_df['autograder_model'] = new_autograder_model # Store the new model name
    regraded_df['autograder_usage'] = [json.dumps(r['autograder_usage']) for r in results]

    # --- Save results ---
    # Create the new output directory structure: runs/{new_autograder_model}/
    safe_ag_model_name = new_autograder_model.replace('/', '_').replace('-', '_').replace('.', '_')
    output_subdir = os.path.join(BASE_OUTPUT_DIR, safe_ag_model_name)
    os.makedirs(output_subdir, exist_ok=True)

    # Construct the new filename
    original_filename = os.path.basename(input_run_file)
    # Remove original _ag part if exists, and add the new one
    base_name = re.sub(r'_ag-[^.]+', '', original_filename.replace('.csv', ''))
    new_filename = f"{base_name}_ag-{safe_ag_model_name}.csv"
    output_file = os.path.join(output_subdir, new_filename)

    regraded_df.to_csv(output_file, index=False)
    print(f"\nRegrade complete. Results saved to {output_file}")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regrade existing benchmark runs with a new autograder")
    parser.add_argument("--input-run", type=str, required=True,
                        help="Path to the original benchmark run CSV file (in runs/)")
    parser.add_argument("--autograder-model", type=str, required=True,
                        help="The NEW autograder model to use for regrading")
    parser.add_argument("--n-threads", type=int, default=10,
                        help="Number of rows to process concurrently (default: 10)")
    args = parser.parse_args()

    # Run the regrade process
    output_filename = asyncio.run(run_regrade(
        args.input_run,
        args.autograder_model,
        args.n_threads
    ))

    # Analyze the regraded results if successful
    if output_filename:
        # We can reuse the analysis function from main_benchmark
        # It calculates PASS rate and cost based on the autograder columns present
        analyze_benchmark_results(output_filename) 