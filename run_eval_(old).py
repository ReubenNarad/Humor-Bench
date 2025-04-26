# This is where the main evaluation script will be implemented.
# Reads the annotation csv, and for each row, it will:
# 1. Call the explainer client to get the explanation of the caption
# 2. Call the autograder client to get the judgment of the explanation
# 3. Save the results to a new csv file that extends the original one with the new columns for explanation, autograder_judgment, and autograder_reasoning.

# Will take a command line arg for model and run name, will be run from a bash script

import os
import sys
import pandas as pd
import argparse
import asyncio
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path so we can import the clients
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explainer import ExplainerClient
from autograder import AutograderClient


async def process_row(explainer, autograder, row):
    """Process a single row from the annotations dataframe"""
    try:
        # Extract needed data from row
        description = row['description']
        caption = row['caption']
        element = row['element']  # This is the anticipated point
        
        # Get explanation using explainer client
        explanation_result = await explainer.explain_cartoon(description, caption)
        explanation = explanation_result['explanation']
        explainer_usage = explanation_result['usage']
        
        # Grade explanation using autograder client
        grade_result = await autograder.grade_explanation(
            description, caption, element, explanation
        )
        
        # Return combined results
        return {
            'explanation': explanation,
            'judgment': grade_result['judgment'],
            'reasoning': grade_result['reasoning'],
            'explainer_usage': explainer_usage,
            'autograder_usage': grade_result['usage']
        }
    except Exception as e:
        # Log error and return partial data
        print(f"Error processing row {row['idx']}: {str(e)}")
        return {
            'explanation': f"ERROR: {str(e)}",
            'judgment': "ERROR",
            'reasoning': f"Processing error: {str(e)}",
            'explainer_usage': {},
            'autograder_usage': {}
        }


async def run_evaluation(explainer_model, autograder_model, run_name, limit=None, n_threads=10):
    """Run the evaluation process with concurrent processing"""
    print(f"Starting evaluation with explainer: {explainer_model}, autograder: {autograder_model}, run name: {run_name}")
    
    # Load the annotations CSV
    df = pd.read_csv("comprehensive_annotations.csv")
    print(f"Loaded {len(df)} annotations")
    
    # Limit rows for testing if specified
    if limit:
        df = df.head(limit)
        print(f"Limited to first {limit} rows for testing")
    
    # Initialize clients with their respective models
    explainer = ExplainerClient(model=explainer_model)
    autograder = AutograderClient(model=autograder_model)
    print(f"Initialized explainer with model: {explainer_model} (Family: {explainer.family})")
    print(f"Initialized autograder with model: {autograder_model} (Family: {autograder.family})")
    
    # Process rows concurrently in batches
    results = []
    total_batches = (len(df) + n_threads - 1) // n_threads
    
    with tqdm(total=len(df), desc="Processing cartoons") as pbar:
        for i in range(0, len(df), n_threads):
            batch = df.iloc[i:i+n_threads]
            
            # Create tasks for concurrent processing
            tasks = [process_row(explainer, autograder, row) for _, row in batch.iterrows()]
            
            # Run batch concurrently and gather results
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Update progress bar
            pbar.update(len(batch))
    
    # Add results to dataframe
    df['explanation'] = [r['explanation'] for r in results]
    df['judgment'] = [r['judgment'] for r in results]
    df['reasoning'] = [r['reasoning'] for r in results]
    
    # Create usage columns as JSON strings
    df['explainer_usage'] = [json.dumps(r['explainer_usage']) for r in results]
    df['autograder_usage'] = [json.dumps(r['autograder_usage']) for r in results]
    
    # Create runs directory if it doesn't exist
    os.makedirs("runs", exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    output_file = f"runs/{timestamp}_{run_name}_ex-{explainer_model.replace('-', '_')}_ag-{autograder_model.replace('-', '_')}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cartoon explanation evaluation")
    parser.add_argument("--explainer-model", type=str, default="gpt-4o", 
                        help="The model to use for generating explanations")
    parser.add_argument("--autograder-model", type=str, default="gpt-4o", 
                        help="The model to use for grading explanations")
    parser.add_argument("--run-name", type=str, default="test_run", 
                        help="Name for this evaluation run")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit to first N rows (for testing)")
    parser.add_argument("--n-threads", type=int, default=10, 
                        help="Number of rows to process concurrently")
    args = parser.parse_args()
    
    # Run the evaluation with separate models
    asyncio.run(run_evaluation(
        args.explainer_model, 
        args.autograder_model, 
        args.run_name, 
        args.limit, 
        args.n_threads
    ))