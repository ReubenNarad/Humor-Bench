import os
import sys
import pandas as pd
import asyncio
import argparse
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path so we can import the clients
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autograder import AutograderClient

async def evaluate_single_row(autograder, row):
    """Process a single row from the rubric dataframe"""
    try:
        # Extract needed data from row
        description = row['description']
        caption = row['caption']
        element = row['element']  # This is the anticipated point
        explanation = row['explanation']
        human_label = row['label']
        
        # Grade explanation using autograder client
        grade_result = await autograder.grade_explanation(
            description, caption, element, explanation
        )
        
        # Return combined results
        return {
            'autograder_judgment': grade_result['judgment'],
            'autograder_reasoning': grade_result['reasoning'],
            'match': human_label == grade_result['judgment'],  # For statistics only
            'usage': grade_result['usage']
        }
    except Exception as e:
        # Log error and return partial data
        print(f"Error processing row {row.name}: {str(e)}")
        return {
            'autograder_judgment': "ERROR",
            'autograder_reasoning': f"Processing error: {str(e)}",
            'match': False,
            'usage': {}
        }

async def evaluate_autograder(autograder_model, rubric_path="rubric/rubric.csv", n_threads=5):
    """Evaluate the autograder against human annotations"""
    print(f"Starting autograder evaluation with model: {autograder_model}")
    
    # Load the rubric CSV
    rubric_df = pd.read_csv(rubric_path)
    print(f"Loaded {len(rubric_df)} annotations from {rubric_path}")
    
    # Check that required columns exist
    required_cols = ['description', 'caption', 'element', 'explanation', 'label']
    missing_cols = [col for col in required_cols if col not in rubric_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in rubric: {missing_cols}")
        return None
        
    # Filter to rows that have human labels
    labeled_df = rubric_df.dropna(subset=['label'])
    if len(labeled_df) < len(rubric_df):
        print(f"Note: {len(rubric_df) - len(labeled_df)} rows were skipped because they have no human label")
    
    # Initialize autograder client
    autograder = AutograderClient(model=autograder_model)
    print(f"Initialized autograder with model: {autograder_model} (Family: {autograder.family})")
    
    # Process rows concurrently in batches
    results = []
    
    with tqdm(total=len(labeled_df), desc="Evaluating explanations") as pbar:
        for i in range(0, len(labeled_df), n_threads):
            batch = labeled_df.iloc[i:i+n_threads]
            
            # Create tasks for concurrent processing
            tasks = [evaluate_single_row(autograder, row) for _, row in batch.iterrows()]
            
            # Run batch concurrently and gather results
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Update progress bar
            pbar.update(len(batch))
    
    # Create results dataframe (copy original with new columns)
    results_df = labeled_df.copy()
    results_df['autograder_judgment'] = [r['autograder_judgment'] for r in results]
    results_df['autograder_reasoning'] = [r['autograder_reasoning'] for r in results]
    
    # For statistics calculation only (not saved to CSV)
    matches = [r['match'] for r in results]
    
    # Create output directory if needed
    output_dir = "autograder_runs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp and run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{autograder_model.replace('-', '_')}_{timestamp}"
    output_file = f"{output_dir}/{run_id}.csv"
    
    # Save to CSV (without the match column since we don't want stats in the CSV)
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Calculate and print statistics (but don't include in CSV)
    total = len(results_df)
    correct_matches = sum(matches)
    accuracy = correct_matches / total if total > 0 else 0
    
    # Print summary
    print(f"\n===== Autograder Evaluation Statistics =====")
    print(f"Run ID: {run_id}")
    print(f"Total examples: {total}")
    print(f"Correct judgments: {correct_matches}")
    print(f"Overall accuracy: {accuracy:.2%}")
    
    # Create detailed breakdown by model if that info is available
    if 'explainer_model' in results_df.columns:
        print("\nAccuracy breakdown by explanation model:")
        for model in results_df['explainer_model'].unique():
            # Get indices of rows for this model
            model_indices = results_df.index[results_df['explainer_model'] == model].tolist()
            # Get corresponding matches
            model_matches = [matches[results_df.index.get_loc(idx)] for idx in model_indices]
            
            model_total = len(model_indices)
            model_correct = sum(model_matches)
            model_accuracy = model_correct / model_total if model_total > 0 else 0
            print(f"  {model}: {model_accuracy:.2%} ({model_correct}/{model_total})")
    
    # Create confusion matrix
    if total > 0:
        print("\nConfusion Matrix:")
        print("              | Autograder PASS | Autograder FAIL |")
        print("------------- | -------------- | -------------- |")
        
        # Calculate values for confusion matrix
        true_pass = sum(1 for i, m in enumerate(matches) if m and results_df.iloc[i]['label'] == "PASS")
        false_fail = sum(1 for i, m in enumerate(matches) if not m and results_df.iloc[i]['label'] == "PASS")
        false_pass = sum(1 for i, m in enumerate(matches) if not m and results_df.iloc[i]['label'] == "FAIL")
        true_fail = sum(1 for i, m in enumerate(matches) if m and results_df.iloc[i]['label'] == "FAIL")
        
        # Print confusion matrix
        print(f"Human PASS    | {true_pass:14d} | {false_fail:14d} |")
        print(f"Human FAIL    | {false_pass:14d} | {true_fail:14d} |")
    
    return results_df, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate autograder against human annotations")
    parser.add_argument("--autograder-model", type=str, default="gpt-4o", 
                        help="The model to use for the autograder")
    parser.add_argument("--rubric", type=str, default="rubric/rubric.csv", 
                        help="Path to the rubric CSV file with human annotations")
    parser.add_argument("--n-threads", type=int, default=5, 
                        help="Number of rows to process concurrently")
    args = parser.parse_args()
    
    # Run the evaluation
    asyncio.run(evaluate_autograder(
        args.autograder_model,
        args.rubric,
        args.n_threads
    ))
