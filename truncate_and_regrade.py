#!/usr/bin/env python3
import os
import sys
import pandas as pd
import json
import re
import argparse
import asyncio
from tqdm import tqdm
import datetime
from autograder import AutograderClient

# Try to import DeepSeek tokenizer, provide helpful error if not available
try:
    from deepseek_tokenizer import ds_token
except ImportError:
    print("Error: DeepSeek tokenizer not found. Please install it with:")
    print("pip install deepseek-tokenizer")
    sys.exit(1)

def truncate_explanation(explanation, k):
    """Tokenize explanation with DeepSeek tokenizer and keep only the last k tokens."""
    if not explanation:
        return ""
    
    try:
        # Tokenize the explanation
        token_ids = ds_token.encode(explanation)
        # Keep only the last k tokens
        if len(token_ids) > k:
            truncated_token_ids = token_ids[-k:]
            # Decode the truncated tokens
            truncated_explanation = ds_token.decode(truncated_token_ids)
            return truncated_explanation
        else:
            # If explanation has fewer tokens than k, return it unchanged
            return explanation
    except Exception as e:
        print(f"Error tokenizing explanation: {e}")
        return explanation  # Return original explanation on error

async def process_row(autograder, row, k):
    """Process a single row: truncate explanation and get new grading."""
    try:
        description = row['description']
        caption = row['caption']
        element = row['element']
        original_explanation = row['explanation']
        
        # Truncate the explanation
        truncated_explanation = truncate_explanation(original_explanation, k)
        
        # Grade the truncated explanation
        grade_result = await autograder.grade_explanation(
            description, caption, element, truncated_explanation
        )
        
        # Return both the original and new judgments and other useful data
        return {
            'description': description,
            'caption': caption, 
            'element': element,
            'original_explanation': original_explanation,
            'truncated_explanation': truncated_explanation,
            'original_judgment': row.get('autograder_judgment', 'N/A'),
            'original_reasoning': row.get('autograder_reasoning', ''),
            'new_judgment': grade_result['judgment'],
            'new_reasoning': grade_result['reasoning'],
            'autograder_model': autograder.model,
            'autograder_usage': grade_result['usage'],
            'original_tokens': len(ds_token.encode(original_explanation)),
            'truncated_tokens': len(ds_token.encode(truncated_explanation)),
        }
    except Exception as e:
        print(f"Error processing row: {e}")
        return {
            'description': row.get('description', ''),
            'caption': row.get('caption', ''),
            'element': row.get('element', ''),
            'original_explanation': row.get('explanation', ''),
            'truncated_explanation': 'ERROR',
            'original_judgment': row.get('autograder_judgment', 'N/A'),
            'original_reasoning': row.get('autograder_reasoning', ''),
            'new_judgment': 'ERROR',
            'new_reasoning': f'Processing error: {str(e)}',
            'autograder_model': autograder.model if autograder else 'ERROR',
            'autograder_usage': {},
            'original_tokens': -1,
            'truncated_tokens': -1,
        }

async def worker(name, queue, autograder, results, pbar, k):
    """Worker that processes items from the queue."""
    while True:
        item = await queue.get()
        if item is None:  # Sentinel value to exit
            queue.task_done()
            break
            
        row_idx, row = item
        try:
            result = await process_row(autograder, row, k)
            results[row_idx] = result
        except Exception as e:
            print(f"Worker {name} error: {e}")
        finally:
            pbar.update(1)
            queue.task_done()

async def enqueue_jobs(df, queue):
    """Helper function to put all jobs onto the queue."""
    for idx, row in df.iterrows():
        await queue.put((idx, row))

async def main(args):
    print(f"Loading run file: {args.input_file}")
    # Load the input CSV
    try:
        df = pd.read_csv(args.input_file)
        print(f"Loaded {len(df)} rows from {args.input_file}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Initialize the autograder
    try:
        autograder = AutograderClient(model=args.autograder_model)
        print(f"Initialized autograder with model: {autograder.model}")
    except Exception as e:
        print(f"Error initializing autograder: {e}")
        return
        
    # Prepare for parallel processing
    queue = asyncio.Queue()
    results = {}
    
    # Set up progress bar
    total_rows = min(args.limit, len(df)) if args.limit else len(df)
    
    # Limit rows if requested
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit} rows")
    
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        # Start workers
        workers = []
        for i in range(args.n_workers):
            task = asyncio.create_task(
                worker(f"Worker-{i+1}", queue, autograder, results, pbar, args.k)
            )
            workers.append(task)
        
        # Enqueue all jobs
        producer = asyncio.create_task(enqueue_jobs(df, queue))
        await producer
        
        # Wait for queue to be processed
        await queue.join()
        
        # Add sentinel values to stop workers
        for _ in range(args.n_workers):
            await queue.put(None)
        
        # Wait for all workers to finish
        await asyncio.gather(*workers)
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_index()  # Sort by original index
    
    # Generate output filename and ensure output directory exists
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "runs/truncated"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_file = f"{output_dir}/truncated_{args.k}_tokens_{timestamp}.csv"
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    original_pass_count = sum(results_df['original_judgment'] == 'PASS')
    new_pass_count = sum(results_df['new_judgment'] == 'PASS')
    
    print("\nSummary:")
    print(f"Total rows processed: {len(results_df)}")
    print(f"Original PASS rate: {original_pass_count / len(results_df):.2%}")
    print(f"New PASS rate: {new_pass_count / len(results_df):.2%}")
    print(f"Change: {(new_pass_count - original_pass_count) / len(results_df):.2%}")
    
    # Count how many changed from PASS to FAIL and vice versa
    pass_to_fail = sum((results_df['original_judgment'] == 'PASS') & (results_df['new_judgment'] == 'FAIL'))
    fail_to_pass = sum((results_df['original_judgment'] == 'FAIL') & (results_df['new_judgment'] == 'PASS'))
    print(f"PASS → FAIL: {pass_to_fail}")
    print(f"FAIL → PASS: {fail_to_pass}")
    
    # Output token statistics
    mean_original = results_df['original_tokens'].mean()
    mean_truncated = results_df['truncated_tokens'].mean()
    print(f"\nToken statistics:")
    print(f"Mean original tokens: {mean_original:.2f}")
    print(f"Mean truncated tokens: {mean_truncated:.2f}")
    print(f"Token reduction: {(mean_original - mean_truncated) / mean_original:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate explanations and regrade")
    parser.add_argument("--input-file", type=str, required=True, help="Input CSV file from a benchmark run")
    parser.add_argument("--autograder-model", type=str, default="gpt-4o", help="Autograder model to use (default: gpt-4o)")
    parser.add_argument("--k", type=int, default=500, help="Number of tokens to keep from the end (default: 500)")
    parser.add_argument("--n-workers", type=int, default=10, help="Number of concurrent worker tasks (default: 10)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N rows for testing (default: process all)")
    
    args = parser.parse_args()
    asyncio.run(main(args)) 