# main_benchmark.py
import os
import sys
import pandas as pd
import argparse
import asyncio
from asyncio import Queue # Import Queue
from tqdm import tqdm
import json
from datetime import datetime
from explainer import ExplainerClient
from autograder import AutograderClient
from utils import MessageChain
import traceback

# --- Constants ---
MODEL_PRICES_PATH = "model_data/model_prices.json"
DEFAULT_INPUT_CSV = "comprehensive_annotations.csv"
OUTPUT_DIR = "runs"

async def process_row(explainer, autograder, row_tuple):
    """Process a single row tuple (index, row_data): get explanation, then grade it."""
    index, row = row_tuple # Unpack the tuple
    try:
        # Extract needed data from row
        description = row['description']
        caption = row['caption']
        element = row['element']  # This is the anticipated point
        
        # Add a small delay for XAI models to avoid rate limiting
        if explainer.family == 'xai':
            await asyncio.sleep(1.0)  # 1 second delay for XAI API
        
        # 1. Get explanation using explainer client
        explanation_result = await explainer.explain_cartoon(description, caption)
        explanation = explanation_result['explanation']
        explainer_usage = explanation_result['usage']
        
        # 2. Grade explanation using autograder client
        grade_result = await autograder.grade_explanation(
            description, caption, element, explanation
        )
        
        # Return combined results along with the original index
        return index, {
            'explanation': explanation,
            'autograder_judgment': grade_result['judgment'],
            'autograder_reasoning': grade_result['reasoning'],
            'explainer_model': explainer.model, # Add explainer model info
            'explainer_usage': explainer_usage,
            'autograder_model': autograder.model, # Add autograder model info
            'autograder_usage': grade_result['usage']
        }
    except Exception as e:
        # Log error and return partial data with the index
        print(f"Error processing row index {index}: {str(e)}")
        return index, {
            'explanation': f"ERROR: {str(e)}",
            'autograder_judgment': "ERROR",
            'autograder_reasoning': f"Processing error: {str(e)}",
            'explainer_model': explainer.model if explainer else "ERROR",
            'autograder_model': autograder.model if autograder else "ERROR",
            # Ensure usage is always a dict-like structure, even on error
            'explainer_usage': {},
            'autograder_usage': {}
        }

def calculate_cost(row, model_prices):
    """Calculate the cost for a single row based on usage and prices."""
    cost = 0.0
    
    # Explainer cost
    explainer_model = row.get('explainer_model')
    # Usage might already be a dict if coming directly from results,
    # or a string if loaded from CSV. Handle both.
    explainer_usage = row.get('explainer_usage', {})
    if isinstance(explainer_usage, str):
        try:
            explainer_usage = json.loads(explainer_usage)
        except json.JSONDecodeError:
            explainer_usage = {}

    if explainer_model in model_prices and isinstance(explainer_usage, dict):
        try:
            tokens_in = explainer_usage.get('tokens_in', 0)
            tokens_out = explainer_usage.get('tokens_out', 0)
            prices = model_prices[explainer_model]
            cost += (tokens_in / 1_000_000) * prices.get("input $/M", 0)
            cost += (tokens_out / 1_000_000) * prices.get("output $/M", 0)
        except (TypeError, KeyError) as e:
            # Use .get('name', 'N/A') for safer access in f-string if row might not be a Series
            row_identifier = row.get('name', 'N/A') if isinstance(row, pd.Series) else row.get('index', 'N/A')
            print(f"Warning: Could not parse/calculate explainer cost for row {row_identifier}: {e}")
            
    # Autograder cost
    autograder_model = row.get('autograder_model')
    autograder_usage = row.get('autograder_usage', {})
    if isinstance(autograder_usage, str):
         try:
            autograder_usage = json.loads(autograder_usage)
         except json.JSONDecodeError:
            autograder_usage = {}

    if autograder_model in model_prices and isinstance(autograder_usage, dict):
         try:
            tokens_in = autograder_usage.get('tokens_in', 0)
            tokens_out = autograder_usage.get('tokens_out', 0)
            prices = model_prices[autograder_model]
            cost += (tokens_in / 1_000_000) * prices.get("input $/M", 0)
            cost += (tokens_out / 1_000_000) * prices.get("output $/M", 0)
         except (TypeError, KeyError) as e:
            row_identifier = row.get('name', 'N/A') if isinstance(row, pd.Series) else row.get('index', 'N/A')
            print(f"Warning: Could not parse/calculate autograder cost for row {row_identifier}: {e}")

    return cost

def analyze_benchmark_results(output_file):
    """Load results CSV and print analysis."""
    print(f"\n--- Analyzing Benchmark Results: {output_file} ---")
    try:
        df = pd.read_csv(output_file)
        
        # Load prices
        model_prices = {}
        if os.path.exists(MODEL_PRICES_PATH):
            with open(MODEL_PRICES_PATH, 'r') as f:
                model_prices = json.load(f)
            print(f"Loaded model prices from {MODEL_PRICES_PATH}")
        else:
            print(f"Warning: {MODEL_PRICES_PATH} not found. Cannot calculate costs.")

        # --- Calculate PASS Rate ---
        valid_judgments = df[df['autograder_judgment'].isin(['PASS', 'FAIL'])]
        total_valid = len(valid_judgments)
        passes = sum(valid_judgments['autograder_judgment'] == 'PASS')
        pass_rate = passes / total_valid if total_valid > 0 else 0
        
        print(f"\nAutograder Performance:")
        print(f"  Total Processed: {len(df)}")
        print(f"  Valid Judgments: {total_valid}")
        print(f"  PASS Count:      {passes}")
        print(f"  FAIL Count:      {total_valid - passes}")
        print(f"  PASS Rate:       {pass_rate:.2%}")

        # --- Calculate Cost ---
        if model_prices:
            # Ensure usage columns are loaded correctly if they were saved as strings
            for col in ['explainer_usage', 'autograder_usage']:
                 if col in df.columns and isinstance(df[col].iloc[0], str):
                      df[col] = df[col].apply(json.loads)

            df['estimated_cost'] = df.apply(lambda row: calculate_cost(row, model_prices), axis=1)
            total_cost = df['estimated_cost'].sum()
            print(f"\nEstimated Cost:")
            print(f"  Total Cost for Run: ${total_cost:.4f}")
            # Use len(df) for avg cost calculation as errors still incur cost potentially
            if len(df) > 0:
                 print(f"  Avg Cost per Item:  ${total_cost / len(df):.6f}")
        
        print("-" * (len(output_file) + 22)) # Footer length based on header

    except FileNotFoundError:
        print(f"Error: Results file not found at {output_file}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()


async def worker(name, queue, explainer, autograder, results_list, pbar):
    """Worker task that processes items from the queue."""
    while True:
        row_tuple = await queue.get()
        if row_tuple is None: # Sentinel value received
            # print(f"{name}: Received sentinel, exiting.") # Optional debug print
            queue.task_done() # Signal completion for the sentinel
            break
        # print(f"{name}: Processing row {row_tuple[0]}") # Optional debug print
        try:
            result_index, result_data = await process_row(explainer, autograder, row_tuple)
            results_list.append((result_index, result_data))
        except Exception as e:
            # This catches errors within process_row itself, though process_row already handles internal errors
            print(f"{name}: Error processing item {row_tuple[0]}: {e}")
            # Store an error marker with the index
            results_list.append((row_tuple[0], {'error': str(e)}))
        finally:
            # print(f"{name}: Finished row {row_tuple[0]}") # Optional debug print
            pbar.update(1) # Update progress bar for each item finished
            queue.task_done() # Signal completion for the actual item


async def run_benchmark(explainer_model, autograder_model, run_name, input_csv=DEFAULT_INPUT_CSV, limit=None, n_workers=10, thinking_budget=None, reasoning_effort=None): # Add reasoning_effort parameter
    """Run the full explanation and grading benchmark process using a worker queue."""
    print(f"--- Starting Benchmark Run: {run_name} ---")
    print(f"Explainer: {explainer_model}, Autograder: {autograder_model}")
    print(f"Input: {input_csv}, Limit: {limit}, Workers: {n_workers}")
    if thinking_budget is not None:
        print(f"Thinking Budget: {thinking_budget} (for Claude, Gemini, or Alibaba models)")
    if reasoning_effort is not None: # Log reasoning effort if provided
        print(f"OpenAI Reasoning Effort: {reasoning_effort}")

    # Load the input annotations CSV
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} base annotations from {input_csv}")
    except FileNotFoundError:
         print(f"Error: Input file not found at {input_csv}")
         return None, None # Indicate failure
         
    # Check required columns in input
    required_input_cols = ['description', 'caption', 'element']
    missing_input_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_input_cols:
        print(f"Error: Missing required columns in input CSV: {missing_input_cols}")
        return None, None
         
    # Limit rows for testing if specified
    if limit and limit > 0:
        df_limited = df.head(limit).copy() # Work on a copy
        print(f"Limited to first {limit} rows for processing")
    else:
        df_limited = df.copy() # Work on a copy

    total_rows_to_process = len(df_limited)

    # Initialize clients
    try:
        # Explicitly pass API key if XAI family
        xai_api_key = None
        if explainer_model and 'grok' in explainer_model.lower():
             xai_api_key = os.environ.get("XAI_API_KEY")
             if not xai_api_key:
                  print("Warning: XAI_API_KEY environment variable not found for grok model.")

        # Pass thinking_budget and reasoning_effort to ExplainerClient
        explainer = ExplainerClient(
            model=explainer_model, 
            api_key=xai_api_key, # Pass the key explicitly
            thinking_budget=thinking_budget,
            reasoning_effort=reasoning_effort # Pass effort here
        )
        # Autograder does not get reasoning_effort or thinking_budget
        autograder = AutograderClient(model=autograder_model) 

        # Add validation check (Now mostly handled within client __init__)
        if thinking_budget is not None:
            is_claude_explainer = explainer.family == MessageChain.CLAUDE
            is_alibaba_explainer = explainer.family == MessageChain.ALIBABA
            is_gemini_explainer = explainer.family == MessageChain.GEMINI
            # No need to check autograder family for thinking_budget now
            
            if is_alibaba_explainer:
                print(f"Using thinking budget for Alibaba explainer model: {thinking_budget} tokens")
            elif is_gemini_explainer:
                print(f"Using thinking budget for Gemini explainer model: {thinking_budget} tokens")
            # The warning for Claude explainer without budget, or other explainer families with budget
            # is handled within ExplainerClient.__init__
                
        # Check for reasoning effort with non-OpenAI explainer
        if reasoning_effort is not None and explainer.family != MessageChain.OPENAI:
             # This case is also handled by the warning within the client init now
             pass


        print(f"Initialized explainer: {explainer_model} (Family: {explainer.family})")
        print(f"Initialized autograder: {autograder_model} (Family: {autograder.family})")
    except Exception as e:
        print(f"Error initializing clients: {e}")
        return None, None

    # --- Set up Queue and Workers ---
    queue = Queue()
    results_list = [] # Store results as (index, data) tuples
    worker_tasks = []

    # Use tqdm for progress tracking
    with tqdm(total=total_rows_to_process, desc="Processing cartoons") as pbar:
        # Start worker tasks
        for i in range(n_workers):
            task = asyncio.create_task(worker(f'Worker-{i+1}', queue, explainer, autograder, results_list, pbar))
            worker_tasks.append(task)

        # Enqueue jobs (put tuples of (index, row_data) onto the queue)
        producer_task = asyncio.create_task(enqueue_jobs(df_limited, queue))

        # Wait for the producer to finish enqueuing
        await producer_task

        # Wait for all items in the queue to be processed
        await queue.join()
        # print("Queue joined. All items processed.") # Optional debug print

        # Send sentinel values to stop workers
        for _ in range(n_workers):
            await queue.put(None)

        # Wait for all worker tasks to finish cleanly
        await asyncio.gather(*worker_tasks)
        # print("Worker tasks finished.") # Optional debug print

    # --- Combine results ---
    # Sort results by the original index to maintain order
    results_list.sort(key=lambda x: x[0])

    # Create a dictionary from the sorted results for easier lookup
    results_dict = {index: data for index, data in results_list}

    # Map results back to the limited dataframe
    df_limited['explanation'] = df_limited.index.map(lambda i: results_dict.get(i, {}).get('explanation', 'ERROR: Missing result'))
    df_limited['autograder_judgment'] = df_limited.index.map(lambda i: results_dict.get(i, {}).get('autograder_judgment', 'ERROR'))
    df_limited['autograder_reasoning'] = df_limited.index.map(lambda i: results_dict.get(i, {}).get('autograder_reasoning', 'ERROR'))
    df_limited['explainer_model'] = df_limited.index.map(lambda i: results_dict.get(i, {}).get('explainer_model', 'ERROR'))
    # Ensure usage is stored as JSON string in the final DataFrame
    df_limited['explainer_usage'] = df_limited.index.map(lambda i: json.dumps(results_dict.get(i, {}).get('explainer_usage', {})))
    df_limited['autograder_model'] = df_limited.index.map(lambda i: results_dict.get(i, {}).get('autograder_model', 'ERROR'))
    df_limited['autograder_usage'] = df_limited.index.map(lambda i: json.dumps(results_dict.get(i, {}).get('autograder_usage', {})))

    # --- Save results ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize model names for filename
    safe_ex_model = explainer_model.replace('/', '_').replace('-', '_').replace('.', '_')
    safe_ag_model = autograder_model.replace('/', '_').replace('-', '_').replace('.', '_')
    
    output_file = f"{OUTPUT_DIR}/{timestamp}_{run_name}_exp-{safe_ex_model}_ag-{safe_ag_model}.csv"
    df_limited.to_csv(output_file, index=False)
    print(f"\nBenchmark complete. Results saved to {output_file}")
    
    return df_limited, output_file # Return dataframe and output path

async def enqueue_jobs(df, queue):
    """Helper function to put all jobs onto the queue."""
    for index, row in df.iterrows():
        await queue.put((index, row))
    # print("All jobs enqueued.") # Optional debug print


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main benchmark: Explain and Autograde Cartoons")
    parser.add_argument("--explainer-model", type=str, required=False, # Make not required if analyzing only
                        help="Model for generating explanations")
    parser.add_argument("--autograder-model", type=str, required=False, # Make not required if analyzing only
                        help="Model for grading explanations")
    parser.add_argument("--run-name", type=str, default="benchmark_run",
                        help="Name for this benchmark run (used in filename)")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_CSV,
                        help=f"Input CSV file (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N rows (for testing)")
    parser.add_argument("--n-workers", type=int, default=10,
                        help="Number of concurrent worker tasks (default: 10)")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Max tokens for Claude extended thinking or Alibaba reasoning (optional)")
    # --- Add argument for reasoning effort ---
    parser.add_argument("--reasoning-effort", type=str, default=None, choices=["low", "medium", "high"],
                        help="Reasoning effort for OpenAI 'o' models (optional, only for Explainer)")
    # --- Add argument for analyzing an existing file ---
    parser.add_argument("--analyze-only", type=str, default=None,
                        help="Path to an existing benchmark CSV file to analyze (skips running the benchmark)")

    args = parser.parse_args()

    # --- Check if we are only analyzing ---
    if args.analyze_only:
        if not os.path.exists(args.analyze_only):
            print(f"Error: File not found for analysis: {args.analyze_only}")
        else:
            analyze_benchmark_results(args.analyze_only)
    # --- Otherwise, run the benchmark (original logic) ---
    else:
        # Check if required args for benchmark run are provided
        if not args.explainer_model or not args.autograder_model:
             parser.error("--explainer-model and --autograder-model are required unless --analyze-only is used.")

        # Run the benchmark using asyncio.run, passing the new arguments
        df_results, output_filename = asyncio.run(run_benchmark(
            args.explainer_model,
            args.autograder_model,
            args.run_name,
            args.input,
            args.limit,
            args.n_workers,
            args.thinking_budget, # Pass the budget
            args.reasoning_effort # Pass the effort
        ))

        # Analyze the results if the run was successful
        if output_filename:
            analyze_benchmark_results(output_filename)