#!/usr/bin/env python3
import os
import sys
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
import argparse
import re
from explainer import ExplainerClient
from autograder import AutograderClient
from pathlib import Path
import json

# Constants
OUTPUT_DIR = "runs/vllm/summarized"
BATCH_SIZE = 5  # Number of concurrent API calls

async def extract_clean_answer(explainer, full_reasoning):
    """
    Use microsoft/phi-4 to extract just the final answer from the CoT reasoning.
    
    Args:
        explainer: ExplainerClient instance
        full_reasoning: The full chain-of-thought reasoning text
        
    Returns:
        Clean extracted explanation
    """
    # Create prompt to extract the answer
    prompt = f"""
You are a summarizer, extracting the final response of a model from a full message, containing a chain of thought reasoning AND the final answer.
The model you are summarizing is being benchmarked on a humor understanding task, and its final answer is an explanation of the joke in a given caption to a cartoon.
At the end of the message, the model will switch from its internal reasoning to the final answer. Your job is to exactly extract the final answer.
It is important that you repeat, verbatim, the other model's answer, starting from the end of the reasoning. Because there is no rule on when the model will switch to the final answer, your primary task is identifying where the reasoning ends and the final answer begins.
Return, verbatim, the other model's answer, starting from the end of the reasoning.

Other model's FULL (reasoning + answer) message:
{full_reasoning}
"""

    # Call the model
    # We're using the utils.MessageChain directly
    from utils import MessageChain
    message_chain = MessageChain(family=explainer.family)
    message_chain.add_user_message(prompt)
    formatted_messages = message_chain.format_messages()
    response = await explainer._make_api_call(formatted_messages)
    
    # Extract the response content
    clean_answer = response.get("content", "")
    
    # Additional clean-up if needed
    clean_answer = clean_answer.strip()
    
    # Optional: Remove any remaining tags or artifacts
    clean_answer = re.sub(r'^(The joke is |The humor is |The explanation is )', '', clean_answer)
    clean_answer = re.sub(r'^["\'](.*)["\']$', r'\1', clean_answer)  # Remove quotations if present
    
    return clean_answer

async def process_batch(explainer, autograder, batch_data):
    """Process a batch of rows concurrently"""
    clean_answer_tasks = []
    
    # Create a list to track which rows had valid explanations
    valid_row_indices = []
    
    # First, extract clean answers
    for index, row in batch_data.iterrows():
        full_reasoning = row.get('explanation', '')
        if not full_reasoning or len(full_reasoning) < 50:  # Skip empty or very short entries
            continue
        clean_answer_tasks.append(extract_clean_answer(explainer, full_reasoning))
        valid_row_indices.append(index)
    
    # Execute all clean answer tasks concurrently
    clean_answers = await asyncio.gather(*clean_answer_tasks)
    
    # Now, grade each clean answer
    grading_tasks = []
    for i, (index, clean_answer) in enumerate(zip(valid_row_indices, clean_answers)):
        description = batch_data.loc[index, 'description']
        caption = batch_data.loc[index, 'caption']
        element = batch_data.loc[index, 'element']
        grading_tasks.append(autograder.grade_explanation(description, caption, element, clean_answer))
    
    # Execute all grading tasks concurrently
    grading_results = await asyncio.gather(*grading_tasks)
    
    # Return both clean answers and grading results with their indices
    return list(zip(valid_row_indices, clean_answers, grading_results))

async def process_file(file_path):
    """Process a single CSV file"""
    print(f"Processing: {file_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the CSV
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    
    # Initialize clients
    explainer = ExplainerClient(model="microsoft/phi-4")
    autograder = AutograderClient(model="gpt-4o")
    print(f"Using {explainer.model} as summarizer, {autograder.model} as autograder")
    
    # Create new columns for clean answers and updated grading
    df['clean_explanation'] = "Not processed"
    # We'll replace existing autograder columns later
    
    # Process in batches to avoid overwhelming the API
    batch_indices = range(0, len(df), BATCH_SIZE)
    
    for i in tqdm_asyncio(batch_indices, desc="Processing batches"):
        batch = df.iloc[i:i+BATCH_SIZE]
        batch_results = await process_batch(explainer, autograder, batch)
        
        # Update dataframe with processed results
        for idx, clean_answer, grade_result in batch_results:
            df.at[idx, 'clean_explanation'] = clean_answer
            # Replace old judgments with new ones based on the clean explanation
            df.at[idx, 'autograder_judgment'] = grade_result['judgment']
            df.at[idx, 'autograder_reasoning'] = grade_result['reasoning']
            # Update usage statistics as JSON string
            df.at[idx, 'autograder_usage'] = json.dumps(grade_result['usage'])

        # Save intermediate results periodically
        if i % (BATCH_SIZE * 5) == 0 and i > 0:
            output_name = os.path.join(OUTPUT_DIR, f"interim_{Path(file_path).name}")
            df.to_csv(output_name, index=False)
            print(f"Saved interim results to {output_name}")
    
    # Save final results
    output_name = os.path.join(OUTPUT_DIR, f"summarized_regraded_{Path(file_path).name}")
    df.to_csv(output_name, index=False)
    print(f"Saved final results to {output_name}")
    
    return output_name

def initialize_clients():
    """Helper method to make sure we can initialize both clients"""
    try:
        explainer = ExplainerClient(model="microsoft/phi-4")
        autograder = AutograderClient(model="gpt-4o")
        print(f"Successfully initialized clients:")
        print(f"- Explainer: {explainer.model} (Family: {explainer.family})")
        print(f"- Autograder: {autograder.model} (Family: {autograder.family})")
        return True
    except Exception as e:
        print(f"Error initializing clients: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Summarize phi-4 reasoning traces and regrade with gpt-4o")
    parser.add_argument("--file", type=str, help="Specific file to process")
    parser.add_argument("--all", action="store_true", help="Process all phi-4 files in runs/vllm/")
    args = parser.parse_args()
    
    # Check if we can initialize the clients
    if not initialize_clients():
        print("Failed to initialize clients. Check your API keys and connections.")
        return
    
    if args.file:
        # Process a specific file
        await process_file(args.file)
    elif args.all:
        # Process all phi-4 files in runs/vllm/
        vllm_dir = "runs/vllm"
        files = [os.path.join(vllm_dir, f) for f in os.listdir(vllm_dir) 
                 if f.endswith('.csv') and ('phi4' in f.lower() or 'phi_4' in f.lower())]
        
        print(f"Found {len(files)} phi-4 files to process")
        for file in files:
            await process_file(file)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main()) 