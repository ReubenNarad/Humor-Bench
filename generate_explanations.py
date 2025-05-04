#!/usr/bin/env python3
import asyncio
import pandas as pd
import os
import json
from explainer import ExplainerClient
from datetime import datetime

async def generate_explanations(model_name, input_csv, limit=None, vllm_reasoning_effort=None):
    """Generate explanations only, without grading them, preserving thinking trace."""
    print(f"Generating explanations using model: {model_name}")
    
    # Load the input data
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)
    
    # Initialize the explainer
    explainer = ExplainerClient(
        model=model_name,
        vllm_reasoning_effort=vllm_reasoning_effort
    )
    
    # Prepare output dataframe
    results_df = df.copy()
    results_df['explanation'] = None
    results_df['has_thinking'] = False
    results_df['thinking_trace'] = None
    results_df['explainer_model'] = model_name
    results_df['explainer_usage'] = None
    
    # Process each row
    for idx, row in df.iterrows():
        print(f"Processing {idx+1}/{len(df)}: {row['caption']}")
        try:
            # Get explanation using explainer client
            description = row['description']
            caption = row['caption']
            
            result = await explainer.explain_cartoon(description, caption)
            
            # Store the results, including thinking trace if present
            results_df.at[idx, 'explanation'] = result['explanation']
            results_df.at[idx, 'has_thinking'] = result.get('has_thinking', False)
            if result.get('has_thinking', False):
                results_df.at[idx, 'thinking_trace'] = result.get('thinking', '')
            results_df.at[idx, 'explainer_usage'] = json.dumps(result['usage'])
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results_df.at[idx, 'explanation'] = f"ERROR: {str(e)}"
            results_df.at[idx, 'explainer_usage'] = "{}"
    
    # Create output directory if it doesn't exist
    os.makedirs("explanations", exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    output_file = f"explanations/explanations_{timestamp}_{safe_model_name}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"Explanations saved to {output_file}")
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate explanations without grading")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model to use for generating explanations")
    parser.add_argument("--input", type=str, default="comprehensive_annotations.csv",
                        help="Input CSV file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N rows")
    parser.add_argument("--vllm-reasoning-effort", type=int, default=None,
                        help="Reasoning effort level for VLLM models (integer)")
    
    args = parser.parse_args()
    
    # Run the explanation generator
    asyncio.run(generate_explanations(
        args.model,
        args.input,
        args.limit,
        args.vllm_reasoning_effort
    )) 