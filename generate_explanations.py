import pandas as pd
import asyncio
import argparse
import os
import random
from datetime import datetime
from explainer import ExplainerClient
from tqdm import tqdm

async def generate_explanation(explainer, description, caption):
    """Generate one explanation using the given explainer client"""
    try:
        explanation_result = await explainer.explain_cartoon(description, caption)
        return {
            "explanation": explanation_result["explanation"],
            "explainer_model": explainer.model,
            "explainer_family": explainer.family
        }
    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        return {
            "explanation": f"ERROR: {str(e)}",
            "explainer_model": explainer.model,
            "explainer_family": explainer.family
        }

async def generate_explanations(input_csv, output_csv, models, limit=None, mix_strategy="random"):
    """Generate explanations using a mix of models"""
    # Load the input CSV
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    # Limit rows if specified
    if limit and limit > 0:
        df = df.head(limit)
        print(f"Limited to {limit} rows")
    
    # Create explainer clients for each model
    explainers = []
    for model in models:
        try:
            explainer = ExplainerClient(model=model)
            explainers.append(explainer)
            print(f"Initialized explainer for {model} (Family: {explainer.family})")
        except Exception as e:
            print(f"Error initializing explainer for {model}: {str(e)}")
    
    if not explainers:
        print("No valid explainers initialized. Exiting.")
        return None
    
    # Create results dataframe with original columns
    results_df = df.copy()
    results_df['explanation'] = ""
    results_df['explainer_model'] = ""
    results_df['explainer_family'] = ""
    results_df['label'] = ""  # For manual grading
    
    # Process each row with a selected model
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating explanations"):
        # Select model based on strategy
        if mix_strategy == "random":
            explainer = random.choice(explainers)
        elif mix_strategy == "round_robin":
            explainer = explainers[idx % len(explainers)]
        else:
            explainer = explainers[0]  # Default to first model
        
        # Extract needed data
        description = row['description']
        caption = row['caption']
        
        # Get explanation
        result = await generate_explanation(explainer, description, caption)
        
        # Store results
        results_df.at[idx, 'explanation'] = result["explanation"]
        results_df.at[idx, 'explainer_model'] = result["explainer_model"]
        results_df.at[idx, 'explainer_family'] = result["explainer_family"]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    
    return output_csv

async def main():
    parser = argparse.ArgumentParser(description="Generate cartoon explanations using multiple models")
    parser.add_argument("--input", type=str, default="comprehensive_annotations.csv", 
                        help="Input CSV file with cartoon descriptions and captions")
    parser.add_argument("--output", type=str, 
                        help="Output CSV file for results (default: timestamped file in runs/)")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit to first N rows (for testing)")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=["gpt-4o", "claude-3-7-sonnet-latest", "gemini-2.5-pro-exp-03-25", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"],
                        help="Models to use for explanations")
    parser.add_argument("--strategy", type=str, choices=["random", "round_robin"], default="random",
                        help="Strategy for selecting models")
    args = parser.parse_args()
    
    # Generate default output filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # models_str = "_".join([m.replace('-', '_').replace('.', '_') for m in args.models])[:30]
        args.output = f"rubric/rubric_exp/{timestamp}_explanations_mix.csv"
    
    print(f"Generating explanations using models: {args.models}")
    print(f"Using {args.strategy} selection strategy")
    
    await generate_explanations(
        input_csv=args.input,
        output_csv=args.output,
        models=args.models,
        limit=args.limit,
        mix_strategy=args.strategy
    )

if __name__ == "__main__":
    asyncio.run(main()) 