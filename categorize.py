import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pandas as pd
import argparse
from prompts import categorize_prompt # Assuming prompts.py is in the same directory or PYTHONPATH
from tqdm.asyncio import tqdm_asyncio # For async progress bar

load_dotenv()
client = AsyncOpenAI()

async def get_category_value(description, caption, element, category_name, model_name):
    """
    Formats the prompt and calls the OpenAI API to get the category value.
    Now an async function. Returns (category_value, prompt_tokens, completion_tokens).
    """
    prompt_text = categorize_prompt(description, caption, element, category_name)
    prompt_tokens = 0
    completion_tokens = 0
    category_value_result = None
    
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            model=model_name,
            # temperature=0,
            # max_tokens=10 
        )
        response_content = chat_completion.choices[0].message.content
        if "TRUE" in response_content.upper():
            category_value_result = True
        elif "FALSE" in response_content.upper():
            category_value_result = False
        else:
            print(f"Warning: Unexpected API response for idx processing: {response_content}. Defaulting to None.")
            # category_value_result remains None

        if chat_completion.usage:
            prompt_tokens = chat_completion.usage.prompt_tokens
            completion_tokens = chat_completion.usage.completion_tokens
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # category_value_result remains None, tokens remain 0
    
    return category_value_result, prompt_tokens, completion_tokens

async def worker(name, queue, results_list, category_name, model_name, pbar):
    """
    Worker function to process rows from the queue.
    """
    while True:
        try:
            index, row_data = await queue.get()
            if row_data is None: # Sentinel value to stop the worker
                queue.task_done()
                break

            idx = row_data["idx"]
            description = row_data["description"]
            caption = row_data["caption"]
            element = row_data["element"]
            
            category_value, prompt_tokens, completion_tokens = await get_category_value(description, caption, element, category_name, model_name)
            
            results_list.append({
                "idx": idx, 
                category_name: category_value, 
                "prompt_tokens": prompt_tokens, 
                "completion_tokens": completion_tokens
            })
            queue.task_done()
            pbar.update(1)
        except asyncio.CancelledError:
            print(f"Worker {name} cancelled.")
            break
        except Exception as e:
            print(f"Error in worker {name}: {e}")
            # Ensure idx is defined before trying to use it for error logging
            current_idx = row_data.get("idx", "unknown") if 'row_data' in locals() and row_data is not None else "unknown"
            results_list.append({
                "idx": current_idx, 
                category_name: f"Error: {e}",
                "prompt_tokens": 0, 
                "completion_tokens": 0
            }) 
            queue.task_done()
            pbar.update(1)


async def main():
    parser = argparse.ArgumentParser(description="Categorize cartoon annotations.")
    parser.add_argument("--category", type=str, help="The name of the category to apply.", required=True)
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use for categorization.")
    parser.add_argument("--n_workers", type=int, default=10, help="Number of concurrent workers for API calls.")
    args = parser.parse_args()
    
    category_name = args.category
    model_name = args.model
    n_workers = args.n_workers

    try:
        annotations_df = pd.read_csv("comprehensive_annotations.csv")
    except FileNotFoundError:
        print("Error: comprehensive_annotations.csv not found.")
        return

    results_list = []
    queue = asyncio.Queue()

    total_rows = len(annotations_df)
    pbar = tqdm_asyncio(total=total_rows, desc=f"Categorizing for '{category_name}'")

    worker_tasks = []
    for i in range(n_workers):
        task = asyncio.create_task(worker(f"worker-{i+1}", queue, results_list, category_name, model_name, pbar))
        worker_tasks.append(task)

    for index, row in annotations_df.iterrows():
        await queue.put((index, row.to_dict()))

    for _ in range(n_workers):
        await queue.put((None,None))

    await queue.join()
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    pbar.close()

    output_df = pd.DataFrame(results_list)
    if not output_df.empty and 'idx' in output_df.columns:
        output_df = output_df.sort_values(by="idx").reset_index(drop=True)
    
    print("\nOutput DataFrame:")
    print(output_df[["idx", category_name]].to_string())

    if not output_df.empty:
        output_filename = f"categorized_output_{category_name.replace(' ', '_').lower()}.csv"
        try:
            output_df.to_csv(output_filename, index=False)
            print(f"\nSuccessfully saved output to {output_filename}")
        except Exception as e:
            print(f"\nError saving DataFrame to CSV: {e}")
        
        # Calculate and print total tokens
        if "prompt_tokens" in output_df.columns and "completion_tokens" in output_df.columns:
            total_prompt_tokens = output_df["prompt_tokens"].sum()
            total_completion_tokens = output_df["completion_tokens"].sum()
            overall_total_tokens = total_prompt_tokens + total_completion_tokens
            print(f"\nToken Usage Summary:")
            print(f"  Total Prompt Tokens: {total_prompt_tokens}")
            print(f"  Total Completion Tokens: {total_completion_tokens}")
            print(f"  Overall Total Tokens: {overall_total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())

