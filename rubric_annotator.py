import gradio as gr
import pandas as pd
import os
import glob
from datetime import datetime

# Create rubric directory if it doesn't exist
os.makedirs("rubric", exist_ok=True)

def load_csv_files():
    """Get list of all CSV files in the runs directory"""
    return sorted(glob.glob("runs/*.csv"))

def load_csv(file_path):
    """Load a CSV file and return as a pandas DataFrame"""
    if not file_path:
        return None, "No file selected"
    
    df = pd.read_csv(file_path)
    
    # Make sure we have an explanation column
    if 'explanation' not in df.columns:
        return None, "This file does not contain explanations. Please select a file with explanations."
    
    # Add 'label' column if it doesn't exist
    if 'label' not in df.columns:
        df['label'] = ""
        
    return df, file_path

def update_label(df, file_path, index, label):
    """Update the label for a specific row in the DataFrame"""
    if df is not None and 0 <= index < len(df):
        df.at[index, 'label'] = label
        
        # Return the updated row and dataframe
        row = df.iloc[index]
        return (
            row.get('description', ''),
            row.get('caption', ''),
            row.get('element', ''),
            row.get('explanation', ''),
            row.get('explainer_model', ''),
            row.get('label', ''),
            df,
            index
        )
    
    return "", "", "", "", "", "", df, index

def save_csv(df, file_path):
    """Save the DataFrame to the rubric directory"""
    if df is not None:
        # Create the output filename
        # date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("rubric", f"rubric_{timestamp}.csv")
        
        # Save the DataFrame to CSV
        df.to_csv(output_path, index=False)
        return f"Saved to {output_path}"
    
    return "No data to save"

def view_row(df, index):
    """View a specific row in the DataFrame"""
    if df is not None and 0 <= index < len(df):
        row = df.iloc[index]
        return (
            row.get('description', ''),
            row.get('caption', ''),
            row.get('element', ''),
            row.get('explanation', ''),
            row.get('explainer_model', ''),
            row.get('label', ''),
            index
        )
    return "", "", "", "", "", "", index

def next_row(df, index):
    """Move to the next row"""
    if df is not None and index < len(df) - 1:
        index += 1
        row = df.iloc[index]
        return (
            row.get('description', ''),
            row.get('caption', ''),
            row.get('element', ''),
            row.get('explanation', ''),
            row.get('explainer_model', ''),
            row.get('label', ''),
            index
        )
    return "", "", "", "", "", "", index

def prev_row(df, index):
    """Move to the previous row"""
    if df is not None and index > 0:
        index -= 1
        row = df.iloc[index]
        return (
            row.get('description', ''),
            row.get('caption', ''),
            row.get('element', ''),
            row.get('explanation', ''),
            row.get('explainer_model', ''),
            row.get('label', ''),
            index
        )
    return "", "", "", "", "", "", index

# Define the Gradio interface
with gr.Blocks(title="Cartoon Explanation Grader") as app:
    gr.Markdown("# Cartoon Explanation Rubric Maker")
    gr.Markdown("""
    This app allows you to annotate cartoon explanation files with PASS/FAIL labels.
    
    To generate explanations, use the separate `generate_explanations.py` script:
    ```
    python generate_explanations.py --models gpt-4o claude-3-sonnet-20240229 gemini-1.5-pro-latest --limit 10
    ```
    """)
    
    # State variables
    df_state = gr.State(None)
    file_path_state = gr.State("")
    current_index = gr.State(0)
    
    with gr.Row():
        with gr.Column(scale=1):
            # File selection
            file_dropdown = gr.Dropdown(
                label="Select CSV file from runs directory",
                choices=load_csv_files(),
                interactive=True
            )
            load_button = gr.Button("Load File")
            load_result = gr.Textbox(label="Load Result")
            
            # Navigation
            with gr.Row():
                prev_button = gr.Button("Previous")
                next_button = gr.Button("Next")
            
            # Labeling buttons
            with gr.Row():
                pass_button = gr.Button("PASS", variant="primary")
                fail_button = gr.Button("FAIL", variant="secondary")
            
            # Save button
            save_button = gr.Button("Save to rubric/rubric.csv", variant="success")
            save_result = gr.Textbox(label="Save Result")
            
            # Stats
            total_count = gr.Textbox(label="Total Items")
            labeled_count = gr.Textbox(label="Labeled Items")
            progress = gr.Textbox(label="Progress")
            
        with gr.Column(scale=2):
            # Display row content
            description_display = gr.Textbox(label="Cartoon Description", lines=3)
            caption_display = gr.Textbox(label="Caption", lines=1)
            element_display = gr.Textbox(label="Anticipated Element", lines=2)
            explanation_display = gr.Textbox(label="Explanation", lines=8)
            
            # Display model info and index
            with gr.Row():
                model_display = gr.Textbox(label="Model Used")
                current_label = gr.Textbox(label="Current Label")
                index_display = gr.Number(label="Current Row Index", precision=0)
    
    # Calculate and display statistics
    def update_stats(df):
        if df is None:
            return "0", "0", "0%"
        
        total = len(df)
        labeled = df['label'].notna().sum()
        percent = (labeled / total) * 100 if total > 0 else 0
        
        return str(total), str(labeled), f"{percent:.1f}%"
    
    # Set up event handlers for grading
    load_button.click(
        load_csv,
        inputs=[file_dropdown],
        outputs=[df_state, load_result]
    ).then(
        view_row,
        inputs=[df_state, current_index],
        outputs=[
            description_display,
            caption_display,
            element_display,
            explanation_display,
            model_display,
            current_label,
            index_display
        ]
    ).then(
        update_stats,
        inputs=[df_state],
        outputs=[total_count, labeled_count, progress]
    )
    
    prev_button.click(
        prev_row,
        inputs=[df_state, current_index],
        outputs=[
            description_display,
            caption_display,
            element_display,
            explanation_display,
            model_display,
            current_label,
            current_index
        ]
    ).then(
        lambda x: x,
        inputs=[current_index],
        outputs=[index_display]
    )
    
    next_button.click(
        next_row,
        inputs=[df_state, current_index],
        outputs=[
            description_display,
            caption_display,
            element_display,
            explanation_display,
            model_display,
            current_label,
            current_index
        ]
    ).then(
        lambda x: x,
        inputs=[current_index],
        outputs=[index_display]
    )
    
    pass_button.click(
        update_label,
        inputs=[df_state, file_path_state, current_index, gr.Textbox(value="PASS", visible=False)],
        outputs=[
            description_display,
            caption_display,
            element_display,
            explanation_display,
            model_display,
            current_label,
            df_state,
            current_index
        ]
    ).then(
        update_stats,
        inputs=[df_state],
        outputs=[total_count, labeled_count, progress]
    )
    
    fail_button.click(
        update_label,
        inputs=[df_state, file_path_state, current_index, gr.Textbox(value="FAIL", visible=False)],
        outputs=[
            description_display,
            caption_display,
            element_display,
            explanation_display,
            model_display,
            current_label,
            df_state,
            current_index
        ]
    ).then(
        update_stats,
        inputs=[df_state],
        outputs=[total_count, labeled_count, progress]
    )
    
    save_button.click(
        save_csv,
        inputs=[df_state, file_path_state],
        outputs=[save_result]
    )

if __name__ == "__main__":
    app.launch() 