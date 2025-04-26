import gradio as gr
import pandas as pd
import os
import glob

# Create rubric directory if it doesn't exist
os.makedirs("rubric", exist_ok=True)

def load_csv_files():
    """Get list of all CSV files in the runs directory"""
    return sorted(glob.glob("runs/*.csv"))

def load_csv(file_path):
    """Load a CSV file and return as a pandas DataFrame"""
    df = pd.read_csv(file_path)
    
    # Add 'label' column if it doesn't exist
    if 'label' not in df.columns:
        df['label'] = ""
        
    return df, file_path

def update_label(df, file_path, index, label):
    """Update the label for a specific row in the DataFrame"""
    if df is not None and 0 <= index < len(df):
        df.at[index, 'label'] = label
        
        # Return the updated row for display
        return df.iloc[index].to_dict(), df, index
    
    return None, df, index

def save_csv(df, file_path):
    """Save the DataFrame to the rubric directory"""
    if df is not None:
        # Create the output filename based on the input filename
        input_filename = os.path.basename(file_path)
        output_path = os.path.join("rubric", "rubric.csv")
        
        # Save the DataFrame to CSV
        df.to_csv(output_path, index=False)
        return f"Saved to {output_path}"
    
    return "No data to save"

def view_row(df, index):
    """View a specific row in the DataFrame"""
    if df is not None and 0 <= index < len(df):
        return df.iloc[index].to_dict(), index
    return None, index

def next_row(df, index):
    """Move to the next row"""
    if df is not None and index < len(df) - 1:
        index += 1
        return df.iloc[index].to_dict(), index
    return None, index

def prev_row(df, index):
    """Move to the previous row"""
    if df is not None and index > 0:
        index -= 1
        return df.iloc[index].to_dict(), index
    return None, index

# Define the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# CSV Annotation Tool")
    
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
            
            # Navigation
            with gr.Row():
                prev_button = gr.Button("Previous Row")
                next_button = gr.Button("Next Row")
            
            # Labeling buttons
            with gr.Row():
                pass_button = gr.Button("PASS", variant="primary")
                fail_button = gr.Button("FAIL", variant="secondary")
            
            # Save button
            save_button = gr.Button("Save to rubric/rubric.csv", variant="success")
            save_result = gr.Textbox(label="Save Result")
        
        with gr.Column(scale=2):
            # Display current row
            row_display = gr.JSON(label="Current Row")
            
            # Display current index
            index_display = gr.Number(label="Current Row Index", precision=0)
    
    # Set up event handlers
    load_button.click(
        load_csv,
        inputs=[file_dropdown],
        outputs=[df_state, file_path_state]
    ).then(
        view_row,
        inputs=[df_state, current_index],
        outputs=[row_display, index_display]
    )
    
    prev_button.click(
        prev_row,
        inputs=[df_state, current_index],
        outputs=[row_display, current_index]
    ).then(
        lambda x: x,
        inputs=[current_index],
        outputs=[index_display]
    )
    
    next_button.click(
        next_row,
        inputs=[df_state, current_index],
        outputs=[row_display, current_index]
    ).then(
        lambda x: x,
        inputs=[current_index],
        outputs=[index_display]
    )
    
    pass_button.click(
        update_label,
        inputs=[df_state, file_path_state, current_index, gr.Textbox(value="PASS", visible=False)],
        outputs=[row_display, df_state, current_index]
    )
    
    fail_button.click(
        update_label,
        inputs=[df_state, file_path_state, current_index, gr.Textbox(value="FAIL", visible=False)],
        outputs=[row_display, df_state, current_index]
    )
    
    save_button.click(
        save_csv,
        inputs=[df_state, file_path_state],
        outputs=[save_result]
    )

if __name__ == "__main__":
    app.launch() 