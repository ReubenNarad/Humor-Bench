import json
import os
import shutil
from datetime import datetime

# Define the HTML template with placeholders for data and Plotly.js
# NOTE: All literal curly braces for JS/CSS must be doubled ({{ or }})
# Placeholders for Python's .format() remain single ({timestamp}, {autograder_model}, {json_data})
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: sans-serif; 
            margin: 0; 
            padding: 0; 
            color: #333; 
        }}
        .header {{ 
            background-color: #f8f8f8; 
            border-bottom: 1px solid #ddd; 
            padding: 15px 20px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        }}
        .title-container {{ 
            flex: 1; 
        }}
        h1 {{ 
            margin: 0; 
            font-size: 24px; 
            color: #333;
        }}
        .metadata {{ 
            font-size: 12px; 
            color: #666; 
            margin-top: 5px; 
        }}
        .controls {{ 
            display: flex; 
            align-items: center; 
            gap: 15px;
        }}
        .control-group {{ 
            display: flex; 
            align-items: center; 
        }}
        label {{ 
            margin-right: 8px; 
            font-weight: 600; 
            font-size: 14px; 
        }}
        select {{ 
            padding: 6px 10px; 
            border: 1px solid #ccc; 
            border-radius: 4px; 
            background-color: white; 
            min-width: 180px; 
        }}
        .checkbox-wrapper {{ 
            display: flex; 
            align-items: center; 
            margin-left: 5px; 
        }}
        .checkbox-wrapper input {{ 
            margin-right: 5px; 
        }}
        .content {{ 
            padding: 20px; 
        }}
        #plot {{ 
            height: calc(100vh - 110px); 
            width: 100%; 
            min-height: 500px; 
        }}
        
        /* Dropdown Style */
        .dropdown {{
            position: relative;
            display: inline-block;
        }}
        .dropdown-btn {{
            padding: 6px 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: white;
            cursor: pointer;
            min-width: 180px;
            text-align: left;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .dropdown-btn:after {{
            content: "▼";
            font-size: 10px;
            margin-left: 5px;
        }}
        .dropdown-content {{
            display: none;
            position: absolute;
            background-color: white;
            min-width: 240px;
            max-height: 300px;
            overflow-y: auto;
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            z-index: 1;
            border-radius: 4px;
            padding: 5px 0;
        }}
        .dropdown-content.show {{
            display: block;
        }}
        .dropdown-item {{
            display: flex;
            padding: 5px 10px;
            align-items: center;
        }}
        .dropdown-item:hover {{
            background-color: #f1f1f1;
        }}
        .dropdown-actions {{
            display: flex;
            justify-content: space-between;
            padding: 8px 10px;
            border-top: 1px solid #eee;
            margin-top: 5px;
        }}
        .dropdown-action {{
            font-size: 12px;
            color: #0066cc;
            cursor: pointer;
            background: none;
            border: none;
            padding: 2px 5px;
        }}
        .dropdown-action:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>

<div class="header">
    <div class="title-container">
        <h1>Interactive Benchmark Report</h1>
        <div class="metadata">
            Generated: {timestamp} | Autograder: {autograder_model}
        </div>
    </div>
    <div class="controls">
        <div class="control-group">
            <label for="y-axis-select">Y-Axis:</label>
            <select id="y-axis-select">
                <option value="pass_rate_per_row" selected>Pass Rate (Per Row)</option>
                <option value="pass_rate_per_caption_all">Pass Rate (Per Caption - All)</option>
                <option value="pass_rate_per_caption_some">Pass Rate (Per Caption - Some)</option>
            </select>
        </div>
        <div class="control-group">
            <label for="x-axis-select">X-Axis:</label>
            <select id="x-axis-select">
                <option value="total_cost" selected>Total Cost ($)</option>
                <option value="gpqa_score">GPQA Score (%)</option>
                <option value="arc_agi_score">ARC-AGI Score (%)</option>
                <option value="lmarena_elo_score">LM Arena ELO Score</option>
            </select>
            <div class="checkbox-wrapper">
                <input type="checkbox" id="log-scale-x">
                <label for="log-scale-x" style="font-weight: normal;">Log Scale</label>
            </div>
        </div>
        <div class="control-group">
            <div class="dropdown" id="model-dropdown">
                <button class="dropdown-btn" id="model-dropdown-btn">All Models ▼</button>
                <div class="dropdown-content" id="model-dropdown-content">
                    <!-- Will be populated dynamically -->
                    <div class="dropdown-actions">
                        <button class="dropdown-action" id="select-all-models">Select All</button>
                        <button class="dropdown-action" id="deselect-all-models">Deselect All</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="content">
    <div id="plot"></div>
</div>

<script>
    // Embedded data - injected during HTML generation
    const reportData = {json_data};

    // Wait for the DOM to be fully loaded
    document.addEventListener('DOMContentLoaded', function() {{
        const plotDiv = document.getElementById('plot');
        const yAxisSelect = document.getElementById('y-axis-select');
        const xAxisSelect = document.getElementById('x-axis-select');
        const logScaleXCheckbox = document.getElementById('log-scale-x');
        const modelDropdownBtn = document.getElementById('model-dropdown-btn');
        const modelDropdownContent = document.getElementById('model-dropdown-content');
        const selectAllButton = document.getElementById('select-all-models');
        const deselectAllButton = document.getElementById('deselect-all-models');

        let allModelNames = [];
        let selectedModels = [];

        // Toggle the dropdown visibility
        modelDropdownBtn.addEventListener('click', function() {{
            modelDropdownContent.classList.toggle('show');
        }});

        // Close dropdown when clicking outside
        window.addEventListener('click', function(event) {{
            if (!event.target.matches('.dropdown-btn') && !event.target.closest('.dropdown-content')) {{
                modelDropdownContent.classList.remove('show');
            }}
        }});

        // Function to initialize data and setup
        function initializeData() {{
            if (!reportData || reportData.length === 0) {{
                plotDiv.innerHTML = '<p>No data available to plot.</p>';
                return;
            }}
            
            // Extract unique model names and sort them
            allModelNames = [...new Set(reportData.map(d => d.display_name))].sort();
            selectedModels = [...allModelNames]; // Start with all models selected
            
            // Check if GPQA data exists
            const hasGpqaData = reportData.some(d => d.gpqa_score !== undefined && d.gpqa_score !== null);
            
            if (!hasGpqaData) {{
                // Disable GPQA option if no data at all
                const gpqaOption = xAxisSelect.querySelector('option[value="gpqa_score"]');
                if (gpqaOption) {{
                    gpqaOption.disabled = true;
                    gpqaOption.text += " (No Data)";
                }}
            }} else {{
                // Count how many models have GPQA data
                const modelsWithGpqa = reportData.filter(d => d.gpqa_score !== undefined && d.gpqa_score !== null);
                const gpqaOption = xAxisSelect.querySelector('option[value="gpqa_score"]');
                
                if (modelsWithGpqa.length < reportData.length) {{
                    // Some but not all models have GPQA data
                    gpqaOption.text += ` (${{modelsWithGpqa.length}}/${{reportData.length}} models)`;
                }}
            }}
            
            // Populate UI elements
            populateModelDropdown();
            updateModelButtonText();
            updatePlot(); // Initial plot rendering
        }}

        // Function to populate model dropdown
        function populateModelDropdown() {{
            // Insert checkboxes before the dropdown actions
            const dropdownActions = modelDropdownContent.querySelector('.dropdown-actions');
            
            allModelNames.forEach(name => {{
                const item = document.createElement('div');
                item.className = 'dropdown-item';
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.value = name;
                checkbox.id = `model-${{name.replace(/\\s+/g, '-')}}`;
                checkbox.checked = true; // Default to checked
                
                checkbox.addEventListener('change', function() {{
                    if (this.checked) {{
                        selectedModels.push(name);
                    }} else {{
                        selectedModels = selectedModels.filter(m => m !== name);
                    }}
                    updateModelButtonText();
                    updatePlot();
                }});
                
                const label = document.createElement('label');
                label.htmlFor = checkbox.id;
                label.textContent = name;
                label.style.fontWeight = 'normal';
                label.style.marginLeft = '5px';
                
                item.appendChild(checkbox);
                item.appendChild(label);
                
                // Insert before the dropdown actions
                modelDropdownContent.insertBefore(item, dropdownActions);
            }});
        }}

        // Function to update the dropdown button text
        function updateModelButtonText() {{
            if (selectedModels.length === 0) {{
                modelDropdownBtn.textContent = "No Models Selected ▼";
            }} else if (selectedModels.length === allModelNames.length) {{
                modelDropdownBtn.textContent = "All Models ▼";
            }} else {{
                modelDropdownBtn.textContent = `${{selectedModels.length}} Models Selected ▼`;
            }}
        }}

        // Function to format values based on axis type
        function formatAxisValue(value, axisType) {{
            if (axisType === 'gpqa_score' || axisType === 'arc_agi_score') {{
                // Format as percentage
                return value * 100;
            }}
            // LM Arena ELO scores don't need transformation
            return value;
        }}

        // Function to get axis formatting settings
        function getAxisSettings(axisType, useLogScale = false) {{
            const settings = {{
                type: useLogScale ? 'log' : 'linear',
                autorange: true,
                tickprefix: '',
                ticksuffix: ''
            }};
            
            // Settings based on axis type
            if (axisType === 'total_cost') {{
                settings.tickprefix = '$';
            }} else if (axisType === 'gpqa_score' || axisType === 'arc_agi_score') {{
                settings.ticksuffix = '%';
            }} else if (axisType === 'lmarena_elo_score') {{
                settings.ticksuffix = ''; // No suffix for ELO scores
            }}
            
            return settings;
        }}

        // Function to update the plot
        function updatePlot() {{
            if (!reportData || reportData.length === 0) {{
                 plotDiv.innerHTML = '<p>No data available to plot.</p>';
                 return;
             }}

            const selectedYAxis = yAxisSelect.value;
            const selectedXAxis = xAxisSelect.value;
            const useLogScaleX = logScaleXCheckbox.checked;

            // Filter data based on selected models
            // Plus filter for only models that have data for the selected axis
            const filteredData = reportData.filter(d => 
                selectedModels.includes(d.display_name) && 
                // Check that the model has the selected data
                d[selectedXAxis] !== undefined && 
                d[selectedXAxis] !== null && 
                d[selectedYAxis] !== undefined && 
                d[selectedYAxis] !== null
            );

            if (filteredData.length === 0) {{
                 plotDiv.innerHTML = '<p>No data available for the selected combination of models and axes.<br>Some models may not have data for GPQA Score.</p>';
                 return;
             }}

            const trace = {{
                x: filteredData.map(d => formatAxisValue(d[selectedXAxis], selectedXAxis)),
                y: filteredData.map(d => d[selectedYAxis] * 100), // Convert to percentage
                text: filteredData.map(d => d.display_name), // Text for hover/annotations
                mode: 'markers+text',
                type: 'scatter',
                marker: {{
                    size: 12,
                    color: filteredData.map(d => d.plot_color),
                    line: {{
                        color: 'black',
                        width: 1
                    }}
                }},
                textposition: 'top center',
                textfont: {{
                    size: 11
                }},
                hoverinfo: 'text+x+y',
                hovertemplate: '<b>%{{text}}</b><br>' +
                               `${{xAxisSelect.options[xAxisSelect.selectedIndex].text}}: %{{x:.2f}}<br>` + 
                               `${{yAxisSelect.options[yAxisSelect.selectedIndex].text}}: %{{y:.2f}}%` +
                               '<extra></extra>' // Hide extra hover info
            }};

            const layout = {{
                title: `Benchmark: ${{yAxisSelect.options[yAxisSelect.selectedIndex].text}} vs ${{xAxisSelect.options[xAxisSelect.selectedIndex].text}}`,
                xaxis: {{
                    title: `${{xAxisSelect.options[xAxisSelect.selectedIndex].text}}${{useLogScaleX ? ' (Log Scale)' : ''}}`,
                    ...getAxisSettings(selectedXAxis, useLogScaleX)
                }},
                yaxis: {{
                    title: yAxisSelect.options[yAxisSelect.selectedIndex].text + ' (%)',
                    ticksuffix: '%',
                    autorange: true
                }},
                hovermode: 'closest',
                margin: {{ l: 60, r: 30, t: 40, b: 70 }},
                autosize: true,
            }};

             // Adjust Y-axis range slightly for better text visibility if needed
            const yValues = filteredData.map(d => d[selectedYAxis] * 100);
            if (yValues.length > 0) {{
                const minY = Math.min(...yValues);
                const maxY = Math.max(...yValues);
                const yPadding = Math.max(2, (maxY - minY) * 0.05);
                layout.yaxis.range = [minY - yPadding, maxY + yPadding * 1.5];
            }}

            // Adjust X-axis range slightly, especially for log scale
             const xValues = filteredData.map(d => formatAxisValue(d[selectedXAxis], selectedXAxis));
             if (xValues.length > 0) {{
                 const minX = Math.min(...xValues);
                 const maxX = Math.max(...xValues);
                 if (useLogScaleX) {{
                    const factor = 1.1;
                    const effectiveMinX = Math.max(minX, 1e-9);
                    layout.xaxis.range = [Math.log10(effectiveMinX / factor), Math.log10(maxX * factor)];
                 }} else {{
                     const xPadding = Math.max(0.01, (maxX - minX) * 0.05);
                     layout.xaxis.range = [minX - xPadding, maxX + xPadding];
                 }}
             }}

            Plotly.newPlot(plotDiv, [trace], layout, {{responsive: true}});
        }}

        // Event listeners for controls
        yAxisSelect.addEventListener('change', updatePlot);
        xAxisSelect.addEventListener('change', updatePlot);
        logScaleXCheckbox.addEventListener('change', updatePlot);

        selectAllButton.addEventListener('click', function(e) {{
            e.stopPropagation(); // Prevent closing the dropdown
            modelDropdownContent.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
            selectedModels = [...allModelNames];
            updateModelButtonText();
            updatePlot();
        }});

        deselectAllButton.addEventListener('click', function(e) {{
            e.stopPropagation(); // Prevent closing the dropdown
            modelDropdownContent.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
            selectedModels = [];
            updateModelButtonText();
            updatePlot();
        }});

        // Initialize data and UI
        initializeData();
    }});
</script>

</body>
</html>
"""

def create_interactive_report(output_dir, data_json_path, autograder_model):
    """
    Generates the interactive index.html file with embedded data.

    Args:
        output_dir (str): The directory where the index.html will be saved.
        data_json_path (str): The path to the data.json file.
        autograder_model (str): The name of the autograder model used.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data from JSON file
    try:
        with open(data_json_path, 'r') as f:
            json_data = json.load(f)
            # Convert the data to a string representation
            json_data_str = json.dumps(json_data)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None
        
    try:
        # Format the template, escaping the JS braces but filling the Python placeholders
        html_content = HTML_TEMPLATE.format(
            timestamp=timestamp,
            autograder_model=autograder_model,
            json_data=json_data_str  # Embed the JSON data directly
        )
    except (ValueError, KeyError) as e:
         print(f"Error formatting HTML template: {e}")
         print("Please double-check that only Python placeholders ({timestamp}, {autograder_model}, {json_data}) use single braces.")
         return None

    # Define the HTML file path
    html_file_path = os.path.join(output_dir, "index.html")

    try:
        # Write the HTML file
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Interactive report with embedded data generated: {html_file_path}")

        return html_file_path

    except Exception as e:
        print(f"Error during report file operations: {e}")
        return None 