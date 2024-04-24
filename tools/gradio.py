import gradio as gr
import pandas as pd

def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
    
    if file_type == 'csv':
        return pd.read_csv(filename, low_memory=False)
    elif file_type == 'xlsx':
        return pd.read_excel(filename)
    elif file_type == 'parquet':
        return pd.read_parquet(filename)


def initial_data_load(in_file):
    new_choices = []
    concat_choices = []
    output_message = ""
    results_df = pd.DataFrame()
    df = pd.DataFrame()
    
    file_list = [string.name for string in in_file]

    data_file_names = [string for string in file_list if "results_on_orig" not in string.lower()]
    if data_file_names:
        df = read_file(data_file_names[0])
    else:
        error_message = "No data file found."
        return error_message, gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), df, results_df

    results_file_names = [string for string in file_list if "results_on_orig" in string.lower()]
    if results_file_names:
        results_df = read_file(results_file_names[0])
    
    new_choices = list(df.columns)
    concat_choices.extend(new_choices)

    output_message = "Data successfully loaded"  
        
    return output_message, gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices), df, results_df


def dummy_function(in_colnames):
    """
    A dummy function that exists just so that dropdown updates work correctly.
    """
    return None    


def clear_inputs(in_file, in_ref, in_text):
    return gr.File.update(value=[]), gr.File.update(value=[]), gr.Textbox.update(value='')