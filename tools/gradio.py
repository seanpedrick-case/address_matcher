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


def put_columns_in_df(in_file):
    new_choices = []
    concat_choices = []
    
    for file in in_file:
        file_list = [string.name for string in in_file]

        #print(file_list)

        data_file_name = [string.lower() for string in file_list if "results_on_orig" not in string.lower()]

        df = read_file(data_file_name[0])
        new_choices = list(df.columns)

        concat_choices.extend(new_choices)     
        
    return gr.Dropdown(choices=concat_choices), gr.Dropdown(choices=concat_choices)


def dummy_function(in_colnames):
    """
    A dummy function that exists just so that dropdown updates work correctly.
    """
    return None    


def clear_inputs(in_file, in_ref, in_text):
    return gr.File.update(value=[]), gr.File.update(value=[]), gr.Textbox.update(value='')