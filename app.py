# Load in packages, variables for fuzzy matching

import os

# Need to overwrite version of gradio present in Huggingface spaces as it doesn't have like buttons/avatars (Oct 2023)
#os.system("pip uninstall -y gradio")
os.system("pip install gradio==3.50.0")

# +
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import copy

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")
# -

import gradio as gr

# Base folder is where the code file is stored
base_folder = Path(os.getcwd())
input_folder = base_folder/"Input/"
output_folder = base_folder/"Output/"
diagnostics_folder = base_folder/"Diagnostics/"
prep_folder = base_folder/"Helper functions/"

#import tools
from tools.constants import *

# +
#tools.fuzzy_funcs.labels_list = labels_list

from tools import fuzzy_funcs



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
        df = read_file(file.name)
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


run_fuzzy_match = True
run_nnet_match = True
run_standardise = True


def fuzz_match_single(in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol, in_existing, progress=gr.Progress(), InitMatch = InitMatch):  
    
    overall_tic = time.perf_counter()
    

    ''' Load in data '''
    InitMatch = fuzzy_funcs.load_matcher_data(in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol, in_existing, InitMatch)

    if len(InitMatch.search_df) == 0:
        print("Nothing to match!")
        return "Nothing to match!", [InitMatch.results_orig_df_name, InitMatch.match_outputs_name]

    if run_fuzzy_match == True:
    
        #print("starting")
         
        progress(0, desc="Fuzzy match - non-standardised dataset")
        df_name = "Fuzzy not standardised"
        
        
            
        ''' FUZZY MATCHING '''
            
        ''' Run fuzzy match on non-standardised dataset '''
        
        FuzzyNotStdMatch = fuzzy_funcs.run_fuzzy_match(Matcher = copy.copy(InitMatch), standardise = False, nnet = False, file_stub= "not_std_", df_name = df_name)
        #print(FuzzyNotStdMatch.output_summary)

        if FuzzyNotStdMatch.abort_flag == True:
            print("Nothing to match!")
            return "Nothing to match!", [InitMatch.results_orig_df_name, InitMatch.match_outputs_name]

        FuzzyNotStdMatch = fuzzy_funcs.combine_two_matches(InitMatch, FuzzyNotStdMatch, df_name)
        
        if (len(FuzzyNotStdMatch.search_df_not_matched) == 0) | (sum(FuzzyNotStdMatch.match_results_output[FuzzyNotStdMatch.match_results_output['full_match']==False]['fuzzy_score'])==0): 
            overall_toc = time.perf_counter()
            time_out = f"The fuzzy match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNotStdMatch.output_summary = FuzzyNotStdMatch.output_summary + " Neural net match not attempted. " + time_out
            return FuzzyNotStdMatch.output_summary, [FuzzyNotStdMatch.results_orig_df_name, FuzzyNotStdMatch.match_outputs_name]
    
    
        ''' Run fuzzy match on standardised dataset '''
        
        progress(.25, desc="Fuzzy match - standardised dataset")
        df_name = "Fuzzy standardised"
        
        FuzzyStdMatch = fuzzy_funcs.run_fuzzy_match(Matcher = copy.copy(FuzzyNotStdMatch), standardise = True, nnet = False, file_stub= "std_", df_name = df_name)
        FuzzyStdMatch = fuzzy_funcs.combine_two_matches(FuzzyNotStdMatch, FuzzyStdMatch, df_name)
    
        ''' Continue if reference file in correct format, and neural net model exists. Also if data not too long '''
        if ((len(FuzzyStdMatch.search_df_not_matched) == 0) | (FuzzyStdMatch.standard_llpg_format == False) |\
            (os.path.exists(FuzzyStdMatch.model_dir_name + '/saved_model.zip') == False) | (run_nnet_match == False)):
            overall_toc = time.perf_counter()
            time_out = f"The fuzzy match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyStdMatch.output_summary = FuzzyStdMatch.output_summary + " Neural net match not attempted. " + time_out
            return FuzzyStdMatch.output_summary, [FuzzyStdMatch.results_orig_df_name, FuzzyStdMatch.match_outputs_name]

    if run_nnet_match == True:
    
        ''' NEURAL NET '''

        if run_fuzzy_match == False:
            FuzzyStdMatch = copy.copy(InitMatch)
        
    
        ''' First on non-standardised addresses '''
        progress(.50, desc="Neural net - non-standardised dataset")
        df_name = "Neural net not standardised"
        
        
        FuzzyNNetNotStdMatch = fuzzy_funcs.run_fuzzy_match(Matcher = copy.copy(FuzzyStdMatch), standardise = False, nnet = True, file_stub= "nnet_not_std_", df_name = df_name)
        FuzzyNNetNotStdMatch = fuzzy_funcs.combine_two_matches(FuzzyStdMatch, FuzzyNNetNotStdMatch, df_name)
        
    
        if (len(FuzzyNNetNotStdMatch.search_df_not_matched) == 0):
            overall_toc = time.perf_counter()
            time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNNetNotStdMatch.output_summary = FuzzyNNetNotStdMatch.output_summary + time_out
            return FuzzyNNetNotStdMatch.output_summary, [FuzzyNNetNotStdMatch.results_orig_df_name, FuzzyNNetNotStdMatch.match_outputs_name]
    
    
        ''' Next on standardised addresses '''
        progress(.75, desc="Neural net - standardised dataset")
        df_name = "Neural net standardised"
        
        FuzzyNNetStdMatch = fuzzy_funcs.run_fuzzy_match(Matcher = copy.copy(FuzzyNNetNotStdMatch), standardise = True, nnet = True, file_stub= "nnet_std_", df_name = df_name)
        FuzzyNNetStdMatch = fuzzy_funcs.combine_two_matches(FuzzyNNetNotStdMatch, FuzzyNNetStdMatch, df_name)
    
        #FuzzyNNetStdMatch.match_results_output = fuzzy_funcs.combine_std_df_remove_dups(FuzzyNNetNotStdMatch.match_results_output, FuzzyNNetStdMatch.match_results_output, orig_addr_col = FuzzyNNetStdMatch.search_df_key_field)    
        #FuzzyNNetStdMatch.results_on_orig_df = fuzzy_funcs.combine_std_df_remove_dups(FuzzyNNetNotStdMatch.results_on_orig_df, FuzzyNNetStdMatch.results_on_orig_df, orig_addr_col = FuzzyNNetStdMatch.search_df_key_field, match_col = 'Matched with ref record')

        if run_fuzzy_match ==  False:
            overall_toc = time.perf_counter()
            time_out = f"The neural net match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNNetStdMatch.output_summary = FuzzyNNetStdMatch.output_summary + " Only Neural net match attempted. " + time_out
            return FuzzyNNetStdMatch.output_summary, [FuzzyNNetStdMatch.results_orig_df_name, FuzzyNNetStdMatch.match_outputs_name]
    
    overall_toc = time.perf_counter()
    time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"
    summary_of_summaries = FuzzyNotStdMatch.output_summary + "\n" + FuzzyStdMatch.output_summary + "\n" + FuzzyNNetStdMatch.output_summary + "\n" + time_out
    
    return summary_of_summaries, [FuzzyNNetStdMatch.results_orig_df_name, FuzzyNNetStdMatch.match_outputs_name]

#import importlib
#importlib.reload(fuzzy_funcs)

# +
''' Create the gradio interface '''

block = gr.Blocks(theme = gr.themes.Base())

with block:
    gr.Markdown(
    """
    # Address matcher
    Match single or multiple addresses to the reference address file of your choice. Fuzzy matching should work on any address columns as long as you specify the postcode column at the end. The neural network component only activates with the in-house neural network model - contact me for details if you have access to AddressBase already.The neural network component works with LLPG files in the LPI format.
    
    The tool can accept csv, xlsx (with one sheet), and parquet files. You need to specify the address columns of the file to match specifically in the address column area with postcode at the end. 
    
    Use the 'New Column' button to create a new cell for each column name. After you have chosen a reference file, an address match file, and specified its address columns (plus postcode), you can press 'Match addresses' to run the tool.
    """)
    
    with gr.Accordion("I have multiple addresses", open = True):
        in_file = gr.File(label="Input addresses from file", file_count= "multiple")
        #in_colnames = gr.Dataframe(label="Input file address column names (put postcode on end)", type="numpy", row_count=(1,"fixed"), col_count = 1,
                              # headers=["Address column name 1"])#, "Address column name 2", "Address column name 3", "Address column name 4"])
        in_colnames = gr.Dropdown(choices=[], multiselect=True, label="Select columns that make up the address. Make sure postcode is at the end")

        in_existing = gr.Dropdown(choices=[], multiselect=False, label="Select columns that indicate existing matches.")
    
    with gr.Accordion("I only have a single address", open = False):
        in_text = gr.Textbox(label="Input a single address as text")
    
    gr.Markdown(
    """
    ## Choose reference file
    Fuzzy matching will work on any address format, but the neural network will only work with the LLPG LPI format, e.g. with columns SaoText, SaoStartNumber etc.. This joins on the UPRN column. If any of these are different for you, 
    open 'Custom reference file format or join columns' below.
    """)
    
    in_ref = gr.File(label="Input reference addresses from file", file_count= "multiple")
    
    with gr.Accordion("Custom reference file format or join columns (i.e. not LLPG LPI format)", open = False):
        in_refcol = gr.Dropdown(choices=[], multiselect=True, label="Select columns that make up the reference address. Make sure postcode is at the end")
                    #.Dataframe(label="Input reference column names (put postcode on end)", type="numpy", row_count=(1,"fixed"), col_count = 1,
                               #headers=["Reference column name 1"])
        in_joincol = gr.Dropdown(choices=[], multiselect=True, label="Select columns you want to join on to the search dataset")#label="Input column names to join from reference file", type="numpy", row_count=(1,"fixed"), col_count = 1,
                               #headers=["Reference join column 1"])
    
    match_btn = gr.Button("Match addresses")
    
    with gr.Row():
        output_summary = gr.Textbox(label="Output summary")
        output_file = gr.File(label="Output file")
        
    # Updates to components
        
    #in_file.change(fn=put_columns_in_df, inputs=[in_file], outputs=[in_colnames, in_existing])
    #in_ref.change(fn=put_columns_in_df, inputs=[in_ref], outputs=[in_refcol, in_joincol])
    #in_ref.change(fn=put_columns_in_df, inputs=[in_ref], outputs=[in_joincol])

    in_file.upload(fn=put_columns_in_df, inputs=[in_file], outputs=[in_colnames, in_existing])
    in_ref.upload(fn=put_columns_in_df, inputs=[in_ref], outputs=[in_refcol, in_joincol])      

    in_colnames.change(dummy_function, in_colnames, None)
    in_colnames.change(dummy_function, in_existing, None)
    in_colnames.change(dummy_function, in_refcol, None)
    in_colnames.change(dummy_function, in_joincol, None)

    match_btn.click(fn=fuzz_match_single, inputs=[in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol, in_existing],
                    outputs=[output_summary, output_file], api_name="address")#.\
            #then(fn=clear_inputs, inputs = [in_file, in_ref, in_text], outputs = [in_file, in_ref, in_text], queue=False)
block.queue(concurrency_count=1).launch(debug=True)