# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Load in packages, variables for fuzzy matching

# +
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import os
import copy

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")
# -


import gradio as gr
import tensorflow as tf
import recordlinkage

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


# -

# import importlib
# importlib.reload(fuzzy_funcs)

def put_columns_in_df(in_file):
    new_choices = []
    concat_choices = []
    
    for file in in_file:
        df = fuzzy_funcs.read_file(file.name)
        #print(df.columns)
        #print(list(df.columns))
        new_choices = list(df.columns)

        concat_choices.extend(new_choices)
        concat_choices = list(set(concat_choices))
    return gr.Dropdown.update(choices=new_choices), gr.Dropdown.update(choices=new_choices)


def fuzz_match_single(in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol, progress=gr.Progress(), InitMatch = InitMatch):  
    
    overall_tic = time.perf_counter()
    
    #print("starting")
     
    progress(0, desc="Fuzzy match - non-standardised dataset")
    df_name = "Fuzzy not standardised"
    
    ''' Load in data '''    
    #search_df, search_df_key_field, search_address_cols, search_postcode_col, in_colnames_list, in_joincol_list, ref, ref_address_cols, standard_llpg_format =\
    FuzzyNotStdMatch = fuzzy_funcs.load_matcher_data(in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol, InitMatch)
        
    ''' FUZZY MATCHING '''
        
    ''' Run fuzzy match on non-standardised dataset '''
    
    FuzzyNotStdMatch = fuzzy_funcs.run_fuzzy_match(Matcher = copy.copy(FuzzyNotStdMatch), standardise = False, nnet = False, file_stub= "not_std_", df_name = df_name)
    print(FuzzyNotStdMatch.output_summary)

    FuzzyNotStdMatch.results_on_orig_df = fuzzy_funcs.combine_std_df_remove_dups(FuzzyNotStdMatch.pre_filter_search_df, FuzzyNotStdMatch.results_on_orig_df, orig_addr_col = FuzzyNotStdMatch.search_df_key_field, match_col = 'Matched with ref record')
    FuzzyNotStdMatch.pre_filter_search_df = FuzzyNotStdMatch.results_on_orig_df
    FuzzyNotStdMatch.search_df_not_matched = fuzzy_funcs.filter_not_matched(FuzzyNotStdMatch.match_results_output, FuzzyNotStdMatch.search_df, FuzzyNotStdMatch.search_df_key_field)

    if len(FuzzyNotStdMatch.search_df_not_matched) == 0: 
        overall_toc = time.perf_counter()
        time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"
        FuzzyNotStdMatch.output_summary = FuzzyNotStdMatch.output_summary + ". Neural net match not attempted." + time_out
        return FuzzyNotStdMatch.output_summary, [FuzzyNotStdMatch.match_outputs_name, FuzzyNotStdMatch.results_orig_df_name]


    ''' Run fuzzy match on standardised dataset '''
    
    progress(.25, desc="Fuzzy match - standardised dataset")
    df_name = "Fuzzy standardised"
    
    FuzzyStdMatch = fuzzy_funcs.run_fuzzy_match(Matcher = copy.copy(FuzzyNotStdMatch), standardise = True, nnet = False, file_stub= "std_", df_name = df_name)
    FuzzyStdMatch = fuzzy_funcs.combine_two_matches(FuzzyNotStdMatch, FuzzyStdMatch, df_name)

    ''' Continue if reference file in correct format, and neural net model exists '''
    if ((len(FuzzyStdMatch.search_df_not_matched) == 0) | (FuzzyStdMatch.standard_llpg_format == False) | (os.path.exists(FuzzyStdMatch.model_dir_name + '/saved_model.zip') == False)):
        overall_toc = time.perf_counter()
        time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"
        FuzzyStdMatch.output_summary = FuzzyStdMatch.output_summary + ". Neural net match not attempted." + time_out
        return FuzzyStdMatch.output_summary, [FuzzyStdMatch.match_outputs_name, FuzzyStdMatch.results_orig_df_name]
    
    
    ''' NEURAL NET '''

    ''' First on non-standardised addresses '''
    progress(.50, desc="Neural net - non-standardised dataset")
    df_name = "Neural net not standardised"
    
    
    FuzzyNNetNotStdMatch = fuzzy_funcs.run_fuzzy_match(Matcher = copy.copy(FuzzyStdMatch), standardise = False, nnet = True, file_stub= "nnet_not_std_", df_name = df_name)
    FuzzyNNetNotStdMatch = fuzzy_funcs.combine_two_matches(FuzzyStdMatch, FuzzyNNetNotStdMatch, df_name)
    

    if (len(FuzzyNNetNotStdMatch.search_df_not_matched) == 0):
        overall_toc = time.perf_counter()
        time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"
        FuzzyNNetNotStdMatch.output_summary + time_out
        return FuzzyNNetNotStdMatch.output_summary, [FuzzyNNetNotStdMatch.match_outputs_name, FuzzyNNetNotStdMatch.results_orig_df_name]


    ''' Next on standardised addresses '''
    progress(.75, desc="Neural net - standardised dataset")
    df_name = "Neural net standardised"
    
    FuzzyNNetStdMatch = fuzzy_funcs.run_fuzzy_match(Matcher = copy.copy(FuzzyNNetNotStdMatch), standardise = True, nnet = True, file_stub= "nnet_std_", df_name = df_name)
    FuzzyNNetStdMatch = fuzzy_funcs.combine_two_matches(FuzzyNNetNotStdMatch, FuzzyNNetStdMatch, df_name)

    #FuzzyNNetStdMatch.match_results_output = fuzzy_funcs.combine_std_df_remove_dups(FuzzyNNetNotStdMatch.match_results_output, FuzzyNNetStdMatch.match_results_output, orig_addr_col = FuzzyNNetStdMatch.search_df_key_field)    
    #FuzzyNNetStdMatch.results_on_orig_df = fuzzy_funcs.combine_std_df_remove_dups(FuzzyNNetNotStdMatch.results_on_orig_df, FuzzyNNetStdMatch.results_on_orig_df, orig_addr_col = FuzzyNNetStdMatch.search_df_key_field, match_col = 'Matched with ref record')
    

    overall_toc = time.perf_counter()
    time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"
    summary_of_summaries = FuzzyNotStdMatch.output_summary + "\n" + FuzzyStdMatch.output_summary + "\n" + FuzzyNNetStdMatch.output_summary + "\n" + time_out
    
    return summary_of_summaries, [FuzzyNNetStdMatch.match_outputs_name, FuzzyNNetStdMatch.results_orig_df_name]
# +
''' Create the gradio interface '''

block = gr.Blocks()#(theme = gr.themes.Base())

with block as demo:
    gr.Markdown(
    """
    # Address matcher
    Match single or multiple addresses to the reference address file of your choice. Fuzzy matching should work on any address columns as long as you specify the postcode column at the end. The neural network component only activates with the in-house neural network model - contact me for details if you have access to AddressBase already.The neural network component works with LLPG files in the LPI format.
    
    The tool can accept csv.gz files. You need to specify the address columns of the file to match specifically in the address column area with postcode at the end. 
    
    Use the 'New Column' button to create a new cell for each column name. After you have chosen a reference file, an address match file, and specified its address columns (plus postcode), you can press 'Match addresses' to run the tool.
    """)
    
    with gr.Accordion("I have multiple addresses", open = True):
        in_file = gr.File(label="Input addresses from file", file_count= "multiple")
        #in_colnames = gr.Dataframe(label="Input file address column names (put postcode on end)", type="numpy", row_count=(1,"fixed"), col_count = 1,
                              # headers=["Address column name 1"])#, "Address column name 2", "Address column name 3", "Address column name 4"])
        in_colnames = gr.Dropdown(choices=[], multiselect=True, label="Select columns that make up the address. Make sure postcode is at the end")
    
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
        
    in_file.change(fn=put_columns_in_df, inputs=[in_file], outputs=[in_colnames, in_colnames])
    in_ref.change(fn=put_columns_in_df, inputs=[in_ref], outputs=[in_refcol, in_joincol])
    #in_ref.change(fn=put_columns_in_df, inputs=[in_ref], outputs=[in_joincol])

    match_btn.click(fn=fuzz_match_single, inputs=[in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol],
                    outputs=[output_summary, output_file], api_name="address")

demo.queue(concurrency_count=1).launch(debug=True)
# -

