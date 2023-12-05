# Load in packages, variables for fuzzy matching
import os

# Need to overwrite version of gradio present in Huggingface spaces as it doesn't have like buttons/avatars (Oct 2023)
#os.system("pip uninstall -y gradio")
#os.system("pip install gradio==4.50.0")

from datetime import datetime
from pathlib import Path
import time
import copy
import gradio as gr

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# Base folder is where the code file is stored
base_folder = Path(os.getcwd())
input_folder = base_folder/"Input/"
output_folder = base_folder/"Output/"
diagnostics_folder = base_folder/"Diagnostics/"
prep_folder = base_folder/"Helper functions/"

from tools.constants import *
from tools import matcher_funcs
from tools import gradio

def run_matcher(in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol, in_existing, InitMatch = InitMatch):  
    '''
    Split search and reference data into batches. Loop and run through the match script.
    '''

    overall_tic = time.perf_counter()    

    # Load in initial data
    InitMatch = matcher_funcs.load_matcher_data(in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol, in_existing, InitMatch)

    if len(InitMatch.search_df) == 0:
        print("Nothing to match!")
        return "Nothing to match!", InitMatch

    # Determine length of search df to create batches to send through the functions.
    
    def create_batch_ranges(dataframe, batch_size=25000):
        total_rows = dataframe.shape[0]
        ranges = []
        
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            ranges.append(range(start, end))
        
        return ranges

    batch_ranges = create_batch_ranges(InitMatch.search_df)

    print(batch_ranges)

    OutputMatch = copy.copy(InitMatch)

    for n, row_range in enumerate(batch_ranges):
        print("Running batch ", str(n+1))

        BatchMatch = copy.copy(InitMatch)
        BatchMatch.search_df = BatchMatch.search_df.iloc[row_range]

        BatchMatch.search_df_not_matched = BatchMatch.search_df_not_matched.iloc[row_range]

        # TURN ALL THE BELOW INTO A FUNCTION AND RUN THROUGH THE LOOP
        summary_of_summaries, BatchMatch_out = matcher_funcs.run_match_batch(BatchMatch, n, len(batch_ranges))

        OutputMatch = matcher_funcs.combine_two_matches(OutputMatch, BatchMatch_out, "All up to and including batch " + str(n+1))

    
    overall_toc = time.perf_counter()
    time_out = f"The overall match (all batches) took {overall_toc - overall_tic:0.1f} seconds"

    print(OutputMatch.output_summary)

    if OutputMatch.output_summary == "":
        OutputMatch.output_summary = "No matches were found."

    fuzzy_not_std_output = OutputMatch.match_results_output.copy()
    fuzzy_not_std_output_mask = ~(fuzzy_not_std_output["match_method"].str.contains("Fuzzy match")) | (fuzzy_not_std_output["standardised_address"] == True)
    fuzzy_not_std_output.loc[fuzzy_not_std_output_mask, "full_match"] = False
    fuzzy_not_std_summary = matcher_funcs.create_match_summary(fuzzy_not_std_output, "Fuzzy not standardised")

    fuzzy_std_output = OutputMatch.match_results_output.copy()
    fuzzy_std_output_mask = fuzzy_std_output["match_method"].str.contains("Fuzzy match")
    fuzzy_std_output.loc[fuzzy_std_output_mask == False, "full_match"] = False
    fuzzy_std_summary = matcher_funcs.create_match_summary(fuzzy_std_output, "Fuzzy standardised")

    nnet_std_output = OutputMatch.match_results_output.copy()
    nnet_std_summary = matcher_funcs.create_match_summary(nnet_std_output, "Neural net standardised")


    final_summary = fuzzy_not_std_summary + "\n" + fuzzy_std_summary + "\n" + nnet_std_summary + "\n" + time_out


    return final_summary, [OutputMatch.results_orig_df_name, OutputMatch.match_outputs_name]

''' Create the gradio interface '''

block = gr.Blocks(theme = gr.themes.Base())

with block:
    gr.Markdown(
    """
    # Address matcher
    Match single or multiple addresses to the reference address file of your choice. Fuzzy matching should work on any address columns as long as you specify the postcode column at the end. The neural network component only activates with the in-house neural network model - contact me for details if you have access to AddressBase already.The neural network component works with LLPG files in the LPI format.
    
    The tool can accept csv, xlsx (with one sheet), and parquet files. You
     need to specify the address columns of the file to match specifically in the address column area with postcode at the end. 
    
    Use the 'New Column' button to create a new cell for each column name. After you have chosen a reference file, an address match file, and specified its address columns (plus postcode), you can press 'Match addresses' to run the tool.
    """)
    
    with gr.Accordion("I have multiple addresses", open = True):
        in_file = gr.File(label="Input addresses from file", file_count= "multiple")
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
        in_joincol = gr.Dropdown(choices=[], multiselect=True, label="Select columns you want to join on to the search dataset")
    
    match_btn = gr.Button("Match addresses")
    
    with gr.Row():
        output_summary = gr.Textbox(label="Output summary")
        output_file = gr.File(label="Output file")
    
    # Updates to components    
    in_file.upload(fn=gradio.put_columns_in_df, inputs=[in_file], outputs=[in_colnames, in_existing])
    in_ref.upload(fn=gradio.put_columns_in_df, inputs=[in_ref], outputs=[in_refcol, in_joincol])      

    #in_colnames.change(gradio.dummy_function, in_colnames, None)
    #in_colnames.change(gradio.dummy_function, in_existing, None)
    #in_colnames.change(gradio.dummy_function, in_refcol, None)
    #in_colnames.change(gradio.dummy_function, in_joincol, None)

    match_btn.click(fn=run_matcher, inputs=[in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol, in_existing],
                    outputs=[output_summary, output_file], api_name="address")
    
block.queue().launch(debug=True)