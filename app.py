# Load in packages, variables for fuzzy matching
import os
from datetime import datetime
from pathlib import Path
import time
import copy
import gradio as gr
import re
#import polars as pl

from tools.constants import *
from tools.matcher_funcs import load_matcher_data, run_match_batch, combine_two_matches, create_match_summary
from tools.gradio import initial_data_load
from tools.aws_functions import load_data_from_aws
from tools.preparation import prepare_search_address_string, prepare_search_address, prepare_ref_address, remove_non_postal, check_no_number_addresses
from tools.standardise import standardise_wrapper_func

import warnings
warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression')
warnings.filterwarnings("ignore", 'Downcasting behavior')
warnings.filterwarnings("ignore", 'A value is trying to be set on a copy of a slice from a DataFrame')
warnings.filterwarnings("ignore")


today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# Base folder is where the code file is stored
base_folder = Path(os.getcwd())
input_folder = base_folder/"Input/"
output_folder = base_folder/"Output/"
diagnostics_folder = base_folder/"Diagnostics/"
prep_folder = base_folder/"Helper functions/"

def create_simple_batch_ranges(df:PandasDataFrame, ref_df:PandasDataFrame, batch_size:int, ref_batch_size:int):
    #print("Search df batch size: ", batch_size)
    #print("ref_df df batch size: ", ref_batch_size)

    total_rows = df.shape[0]
    ref_total_rows = ref_df.shape[0]

    # Creating bottom and top limits for search data
    search_ranges = []
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size - 1, total_rows - 1)  # Adjusted to get the top limit
        search_ranges.append((start, end))

    # Creating bottom and top limits for reference data
    ref_ranges = []
    for start in range(0, ref_total_rows, ref_batch_size):
        end = min(start + ref_batch_size - 1, ref_total_rows - 1)  # Adjusted to get the top limit
        ref_ranges.append((start, end))

    # Create DataFrame with combinations of search_range and ref_range
    result_data = []
    for search_range in search_ranges:
        for ref_range in ref_ranges:
            result_data.append((search_range, ref_range))

    range_df = pd.DataFrame(result_data, columns=['search_range', 'ref_range'])

    return range_df


def create_batch_ranges(df:PandasDataFrame, ref_df:PandasDataFrame, batch_size:int, ref_batch_size:int, search_postcode_col:str, ref_postcode_col:str):
    '''
    Create batches of address indexes for search and reference dataframes based on shortened postcodes.
    '''

    # If df sizes are smaller than the batch size limits, no need to run through everything
    if len(df) < batch_size and len(ref_df) < ref_batch_size:
        print("Dataframe sizes are smaller than maximum batch sizes, no need to split data.")
        lengths_df = pd.DataFrame(data={'search_range':[df.index.tolist()], 'ref_range':[ref_df.index.tolist()], 'batch_length':len(df), 'ref_length':len(ref_df)})
        return lengths_df
    
    #df.index = df[search_postcode_col]

    df['index'] = df.index
    ref_df['index'] = ref_df.index    

    # Remove the last character of postcode
    df['postcode_minus_last_character'] = df[search_postcode_col].str.lower().str.strip().str.replace("\s+", "", regex=True).str[:-1]
    ref_df['postcode_minus_last_character'] = ref_df[ref_postcode_col].str.lower().str.strip().str.replace("\s+", "", regex=True).str[:-1]

    unique_postcodes = df['postcode_minus_last_character'][df['postcode_minus_last_character'].str.len()>=4].unique().tolist()

    df = df.set_index('postcode_minus_last_character')
    ref_df = ref_df.set_index('postcode_minus_last_character')

    df = df.sort_index()
    ref_df = ref_df.sort_index()

    #df.to_csv("batch_search_df.csv")

    # Overall batch variables
    batch_indexes = []
    ref_indexes = []
    batch_lengths = []
    ref_lengths = []

    # Current batch variables for loop
    current_batch = []
    current_ref_batch = []
    current_batch_length = []
    current_ref_length = []

    unique_postcodes_iterator = unique_postcodes.copy()

    while unique_postcodes_iterator:
        
        unique_postcodes_loop = unique_postcodes_iterator.copy()

        #print("Current loop postcodes: ", unique_postcodes_loop)

        for current_postcode in unique_postcodes_loop:
            


            if len(current_batch) >= batch_size or len(current_ref_batch) >= ref_batch_size:
                print("Batch length reached - breaking")
                break
            
            try:
                current_postcode_search_data_add = df.loc[[current_postcode]]#[df['postcode_minus_last_character'].isin(current_postcode)]
                current_postcode_ref_data_add = ref_df.loc[[current_postcode]]#[ref_df['postcode_minus_last_character'].isin(current_postcode)]

                #print(current_postcode_search_data_add)

                if not current_postcode_search_data_add.empty:
                    current_batch.extend(current_postcode_search_data_add['index'])

                if not current_postcode_ref_data_add.empty:
                    current_ref_batch.extend(current_postcode_ref_data_add['index'])  

            except:
                #print("postcode not found: ", current_postcode)
                pass

            unique_postcodes_iterator.remove(current_postcode)

        # Append the batch data to the master lists and reset lists
        batch_indexes.append(current_batch)
        ref_indexes.append(current_ref_batch)

        current_batch_length = len(current_batch)
        current_ref_length = len(current_ref_batch)

        batch_lengths.append(current_batch_length)
        ref_lengths.append(current_ref_length)

        current_batch = []
        current_ref_batch = []
        current_batch_length = []
        current_ref_length = []
        
    # Create df to store lengths
    lengths_df = pd.DataFrame(data={'search_range':batch_indexes, 'ref_range':ref_indexes, 'batch_length':batch_lengths, 'ref_length':ref_lengths})
    
    return lengths_df


def run_matcher(in_text, in_file, in_ref, data_state:PandasDataFrame, results_data_state:PandasDataFrame, ref_data_state:PandasDataFrame, in_colnames:List[str], in_refcol:List[str], in_joincol:List[str], in_existing:List[str], in_api:str, in_api_key:str, InitMatch:MatcherClass = InitMatch, progress=gr.Progress()):  
    '''
    Split search and reference data into batches. Loop and run through the match script.
    '''

    overall_tic = time.perf_counter()

    # Load in initial data. This will filter to relevant addresses in the search and reference datasets that can potentially be matched, and will pull in API data if asked for.
    InitMatch = load_matcher_data(in_text, in_file, in_ref, data_state, results_data_state, ref_data_state, in_colnames, in_refcol, in_joincol, in_existing, InitMatch, in_api, in_api_key)

    if InitMatch.search_df.empty or InitMatch.ref_df.empty:
        out_message = "Nothing to match!"
        print(out_message)
        return out_message, [InitMatch.results_orig_df_name, InitMatch.match_outputs_name]
    
    # Run initial address preparation and standardisation processes   
    # Prepare address format

    # Polars implementation not yet finalised
    #InitMatch.search_df = pl.from_pandas(InitMatch.search_df)
    #InitMatch.ref_df = pl.from_pandas(InitMatch.ref_df)

    
    # Prepare all search addresses
    if type(InitMatch.search_df) == str:
        InitMatch.search_df_cleaned, InitMatch.search_df_key_field, InitMatch.search_address_cols = prepare_search_address_string(InitMatch.search_df)
    else: 
        InitMatch.search_df_cleaned = prepare_search_address(InitMatch.search_df, InitMatch.search_address_cols, InitMatch.search_postcode_col, InitMatch.search_df_key_field)

        # Remove addresses that are not postal addresses
    InitMatch.search_df_cleaned = remove_non_postal(InitMatch.search_df_cleaned, "full_address")

    # Remove addresses that have no numbers in from consideration
    InitMatch.search_df_cleaned = check_no_number_addresses(InitMatch.search_df_cleaned, "full_address")

    # Initial preparation of reference addresses
    InitMatch.ref_df_cleaned = prepare_ref_address(InitMatch.ref_df, InitMatch.ref_address_cols, InitMatch.new_join_col)
    

    # Sort dataframes by postcode - will allow for more efficient matching process if using multiple batches
    #InitMatch.search_df_cleaned = InitMatch.search_df_cleaned.sort_values(by="postcode")
    #InitMatch.ref_df_cleaned = InitMatch.ref_df_cleaned.sort_values(by="Postcode")

    # Polars implementation - not finalised
    #InitMatch.search_df_cleaned = InitMatch.search_df_cleaned.to_pandas()
    #InitMatch.ref_df_cleaned = InitMatch.ref_df_cleaned.to_pandas()

    # Standardise addresses    
    # Standardise - minimal


    tic = time.perf_counter()
    InitMatch.search_df_after_stand, InitMatch.ref_df_after_stand = standardise_wrapper_func(
        InitMatch.search_df_cleaned.copy(),
        InitMatch.ref_df_cleaned.copy(),
        standardise = False,
        filter_to_lambeth_pcodes=filter_to_lambeth_pcodes,
        match_task="fuzzy") # InitMatch.search_df_after_stand_series, InitMatch.ref_df_after_stand_series

    toc = time.perf_counter()
    print(f"Performed the minimal standardisation step in {toc - tic:0.1f} seconds")

    # Standardise - full
    tic = time.perf_counter()
    InitMatch.search_df_after_full_stand, InitMatch.ref_df_after_full_stand = standardise_wrapper_func(
        InitMatch.search_df_cleaned.copy(),
        InitMatch.ref_df_cleaned.copy(),
        standardise = True,
        filter_to_lambeth_pcodes=filter_to_lambeth_pcodes,
        match_task="fuzzy") # , InitMatch.search_df_after_stand_series_full_stand, InitMatch.ref_df_after_stand_series_full_stand

    toc = time.perf_counter()
    print(f"Performed the full standardisation step in {toc - tic:0.1f} seconds")

    # Determine length of search df to create batches to send through the functions.
    #try:
    range_df = create_batch_ranges(InitMatch.search_df_cleaned.copy(), InitMatch.ref_df_cleaned.copy(), batch_size, ref_batch_size, "postcode", "Postcode")
    #except:
    #    range_df = create_simple_batch_ranges(InitMatch.search_df_cleaned, InitMatch.ref_df_cleaned, batch_size, #ref_batch_size)

    print("Batches to run in this session: ", range_df)

    OutputMatch = copy.copy(InitMatch)

    n = 0
    number_of_batches = range_df.shape[0]

    for row in progress.tqdm(range(0,len(range_df)), desc= "Running through batches", unit="batches", total=number_of_batches):
        print("Running batch ", str(n+1))

        search_range = range_df.iloc[row]['search_range']
        ref_range = range_df.iloc[row]['ref_range']

        #print("search_range: ", search_range)
        #pd.DataFrame(search_range).to_csv("search_range.csv")
        #print("ref_range: ", ref_range)
        
        BatchMatch = copy.copy(InitMatch)

        # Subset the search and reference dfs based on current batch ranges
        # BatchMatch.search_df = BatchMatch.search_df.iloc[search_range[0]:search_range[1] + 1,:].reset_index(drop=True)
        # BatchMatch.search_df_not_matched = BatchMatch.search_df.copy()
        # BatchMatch.search_df_cleaned = BatchMatch.search_df_cleaned.iloc[search_range[0]:search_range[1] + 1,:].reset_index(drop=True)
        # BatchMatch.ref_df = BatchMatch.ref_df.iloc[ref_range[0]:ref_range[1] + 1,:].reset_index(drop=True)
        # BatchMatch.ref_df_cleaned = BatchMatch.ref_df_cleaned.iloc[ref_range[0]:ref_range[1] + 1,:].reset_index(drop=True)


        # BatchMatch.search_df_after_stand_series = BatchMatch.search_df_after_stand_series.iloc[search_range[0]:search_range[1] + 1]
        # BatchMatch.ref_df_after_stand_series = BatchMatch.ref_df_after_stand_series.iloc[ref_range[0]:ref_range[1] + 1]
        # BatchMatch.search_df_after_stand_series_full_stand = BatchMatch.search_df_after_stand_series_full_stand.iloc[search_range[0]:search_range[1] + 1]
        # BatchMatch.ref_df_after_stand_series_full_stand = BatchMatch.ref_df_after_stand_series_full_stand.iloc[ref_range[0]:ref_range[1] + 1]

        # BatchMatch.search_df_after_stand = BatchMatch.search_df_after_stand.iloc[search_range[0]:search_range[1] + 1,:].reset_index(drop=True)
        # BatchMatch.ref_df_after_stand = BatchMatch.ref_df_after_stand.iloc[ref_range[0]:ref_range[1] + 1,:].reset_index(drop=True)
        # BatchMatch.search_df_after_full_stand = BatchMatch.search_df_after_full_stand.iloc[search_range[0]:search_range[1] + 1,:].reset_index(drop=True)
        # BatchMatch.ref_df_after_full_stand = BatchMatch.ref_df_after_full_stand.iloc[ref_range[0]:ref_range[1] + 1,:].reset_index(drop=True)

        BatchMatch.search_df = BatchMatch.search_df[BatchMatch.search_df.index.isin(search_range)].reset_index(drop=True)
        BatchMatch.search_df_not_matched = BatchMatch.search_df.copy()
        BatchMatch.search_df_cleaned = BatchMatch.search_df_cleaned[BatchMatch.search_df_cleaned.index.isin(search_range)].reset_index(drop=True)

        BatchMatch.ref_df = BatchMatch.ref_df[BatchMatch.ref_df.index.isin(ref_range)].reset_index(drop=True)
        BatchMatch.ref_df_cleaned = BatchMatch.ref_df_cleaned[BatchMatch.ref_df_cleaned.index.isin(ref_range)].reset_index(drop=True)

        # Dataframes after standardisation process
        BatchMatch.search_df_after_stand = BatchMatch.search_df_after_stand[BatchMatch.search_df_after_stand.index.isin(search_range)].reset_index(drop=True)
        BatchMatch.search_df_after_full_stand = BatchMatch.search_df_after_full_stand[BatchMatch.search_df_after_full_stand.index.isin(search_range)].reset_index(drop=True)

        ### Create lookup lists for fuzzy matches
        # BatchMatch.search_df_after_stand_series = BatchMatch.search_df_after_stand.copy().set_index('postcode_search')['search_address_stand']
        # BatchMatch.search_df_after_stand_series_full_stand = BatchMatch.search_df_after_full_stand.copy().set_index('postcode_search')['search_address_stand']
        # BatchMatch.search_df_after_stand_series = BatchMatch.search_df_after_stand_series.sort_index()
        # BatchMatch.search_df_after_stand_series_full_stand = BatchMatch.search_df_after_stand_series_full_stand.sort_index()

        #BatchMatch.search_df_after_stand.reset_index(inplace=True, drop = True)
        #BatchMatch.search_df_after_full_stand.reset_index(inplace=True, drop = True)

        BatchMatch.ref_df_after_stand = BatchMatch.ref_df_after_stand[BatchMatch.ref_df_after_stand.index.isin(ref_range)].reset_index(drop=True)
        BatchMatch.ref_df_after_full_stand = BatchMatch.ref_df_after_full_stand[BatchMatch.ref_df_after_full_stand.index.isin(ref_range)].reset_index(drop=True)

        # BatchMatch.ref_df_after_stand_series = BatchMatch.ref_df_after_stand.copy().set_index('postcode_search')['ref_address_stand']
        # BatchMatch.ref_df_after_stand_series_full_stand = BatchMatch.ref_df_after_full_stand.copy().set_index('postcode_search')['ref_address_stand']
        # BatchMatch.ref_df_after_stand_series = BatchMatch.ref_df_after_stand_series.sort_index()
        # BatchMatch.ref_df_after_stand_series_full_stand = BatchMatch.ref_df_after_stand_series_full_stand.sort_index()

        # BatchMatch.ref_df_after_stand.reset_index(inplace=True, drop=True)
        # BatchMatch.ref_df_after_full_stand.reset_index(inplace=True, drop=True)

        # Match the data, unless the search or reference dataframes are empty
        if BatchMatch.search_df.empty or BatchMatch.ref_df.empty:
            out_message = "Nothing to match for batch: " + str(n)
            print(out_message)
            BatchMatch_out = BatchMatch
            BatchMatch_out.results_on_orig_df = pd.DataFrame(data={"index":BatchMatch.search_df.index,
                                                                   "Excluded from search":False,
                                                                    "Matched with reference address":False})
        else:
            summary_of_summaries, BatchMatch_out = run_match_batch(BatchMatch, n, number_of_batches)
        
        OutputMatch = combine_two_matches(OutputMatch, BatchMatch_out, "All up to and including batch " + str(n+1))

        n += 1

    if in_api==True:
        OutputMatch.results_on_orig_df['Matched with reference address'] = OutputMatch.results_on_orig_df['Matched with reference address'].replace({1:True, 0:False})
        OutputMatch.results_on_orig_df['Excluded from search'] = OutputMatch.results_on_orig_df['Excluded from search'].replace('nan', False).fillna(False)

    # Remove any duplicates from reference df, prioritise successful matches
    OutputMatch.results_on_orig_df = OutputMatch.results_on_orig_df.sort_values(by=["index", "Matched with reference address"], ascending=[True,False]).drop_duplicates(subset="index")


    overall_toc = time.perf_counter()
    time_out = f"The overall match (all batches) took {overall_toc - overall_tic:0.1f} seconds"

    print(OutputMatch.output_summary)

    if OutputMatch.output_summary == "":
        OutputMatch.output_summary = "No matches were found."

    fuzzy_not_std_output = OutputMatch.match_results_output.copy()
    fuzzy_not_std_output_mask = ~(fuzzy_not_std_output["match_method"].str.contains("Fuzzy match")) | (fuzzy_not_std_output["standardised_address"] == True)
    fuzzy_not_std_output.loc[fuzzy_not_std_output_mask, "full_match"] = False
    fuzzy_not_std_summary = create_match_summary(fuzzy_not_std_output, "Fuzzy not standardised")

    fuzzy_std_output = OutputMatch.match_results_output.copy()
    fuzzy_std_output_mask = fuzzy_std_output["match_method"].str.contains("Fuzzy match")
    fuzzy_std_output.loc[fuzzy_std_output_mask == False, "full_match"] = False
    fuzzy_std_summary = create_match_summary(fuzzy_std_output, "Fuzzy standardised")

    nnet_std_output = OutputMatch.match_results_output.copy()
    nnet_std_summary = create_match_summary(nnet_std_output, "Neural net standardised")

    final_summary = fuzzy_not_std_summary + "\n" + fuzzy_std_summary + "\n" + nnet_std_summary + "\n" + time_out

    return final_summary, [OutputMatch.results_orig_df_name, OutputMatch.match_outputs_name]

# Create the gradio interface

block = gr.Blocks(theme = gr.themes.Base())

with block:

    data_state = gr.State(pd.DataFrame())
    ref_data_state = gr.State(pd.DataFrame())
    results_data_state = gr.State(pd.DataFrame())
    ref_results_data_state =gr.State(pd.DataFrame())

    gr.Markdown(
    """
    # Address matcher
    Match single or multiple addresses to the reference address file of your choice. Fuzzy matching should work on any address columns as long as you specify the postcode column at the end. The neural network component only activates with the in-house neural network model - contact me for details if you have access to AddressBase already.The neural network component works with LLPG files in the LPI format.
    
    The tool can accept csv, xlsx (with one sheet), and parquet files. You
     need to specify the address columns of the file to match specifically in the address column area with postcode at the end. 
    
    Use the 'New Column' button to create a new cell for each column name. After you have chosen a reference file, an address match file, and specified its address columns (plus postcode), you can press 'Match addresses' to run the tool.
    """)

    with gr.Tab("Match addresses"):
    
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

        with gr.Accordion("Use Addressbase API instead of reference file", open = False):
            in_api = gr.Dropdown(label="Choose API type", multiselect=False, value=None, choices=["Postcode", "UPRN"]) #choices=["Address", "Postcode", "UPRN"])
            in_api_key = gr.Textbox(label="Addressbase API key")
        
        with gr.Accordion("Custom reference file format or join columns (i.e. not LLPG LPI format)", open = False):
            in_refcol = gr.Dropdown(choices=[], multiselect=True, label="Select columns that make up the reference address. Make sure postcode is at the end")
            in_joincol = gr.Dropdown(choices=[], multiselect=True, label="Select columns you want to join on to the search dataset")
        
        match_btn = gr.Button("Match addresses")
        
        with gr.Row():
            output_summary = gr.Textbox(label="Output summary")
            output_file = gr.File(label="Output file")

    with gr.Tab(label="Advanced options"):
        with gr.Accordion(label = "AWS data access", open = False):
                aws_password_box = gr.Textbox(label="Password for AWS data access (ask the Data team if you don't have this)")
                with gr.Row():
                    in_aws_file = gr.Dropdown(label="Choose keyword file to load from AWS (only valid for API Gateway app)", choices=["None", "Lambeth address data example file"])
                    load_aws_data_button = gr.Button(value="Load keyword data from AWS", variant="secondary")
                    
                aws_log_box = gr.Textbox(label="AWS data load status")

    
    ### Loading AWS data ###
    load_aws_data_button.click(fn=load_data_from_aws, inputs=[in_aws_file, aws_password_box], outputs=[in_ref, aws_log_box])
    

    # Updates to components
    in_file.change(fn = initial_data_load, inputs=[in_file], outputs=[output_summary, in_colnames, in_existing, data_state, results_data_state])
    in_ref.change(fn = initial_data_load, inputs=[in_ref], outputs=[output_summary, in_refcol, in_joincol, ref_data_state, ref_results_data_state])      

    match_btn.click(fn = run_matcher, inputs=[in_text, in_file, in_ref, data_state, results_data_state, ref_data_state, in_colnames, in_refcol, in_joincol, in_existing, in_api, in_api_key],
                    outputs=[output_summary, output_file], api_name="address")
    
# Simple run for HF spaces or local on your computer
#block.queue().launch(debug=True) # root_path="/address-match", debug=True, server_name="0.0.0.0",

# Simple run for AWS server
block.queue().launch(ssl_verify=False) # root_path="/address-match", debug=True, server_name="0.0.0.0", server_port=7861

# Download OpenSSL from here: 
# Running on local server with https: https://discuss.huggingface.co/t/how-to-run-gradio-with-0-0-0-0-and-https/38003 or https://dev.to/rajshirolkar/fastapi-over-https-for-development-on-windows-2p7d
#block.queue().launch(ssl_verify=False, share=False, debug=False, server_name="0.0.0.0",server_port=443,
#                     ssl_certfile="cert.pem", ssl_keyfile="key.pem") # port 443 for https. Certificates currently not valid

# Running on local server without https
#block.queue().launch(server_name="0.0.0.0", server_port=7861, ssl_verify=False)

