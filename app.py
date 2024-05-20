import os
from datetime import datetime
from pathlib import Path
import gradio as gr
import pandas as pd

from tools.matcher_funcs import run_matcher
from tools.gradio import initial_data_load, ensure_output_folder_exists
from tools.aws_functions import load_data_from_aws

import warnings
# Remove warnings from print statements
warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression')
warnings.filterwarnings("ignore", 'Downcasting behavior')
warnings.filterwarnings("ignore", 'A value is trying to be set on a copy of a slice from a DataFrame')
warnings.filterwarnings("ignore")

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# Base folder is where the code file is stored
base_folder = Path(os.getcwd())
output_folder = "output/"

ensure_output_folder_exists(output_folder)

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
        ## Choose reference file / call API
        Upload a reference file to match against, or alternatively call the Addressbase API (requires API key). Fuzzy matching will work on any address format, but the neural network will only work with the LLPG LPI format, e.g. with columns SaoText, SaoStartNumber etc.. This joins on the UPRN column. If any of these are different for you, 
        open 'Custom reference file format or join columns' below.
        """)
        
        in_ref = gr.File(label="Input reference addresses from file", file_count= "multiple")

        with gr.Accordion("Use Addressbase API instead of reference file", open = False):
            in_api = gr.Dropdown(label="Choose API type", multiselect=False, value=None, choices=["Postcode"])#["Postcode", "UPRN"]) #choices=["Address", "Postcode", "UPRN"])
            in_api_key = gr.Textbox(label="Addressbase API key", type='password')
        
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
block.queue().launch(ssl_verify=False, inbrowser=True) # root_path="/address-match", debug=True, server_name="0.0.0.0", server_port=7861

# Download OpenSSL from here: 
# Running on local server with https: https://discuss.huggingface.co/t/how-to-run-gradio-with-0-0-0-0-and-https/38003 or https://dev.to/rajshirolkar/fastapi-over-https-for-development-on-windows-2p7d
#block.queue().launch(ssl_verify=False, share=False, debug=False, server_name="0.0.0.0",server_port=443,
#                     ssl_certfile="cert.pem", ssl_keyfile="key.pem") # port 443 for https. Certificates currently not valid

# Running on local server without https
#block.queue().launch(server_name="0.0.0.0", server_port=7861, ssl_verify=False)

