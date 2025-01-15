import os
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Type
import time
import re
import math
from datetime import datetime
import copy
import gradio as gr

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]
MatchedResults = Dict[str,Tuple[str,int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")
today_month_rev = datetime.now().strftime("%Y%m")

# Constants
run_fuzzy_match = True
run_nnet_match = True
run_standardise = True

from tools.constants import *
from tools.preparation import prepare_search_address_string, prepare_search_address,  extract_street_name, prepare_ref_address, remove_non_postal, check_no_number_addresses
from tools.fuzzy_match import string_match_by_post_code_multiple, _create_fuzzy_match_results_output, join_to_orig_df
from tools.standardise import standardise_wrapper_func

# Neural network functions
### Predict function for imported model
from tools.model_predict import full_predict_func, full_predict_torch, post_predict_clean
from tools.recordlinkage_funcs import score_based_match
from tools.helper_functions import initial_data_load, sum_numbers_before_seconds

# API functions
from tools.addressbase_api_funcs import places_api_query

# Maximum number of neural net predictions in a single batch
from tools.constants import max_predict_len, MatcherClass


# Load in data functions

def detect_file_type(filename:str) -> str:
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename:str) -> PandasDataFrame:
    """Read the file based on its detected type and convert to Pandas Dataframe. Supports csv, xlsx, and parquet."""
    file_type = detect_file_type(filename)
    
    if file_type == 'csv':
        return pd.read_csv(filename, low_memory=False)
    elif file_type == 'xlsx':
        return pd.read_excel(filename)
    elif file_type == 'parquet':
        return pd.read_parquet(filename)

def get_file_name(in_name: str) -> str:
    """Get the name of a file from a string, handling both Windows and Unix paths."""

    print("in_name: ", in_name)
    match = re.search(rf'{re.escape(os.sep)}(?!.*{re.escape(os.sep)})(.*)', in_name) 
    if match:
        matched_result = match.group(1)
    else:
        matched_result = None

    print("Matched result: ", matched_result)
    
    return matched_result

def filter_not_matched(
    matched_results: pd.DataFrame, 
    search_df: pd.DataFrame, 
    key_col: str
) -> pd.DataFrame:
    """Filters search_df to only rows with key_col not in matched_results"""
    
    # Validate inputs
    if not isinstance(matched_results, pd.DataFrame):
        raise TypeError("not_matched_results must be a Pandas DataFrame")
        
    if not isinstance(search_df, pd.DataFrame):
        raise TypeError("search_df must be a Pandas DataFrame")
        
    if not isinstance(key_col, str):
        raise TypeError("key_col must be a string")
        
    if key_col not in matched_results.columns:
        raise ValueError(f"{key_col} not a column in matched_results")
        
    matched_results_success = matched_results[matched_results["full_match"]==True]

    # Filter search_df
    
    matched = search_df[key_col].astype(str).isin(matched_results_success[key_col].astype(str))
    
    return search_df.iloc[np.where(~matched)[0]]

def query_addressbase_api(in_api_key:str, Matcher:MatcherClass, query_type:str, progress=gr.Progress()):
    
    final_api_output_file_name = ""

    if in_api_key == "":
        print ("No API key provided, please provide one to continue")
        return Matcher, final_api_output_file_name
    else:
        # Call the API
        #Matcher.ref_df = pd.DataFrame()

        # Check if the ref_df file already exists
        def check_and_create_api_folder():
            # Check if the environmental variable is available
            file_path = os.environ.get('ADDRESSBASE_API_OUT') 

            if file_path is None:
                # Environmental variable is not set
                print("API output environmental variable not set.")
                # Create the 'api/' folder if it doesn't already exist
                api_folder_path = 'api/'
                if not os.path.exists(api_folder_path):
                    os.makedirs(api_folder_path)
                print(f"'{api_folder_path}' folder created.")
            else:
                # Environmental variable is set
                api_folder_path = file_path
                print(f"Environmental variable found: {api_folder_path}")
            
            return api_folder_path

        api_output_folder = check_and_create_api_folder()

        # Check if the file exists
        #print("Matcher file name: ", Matcher.file_name)
        search_file_name_without_extension = re.sub(r'\.[^.]+$', '', Matcher.file_name)

        api_ref_save_loc = api_output_folder + search_file_name_without_extension + "_api_" + today_month_rev + "_" + query_type + "_ckpt"
        print("API reference save location: ", api_ref_save_loc)

        final_api_output_file_name = api_ref_save_loc + ".parquet"

        # Allow for csv, parquet and gzipped csv files
        if os.path.isfile(api_ref_save_loc + ".csv"):
            print("API reference CSV file found")
            Matcher.ref_df = pd.read_csv(api_ref_save_loc + ".csv")
        elif os.path.isfile(final_api_output_file_name):
            print("API reference Parquet file found")
            Matcher.ref_df = pd.read_parquet(api_ref_save_loc + ".parquet")
        elif os.path.isfile(api_ref_save_loc + ".csv.gz"):
            print("API reference gzipped CSV file found")
            Matcher.ref_df = pd.read_csv(api_ref_save_loc + ".csv.gz", compression='gzip')
        else:
            print("API reference file not found, querying API for reference data.")


        def conduct_api_loop(in_query, in_api_key, query_type, i, api_ref_save_loc, loop_list, api_search_index):
            ref_addresses = places_api_query(in_query, in_api_key, query_type)

            ref_addresses['Address_row_number'] = api_search_index[i]
        
            loop_list.append(ref_addresses)

            if (i + 1) % 500 == 0:
                print("Saving api call checkpoint for query:", str(i + 1))
                
                pd.concat(loop_list).to_parquet(output_folder + api_ref_save_loc + ".parquet", index=False)

            return loop_list
        
        def check_postcode(postcode):
            # Remove spaces on the ends or in the middle of the postcode, and any symbols
            cleaned_postcode = re.sub(r'[^\w\s]|[\s]', '', postcode)
            # Ensure that the postcode meets the specified format
            postcode_pattern = r'\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?[0-9][A-Z]{2}|GIR0AA|GIR0A{2}|[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?[0-9]{1}?)\b'
            match = re.match(postcode_pattern, cleaned_postcode)
            if match and len(cleaned_postcode) in (6, 7):
                return cleaned_postcode  # Return the matched postcode string
            else:
                return None  # Return None if no match is found

        if query_type == "Address":
            save_file = True
            # Do an API call for each unique address

            if not Matcher.ref_df.empty:
                api_search_df = Matcher.search_df.copy().drop(list(set(Matcher.ref_df["Address_row_number"])))

            else:
                print("Matcher ref_df data empty")
                api_search_df = Matcher.search_df.copy()
            
            i = 0
            loop_df = Matcher.ref_df
            loop_list = [Matcher.ref_df]

            for address in progress.tqdm(api_search_df['full_address_postcode'], desc= "Making API calls", unit="addresses", total=len(api_search_df['full_address_postcode'])):
                print("Query number: " + str(i+1), "with address: ", address)

                api_search_index = api_search_df.index

                loop_list = conduct_api_loop(address, in_api_key, query_type, i, api_ref_save_loc, loop_list, api_search_index)

                i += 1

            loop_df = pd.concat(loop_list)
            Matcher.ref_df = loop_df.drop_duplicates(keep='first', ignore_index=True)
        

        elif query_type == "Postcode":
            save_file = True
            # Do an API call for each unique postcode. Each API call can only return 100 results maximum :/

            if not Matcher.ref_df.empty:
                print("Excluding postcodes that already exist in API call data.")

                # Retain original index values after filtering
                Matcher.search_df["index_keep"] = Matcher.search_df.index
                
                if 'invalid_request' in Matcher.ref_df.columns and 'Address_row_number' in Matcher.ref_df.columns:
                    print("Joining on invalid_request column")
                    Matcher.search_df = Matcher.search_df.merge(Matcher.ref_df[['Address_row_number', 'invalid_request']].drop_duplicates(subset="Address_row_number"), left_on = Matcher.search_df_key_field, right_on='Address_row_number', how='left')

                elif not 'invalid_request' in Matcher.search_df.columns:
                    Matcher.search_df['invalid_request'] = False

                postcode_col = Matcher.search_postcode_col[0]
                
                # Check ref_df df against cleaned and non-cleaned postcodes
                Matcher.search_df[postcode_col] = Matcher.search_df[postcode_col].astype(str)
                search_df_cleaned_pcodes = Matcher.search_df[postcode_col].apply(check_postcode)
                ref_df_cleaned_pcodes = Matcher.ref_df['POSTCODE_LOCATOR'].dropna().apply(check_postcode)

                api_search_df = Matcher.search_df.copy().loc[
                    ~Matcher.search_df[postcode_col].isin(Matcher.ref_df['POSTCODE_LOCATOR']) &
                    ~(Matcher.search_df['invalid_request']==True) &
                    ~(search_df_cleaned_pcodes.isin(ref_df_cleaned_pcodes)), :]
                
                #api_search_index = api_search_df["index_keep"]
                #api_search_df.index = api_search_index
                
                print("Remaining invalid request count: ", Matcher.search_df['invalid_request'].value_counts())
                
            else:
                print("Matcher ref_df data empty")
                api_search_df = Matcher.search_df.copy()
                api_search_index = api_search_df.index
                api_search_df['index_keep'] = api_search_index

                postcode_col = Matcher.search_postcode_col[0]

            unique_pcodes = api_search_df.loc[:, ["index_keep", postcode_col]].drop_duplicates(subset=[postcode_col], keep='first')
            print("Unique postcodes: ", unique_pcodes[postcode_col])

            # Apply the function to each postcode in the Series
            unique_pcodes["cleaned_unique_postcodes"] = unique_pcodes[postcode_col].apply(check_postcode)

            # Filter out the postcodes that comply with the specified format
            valid_unique_postcodes = unique_pcodes.dropna(subset=["cleaned_unique_postcodes"])

            valid_postcode_search_index = valid_unique_postcodes['index_keep']
            valid_postcode_search_index_list = valid_postcode_search_index.tolist()

            if not valid_unique_postcodes.empty:

                print("Unique valid postcodes: ", valid_unique_postcodes)
                print("Number of unique valid postcodes: ", len(valid_unique_postcodes))

                tic = time.perf_counter()

                i = 0
                loop_df = Matcher.ref_df
                loop_list = [Matcher.ref_df]

                for pcode in progress.tqdm(valid_unique_postcodes["cleaned_unique_postcodes"], desc= "Making API calls", unit="unique postcodes", total=len(valid_unique_postcodes["cleaned_unique_postcodes"])):
                    #api_search_index = api_search_df.index
                    
                    print("Query number: " + str(i+1), " with postcode: ", pcode, " and index: ", valid_postcode_search_index_list[i])

                    loop_list = conduct_api_loop(pcode, in_api_key, query_type, i, api_ref_save_loc, loop_list, valid_postcode_search_index_list)
                    
                    i += 1
                    
                loop_df = pd.concat(loop_list)
                Matcher.ref_df = loop_df.drop_duplicates(keep='first', ignore_index=True)

                toc = time.perf_counter()
                print("API call time in seconds: ", toc-tic)
            else:
                print("No valid postcodes found.")

        elif query_type == "UPRN":
            save_file = True
            # Do an API call for each unique address

            if not Matcher.ref_df.empty:
                api_search_df = Matcher.search_df.copy().drop(list(set(Matcher.ref_df["Address_row_number"])))

            else:
                print("Matcher ref_df data empty")
                api_search_df = Matcher.search_df.copy()
            
            i = 0
            loop_df = Matcher.ref_df
            loop_list = [Matcher.ref_df]
            uprn_check_col = 'ADR_UPRN'

            for uprn in progress.tqdm(api_search_df[uprn_check_col], desc= "Making API calls", unit="UPRNs", total=len(api_search_df[uprn_check_col])):
                print("Query number: " + str(i+1), "with address: ", uprn)

                api_search_index = api_search_df.index

                loop_list = conduct_api_loop(uprn, in_api_key, query_type, i, api_ref_save_loc, loop_list, api_search_index)

                i += 1

            loop_df = pd.concat(loop_list)
            Matcher.ref_df = loop_df.drop_duplicates(keep='first', ignore_index=True)

        else:
            print("Reference file loaded from file, no API calls made.")
            save_file = False

        # Post API call processing

        Matcher.ref_name = "API"
        #Matcher.ref_df = Matcher.ref_df.reset_index(drop=True)
        Matcher.ref_df['Reference file'] = Matcher.ref_name

        if query_type == "Postcode":
            #print(Matcher.ref_df.columns)

            cols_of_interest = ["ADDRESS",	"ORGANISATION",	"SAO_TEXT", "SAO_START_NUMBER", "SAO_START_SUFFIX", "SAO_END_NUMBER", "SAO_END_SUFFIX", "PAO_TEXT",	"PAO_START_NUMBER", "PAO_START_SUFFIX", "PAO_END_NUMBER", "PAO_END_SUFFIX", "STREET_DESCRIPTION", "TOWN_NAME"	,"ADMINISTRATIVE_AREA", "LOCALITY_NAME", "POSTCODE_LOCATOR", "UPRN", "PARENT_UPRN",	"USRN",	"LPI_KEY",	"RPC",	"LOGICAL_STATUS_CODE",	"CLASSIFICATION_CODE",	"LOCAL_CUSTODIAN_CODE",	"COUNTRY_CODE",	"POSTAL_ADDRESS_CODE",	"BLPU_STATE_CODE",	"LAST_UPDATE_DATE",	"ENTRY_DATE",	"STREET_STATE_CODE",	"STREET_CLASSIFICATION_CODE",	"LPI_LOGICAL_STATUS_CODE", "invalid_request",	"Address_row_number",	"Reference file"]

            try:
                # Attempt to select only the columns of interest
                Matcher.ref_df = Matcher.ref_df[cols_of_interest]
            except KeyError as e:
                missing_columns = [col for col in e.args[0][1:-1].split(", ") if col not in cols_of_interest]
                # Handle the missing columns gracefully
                print(f"Some columns are missing: {missing_columns}")

            #if "LOCAL_CUSTODIAN_CODE" in Matcher.ref_df.columns:
                # These are items that are 'owned' by Ordnance Survey like telephone boxes, bus shelters
                # Matcher.ref_df = Matcher.ref_df.loc[Matcher.ref_df["LOCAL_CUSTODIAN_CODE"] != 7655,:] 

        if save_file:
            final_api_output_file_name_pq = output_folder + api_ref_save_loc[:-5] + ".parquet"
            final_api_output_file_name = output_folder + api_ref_save_loc[:-5] + ".csv"
            print("Saving reference file to: " + api_ref_save_loc[:-5] + ".parquet")
            Matcher.ref_df.to_parquet(output_folder + api_ref_save_loc + ".parquet", index=False) # Save checkpoint as well
            Matcher.ref_df.to_parquet(final_api_output_file_name_pq, index=False)
            Matcher.ref_df.to_csv(final_api_output_file_name)

        if Matcher.ref_df.empty:
            print ("No reference data found with API")
            return Matcher
                        
    return Matcher, final_api_output_file_name

def load_ref_data(Matcher:MatcherClass, ref_data_state:PandasDataFrame, in_ref:List[str], in_refcol:List[str], in_api:List[str], in_api_key:str, query_type:str, progress=gr.Progress()):
        '''
        Check for reference address data, do some preprocessing, and load in from the Addressbase API if required.
        '''
        final_api_output_file_name = ""

        # Check if reference data loaded, bring in if already there
        if not ref_data_state.empty:
            Matcher.ref_df = ref_data_state
            Matcher.ref_name = get_file_name(in_ref[0].name)

            if not Matcher.ref_name:
                    Matcher.ref_name = '' 

            Matcher.ref_df["Reference file"] = Matcher.ref_name

        # Otherwise check for file name and load in. If nothing found, fail
        else:
            Matcher.ref_df = pd.DataFrame()
            
            if not in_ref:
                if in_api==False:
                    print ("No reference file provided, please provide one to continue")
                    return Matcher, final_api_output_file_name
                # Check if api call required and api key is provided
                else:
                    Matcher, final_api_output_file_name = query_addressbase_api(in_api_key, Matcher, query_type)

            else:
                Matcher.ref_name = get_file_name(in_ref[0].name)
                if not Matcher.ref_name:
                    Matcher.ref_name = '' 

                # Concatenate all in reference files together
                for ref_file in in_ref:
                    #print(ref_file.name)
                    temp_ref_file = read_file(ref_file.name) 

                    file_name_out = get_file_name(ref_file.name)
                    temp_ref_file["Reference file"] = file_name_out
                    
                    Matcher.ref_df = pd.concat([Matcher.ref_df, temp_ref_file])              

        # For the neural net model to work, the llpg columns have to be in the LPI format (e.g. with columns SaoText, SaoStartNumber etc. Here we check if we have that format.    

        if 'Address_LPI' in Matcher.ref_df.columns:
            Matcher.ref_df = Matcher.ref_df.rename(columns={
            "Name_LPI": "PaoText",    
            "Num_LPI": "PaoStartNumber",
            "Num_Suffix_LPI":"PaoStartSuffix",
            "Number End_LPI":"PaoEndNumber",
            "Number_End_Suffix_LPI":"PaoEndSuffix",

            "Secondary_Name_LPI":"SaoText",
            "Secondary_Num_LPI":"SaoStartNumber",
            "Secondary_Num_Suffix_LPI":"SaoStartSuffix",
            "Secondary_Num_End_LPI":"SaoEndNumber",
            "Secondary_Num_End_Suffix_LPI":"SaoEndSuffix",
            "Postcode_LPI":"Postcode",
            "Postal_Town_LPI":"PostTown",
            "UPRN_BLPU": "UPRN"
        })
            
        #print("Matcher reference file: ", Matcher.ref_df['Reference file'])
            
        # Check if the source is the Addressbase places API
        if Matcher.ref_df is not None and len(Matcher.ref_df) > 0:
            first_row = Matcher.ref_df.iloc[0]
            #print(first_row)
            if first_row is not None and 'Reference file' in first_row:
                if first_row['Reference file'] == 'API' or '_api_' in first_row['Reference file']:
                    Matcher.ref_df = Matcher.ref_df.rename(columns={
                    "ORGANISATION_NAME": "Organisation",
                    "ORGANISATION": "Organisation",
                    "PAO_TEXT": "PaoText",    
                    "PAO_START_NUMBER": "PaoStartNumber",
                    "PAO_START_SUFFIX":"PaoStartSuffix",
                    "PAO_END_NUMBER":"PaoEndNumber",
                    "PAO_END_SUFFIX":"PaoEndSuffix",
                    "STREET_DESCRIPTION":"Street",
                    
                    "SAO_TEXT":"SaoText",
                    "SAO_START_NUMBER":"SaoStartNumber",
                    "SAO_START_SUFFIX":"SaoStartSuffix",
                    "SAO_END_NUMBER":"SaoEndNumber",
                    "SAO_END_SUFFIX":"SaoEndSuffix",
                    
                    "POSTCODE_LOCATOR":"Postcode",
                    "TOWN_NAME":"PostTown",
                    "LOCALITY_NAME":"LocalityName",
                    "ADMINISTRATIVE_AREA":"AdministrativeArea"
                    }, errors="ignore")
    
        # Check ref_df file format
        # If standard format, or it's an API call
        if 'SaoText' in Matcher.ref_df.columns or in_api:
            Matcher.standard_llpg_format = True
            Matcher.ref_address_cols = ["Organisation", "SaoStartNumber", "SaoStartSuffix", "SaoEndNumber", "SaoEndSuffix", "SaoText", "PaoStartNumber", "PaoStartSuffix", "PaoEndNumber",
            "PaoEndSuffix", "PaoText", "Street", "PostTown", "Postcode"]
            # Add columns from the list if they don't exist
            for col in Matcher.ref_address_cols:
                if col not in Matcher.ref_df:
                    Matcher.ref_df[col] = np.nan
        else: 
            Matcher.standard_llpg_format = False
            Matcher.ref_address_cols = in_refcol
            Matcher.ref_df = Matcher.ref_df.rename(columns={Matcher.ref_address_cols[-1]:"Postcode"})
            Matcher.ref_address_cols[-1] = "Postcode"


        # Reset index for ref_df as multiple files may have been combined with identical indices
        Matcher.ref_df = Matcher.ref_df.reset_index() #.drop(["index","level_0"], axis = 1, errors="ignore").reset_index().drop(["index","level_0"], axis = 1, errors="ignore")
        Matcher.ref_df.index.name = 'index'

        return Matcher, final_api_output_file_name

def load_match_data_and_filter(Matcher:MatcherClass, data_state:PandasDataFrame, results_data_state:PandasDataFrame, in_file:List[str], in_text:str, in_colnames:List[str], in_joincol:List[str], in_existing:List[str], in_api:List[str]):
    '''
    Check if data to be matched exists. Filter it according to which records are relevant in the reference dataset
    '''

    # Assign join field if not known
    if not Matcher.search_df_key_field:
            Matcher.search_df_key_field = "index"

    # Set search address cols as entered column names
    #print("In colnames in check match data: ", in_colnames)
    Matcher.search_address_cols = in_colnames

    # Check if data loaded already and bring it in
    if not data_state.empty:
        
        Matcher.search_df = data_state        
        Matcher.search_df['index'] = Matcher.search_df.reset_index().index

    else:        
        Matcher.search_df = pd.DataFrame()       

    # If a single address entered into the text box, just load this instead
    if in_text:
        Matcher.search_df, Matcher.search_df_key_field, Matcher.search_address_cols, Matcher.search_postcode_col = prepare_search_address_string(in_text) 

    # If no file loaded yet and a file has been selected
    if Matcher.search_df.empty and in_file:
        output_message, drop1, drop2, Matcher.search_df, results_data_state = initial_data_load(in_file)

        file_list = [string.name for string in in_file]
        data_file_names = [string for string in file_list if "results_" not in string.lower()]

        Matcher.file_name = get_file_name(data_file_names[0])
        
        # search_df makes column to use as index
        Matcher.search_df['index'] = Matcher.search_df.index


    # Join previously created results file onto search_df if previous results file exists
    if not results_data_state.empty:

        print("Joining on previous results file")
        Matcher.results_on_orig_df = results_data_state.copy()
        Matcher.search_df = Matcher.search_df.merge(results_data_state, on = "index", how = "left") 

    # If no join on column suggested, assume the user wants the UPRN
    if not in_joincol:
        Matcher.new_join_col = ['UPRN']
        
    else:  
        Matcher.new_join_col = in_joincol

    if len(in_colnames) > 1:
        Matcher.search_postcode_col = [in_colnames[-1]]

        #print("Postcode col: ", Matcher.search_postcode_col)
        
    elif len(in_colnames) == 1:
        Matcher.search_df['full_address_postcode'] = Matcher.search_df[in_colnames[0]]
        Matcher.search_postcode_col = ['full_address_postcode']
        Matcher.search_address_cols.append('full_address_postcode')

    # Check for column that indicates there are existing matches. The code will then search this column for entries, and will remove them from the data to be searched
    Matcher.existing_match_cols = in_existing

    if in_existing:
        if "Matched with reference address" in Matcher.search_df.columns:
            Matcher.search_df.loc[~Matcher.search_df[in_existing].isna(), "Matched with reference address"] = True
        else: Matcher.search_df["Matched with reference address"] = ~Matcher.search_df[in_existing].isna()
            
    print("Shape of search_df before filtering is: ", Matcher.search_df.shape)

    ### Filter addresses to those with length > 0
    zero_length_search_df = Matcher.search_df.copy()[Matcher.search_address_cols]
    zero_length_search_df = zero_length_search_df.fillna('').infer_objects(copy=False)
    Matcher.search_df["address_cols_joined"] = zero_length_search_df.astype(str).sum(axis=1).str.strip()

    length_more_than_0 = Matcher.search_df["address_cols_joined"].str.len() > 0

    ### Filter addresses to match to postcode areas present in both search_df and ref_df_cleaned only (postcode without the last three characters). Only run if API call is false. When the API is called, relevant addresses and postcodes should be brought in by the API.
    if not in_api:
        if Matcher.filter_to_lambeth_pcodes == True:
            Matcher.search_df["postcode_search_area"] = Matcher.search_df[Matcher.search_postcode_col[0]].str.strip().str.upper().str.replace(" ", "").str[:-2]
            Matcher.ref_df["postcode_search_area"] = Matcher.ref_df["Postcode"].str.strip().str.upper().str.replace(" ", "").str[:-2]
            
            unique_ref_pcode_area = (Matcher.ref_df["postcode_search_area"][Matcher.ref_df["postcode_search_area"].str.len() > 3]).unique()
            postcode_found_in_search = Matcher.search_df["postcode_search_area"].isin(unique_ref_pcode_area)

            Matcher.search_df["Excluded from search"] = "Included in search"
            Matcher.search_df.loc[~(postcode_found_in_search), "Excluded from search"] = "Postcode area not found"
            Matcher.search_df.loc[~(length_more_than_0), "Excluded from search"] = "Address length 0"
            Matcher.pre_filter_search_df = Matcher.search_df.copy()#.drop(["index", "level_0"], axis = 1, errors = "ignore").reset_index()
            Matcher.pre_filter_search_df = Matcher.pre_filter_search_df.drop("address_cols_joined", axis = 1)

            Matcher.excluded_df = Matcher.search_df.copy()[~(postcode_found_in_search) | ~(length_more_than_0)]
            Matcher.search_df = Matcher.search_df[(postcode_found_in_search) & (length_more_than_0)]

            
            # Exclude records that have already been matched separately, i.e. if 'Matched with reference address' column exists, and has trues in it
            if "Matched with reference address" in Matcher.search_df.columns:
                previously_matched = Matcher.pre_filter_search_df["Matched with reference address"] == True 
                Matcher.pre_filter_search_df.loc[previously_matched, "Excluded from search"] = "Previously matched"
                
                Matcher.excluded_df = Matcher.search_df.copy()[~(postcode_found_in_search) | ~(length_more_than_0) | (previously_matched)]
                Matcher.search_df = Matcher.search_df[(postcode_found_in_search) & (length_more_than_0) & ~(previously_matched)]

            else:
                Matcher.excluded_df = Matcher.search_df.copy()[~(postcode_found_in_search) | ~(length_more_than_0)]
                Matcher.search_df = Matcher.search_df[(postcode_found_in_search) & (length_more_than_0)]

            print("Shape of ref_df before filtering is: ", Matcher.ref_df.shape)   

            unique_search_pcode_area = (Matcher.search_df["postcode_search_area"]).unique()
            postcode_found_in_ref = Matcher.ref_df["postcode_search_area"].isin(unique_search_pcode_area)
            Matcher.ref_df = Matcher.ref_df[postcode_found_in_ref]

            Matcher.pre_filter_search_df = Matcher.pre_filter_search_df.drop("postcode_search_area", axis = 1)
            Matcher.search_df = Matcher.search_df.drop("postcode_search_area", axis = 1)
            Matcher.ref_df = Matcher.ref_df.drop("postcode_search_area", axis = 1)
            Matcher.excluded_df = Matcher.excluded_df.drop("postcode_search_area", axis = 1)
        else:
            Matcher.pre_filter_search_df = Matcher.search_df.copy()
            Matcher.search_df.loc[~(length_more_than_0), "Excluded from search"] = "Address length 0"
            
            Matcher.excluded_df = Matcher.search_df[~(length_more_than_0)]
            Matcher.search_df = Matcher.search_df[length_more_than_0]


    Matcher.search_df = Matcher.search_df.drop("address_cols_joined", axis = 1, errors="ignore")
    Matcher.excluded_df = Matcher.excluded_df.drop("address_cols_joined", axis = 1, errors="ignore")

    Matcher.search_df_not_matched = Matcher.search_df

    # If this is for an API call, we need to convert the search_df address columns to one column now. This is so the API call can be made and the reference dataframe created.
    if in_api:

        if in_file:
            output_message, drop1, drop2, df, results_data_state = initial_data_load(in_file)

            file_list = [string.name for string in in_file]
            data_file_names = [string for string in file_list if "results_" not in string.lower()]
        
            Matcher.file_name = get_file_name(data_file_names[0])

            print("File list in in_api bit: ", file_list)
            print("data_file_names in in_api bit: ", data_file_names)
            print("Matcher.file_name in in_api bit: ", Matcher.file_name)

        else:
            if in_text:
                Matcher.file_name = in_text
            else:
                Matcher.file_name = "API call"

        # Exclude records that have already been matched separately, i.e. if 'Matched with reference address' column exists, and has trues in it
        if in_existing:
            print("Checking for previously matched records")
            Matcher.pre_filter_search_df = Matcher.search_df.copy()
            previously_matched = ~Matcher.pre_filter_search_df[in_existing].isnull()
            Matcher.pre_filter_search_df.loc[previously_matched, "Excluded from search"] = "Previously matched"
            
            Matcher.excluded_df = Matcher.search_df.copy()[~(length_more_than_0) | (previously_matched)]
            Matcher.search_df = Matcher.search_df[(length_more_than_0) & ~(previously_matched)]

        if type(Matcher.search_df) == str: search_df_cleaned, search_df_key_field, search_address_cols = prepare_search_address_string(Matcher.search_df)
        else: search_df_cleaned = prepare_search_address(Matcher.search_df, Matcher.search_address_cols, Matcher.search_postcode_col, Matcher.search_df_key_field)


        Matcher.search_df['full_address_postcode'] = search_df_cleaned["full_address"]

    return Matcher

def load_matcher_data(in_text, in_file, in_ref, data_state, results_data_state, ref_data_state, in_colnames, in_refcol, in_joincol, in_existing, Matcher, in_api, in_api_key):
        '''
        Load in user inputs from the Gradio interface. Convert all input types (single address, or csv input) into standardised data format that can be used downstream for the fuzzy matching.
        '''
        final_api_output_file_name = ""

        today_rev = datetime.now().strftime("%Y%m%d")

        # Abort flag for if it's not even possible to attempt the first stage of the match for some reason
        Matcher.abort_flag = False

        ### ref_df FILES ###
        # If not an API call, run this first
        if not in_api:
            Matcher, final_api_output_file_name = load_ref_data(Matcher, ref_data_state, in_ref, in_refcol, in_api, in_api_key, query_type=in_api)

        ### MATCH/SEARCH FILES ###
        # If doing API calls, we need to know the search data before querying for specific addresses/postcodes
        Matcher = load_match_data_and_filter(Matcher, data_state, results_data_state, in_file, in_text, in_colnames, in_joincol, in_existing, in_api)

        # If an API call, ref_df data is loaded after
        if in_api:
            Matcher, final_api_output_file_name = load_ref_data(Matcher, ref_data_state, in_ref, in_refcol, in_api, in_api_key, query_type=in_api)
            
        print("Shape of ref_df after filtering is: ", Matcher.ref_df.shape)
        print("Shape of search_df after filtering is: ", Matcher.search_df.shape)
    
        Matcher.match_outputs_name = output_folder + "diagnostics_initial_" + today_rev + ".csv" 
        Matcher.results_orig_df_name = output_folder + "results_initial_" + today_rev + ".csv" 
    
        Matcher.match_results_output.to_csv(Matcher.match_outputs_name, index = None)
        Matcher.results_on_orig_df.to_csv(Matcher.results_orig_df_name, index = None)
        
        return Matcher, final_api_output_file_name

# Run whole matcher process
def run_matcher(in_text:str, in_file:str, in_ref:str, data_state:PandasDataFrame, results_data_state:PandasDataFrame, ref_data_state:PandasDataFrame, in_colnames:List[str], in_refcol:List[str], in_joincol:List[str], in_existing:List[str], in_api:str, in_api_key:str, InitMatch:MatcherClass = InitMatch, progress=gr.Progress()):  
    '''
    Split search and reference data into batches. Loop and run through the match script for each batch of data.
    '''
    output_files = []

    estimate_total_processing_time = 0.0

    overall_tic = time.perf_counter()

    # Load in initial data. This will filter to relevant addresses in the search and reference datasets that can potentially be matched, and will pull in API data if asked for.
    InitMatch, final_api_output_file_name = load_matcher_data(in_text, in_file, in_ref, data_state, results_data_state, ref_data_state, in_colnames, in_refcol, in_joincol, in_existing, InitMatch, in_api, in_api_key)

    if final_api_output_file_name:
        output_files.append(final_api_output_file_name)

    if InitMatch.search_df.empty or InitMatch.ref_df.empty:
        out_message = "Nothing to match!"
        print(out_message)

        output_files.extend([InitMatch.results_orig_df_name, InitMatch.match_outputs_name])
        return out_message, output_files, estimate_total_processing_time
    
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
        
        BatchMatch = copy.copy(InitMatch)

        # Subset the search and reference dfs based on current batch ranges
        BatchMatch.search_df = BatchMatch.search_df[BatchMatch.search_df.index.isin(search_range)].reset_index(drop=True)
        BatchMatch.search_df_not_matched = BatchMatch.search_df.copy()
        BatchMatch.search_df_cleaned = BatchMatch.search_df_cleaned[BatchMatch.search_df_cleaned.index.isin(search_range)].reset_index(drop=True)

        BatchMatch.ref_df = BatchMatch.ref_df[BatchMatch.ref_df.index.isin(ref_range)].reset_index(drop=True)
        BatchMatch.ref_df_cleaned = BatchMatch.ref_df_cleaned[BatchMatch.ref_df_cleaned.index.isin(ref_range)].reset_index(drop=True)

        # Dataframes after standardisation process
        BatchMatch.search_df_after_stand = BatchMatch.search_df_after_stand[BatchMatch.search_df_after_stand.index.isin(search_range)].reset_index(drop=True)
        BatchMatch.search_df_after_full_stand = BatchMatch.search_df_after_full_stand[BatchMatch.search_df_after_full_stand.index.isin(search_range)].reset_index(drop=True)

        ### Create lookup lists for fuzzy matches
        BatchMatch.ref_df_after_stand = BatchMatch.ref_df_after_stand[BatchMatch.ref_df_after_stand.index.isin(ref_range)].reset_index(drop=True)
        BatchMatch.ref_df_after_full_stand = BatchMatch.ref_df_after_full_stand[BatchMatch.ref_df_after_full_stand.index.isin(ref_range)].reset_index(drop=True)

        # Match the data, unless the search or reference dataframes are empty
        if BatchMatch.search_df.empty or BatchMatch.ref_df.empty:
            out_message = "Nothing to match for batch: " + str(n)
            print(out_message)
            BatchMatch_out = BatchMatch
            BatchMatch_out.results_on_orig_df = pd.DataFrame(data={"index":BatchMatch.search_df.index,
                                                                   "Excluded from search":False,
                                                                    "Matched with reference address":False})
        else:
            summary_of_summaries, BatchMatch_out = run_single_match_batch(BatchMatch, n, number_of_batches)

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

    

    estimate_total_processing_time = sum_numbers_before_seconds(time_out)
    print("Estimated total processing time:", str(estimate_total_processing_time))

    output_files.extend([OutputMatch.results_orig_df_name, OutputMatch.match_outputs_name])
    return final_summary, output_files, estimate_total_processing_time

# Run a match run for a single batch
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

def run_single_match_batch(InitialMatch:MatcherClass, batch_n:int, total_batches:int, progress=gr.Progress()):
    '''
    Over-arching function for running a single batch of data through the full matching process. Calls fuzzy matching, then neural network match functions in order. It outputs a summary of the match, and a MatcherClass with the matched data included.
    '''

    if run_fuzzy_match == True:
    
        overall_tic = time.perf_counter()
        
        progress(0, desc= "Batch " + str(batch_n+1) + " of " + str(total_batches) + ". Fuzzy match - non-standardised dataset")
        df_name = "Fuzzy not standardised"
                                    
        ''' FUZZY MATCHING '''
            
        ''' Run fuzzy match on non-standardised dataset '''
        
        FuzzyNotStdMatch = orchestrate_single_match_batch(Matcher = copy.copy(InitialMatch), standardise = False, nnet = False, file_stub= "not_std_", df_name = df_name)

        if FuzzyNotStdMatch.abort_flag == True:
            message = "Nothing to match! Aborting address check."
            print(message)
            return message, InitialMatch
        
        FuzzyNotStdMatch = combine_two_matches(InitialMatch, FuzzyNotStdMatch, df_name)
        
        if (len(FuzzyNotStdMatch.search_df_not_matched) == 0) | (sum(FuzzyNotStdMatch.match_results_output[FuzzyNotStdMatch.match_results_output['full_match']==False]['fuzzy_score'])==0): 
            overall_toc = time.perf_counter()
            time_out = f"The fuzzy match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNotStdMatch.output_summary = FuzzyNotStdMatch.output_summary + " Neural net match not attempted. "# + time_out
            return FuzzyNotStdMatch.output_summary, FuzzyNotStdMatch
    
        ''' Run fuzzy match on standardised dataset '''
        
        progress(.25, desc="Batch " + str(batch_n+1) + " of " + str(total_batches) + ". Fuzzy match - standardised dataset")
        df_name = "Fuzzy standardised"
        
        FuzzyStdMatch = orchestrate_single_match_batch(Matcher = copy.copy(FuzzyNotStdMatch), standardise = True, nnet = False, file_stub= "std_", df_name = df_name)
        FuzzyStdMatch = combine_two_matches(FuzzyNotStdMatch, FuzzyStdMatch, df_name)
    
        ''' Continue if reference file in correct format, and neural net model exists. Also if data not too long '''
        if ((len(FuzzyStdMatch.search_df_not_matched) == 0) | (FuzzyStdMatch.standard_llpg_format == False) |\
            (os.path.exists(FuzzyStdMatch.model_dir_name + '/saved_model.zip') == False) | (run_nnet_match == False)):
            overall_toc = time.perf_counter()
            time_out = f"The fuzzy match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyStdMatch.output_summary = FuzzyStdMatch.output_summary + " Neural net match not attempted. "# + time_out
            return FuzzyStdMatch.output_summary, FuzzyStdMatch

    if run_nnet_match == True:
    
        ''' NEURAL NET '''

        if run_fuzzy_match == False:
            FuzzyStdMatch = copy.copy(InitialMatch)
            overall_tic = time.perf_counter()
    
        ''' First on non-standardised addresses '''
        progress(.50, desc="Batch " + str(batch_n+1) + " of " + str(total_batches) + ". Neural net - non-standardised dataset")
        df_name = "Neural net not standardised"
        
        FuzzyNNetNotStdMatch = orchestrate_single_match_batch(Matcher = copy.copy(FuzzyStdMatch), standardise = False, nnet = True, file_stub= "nnet_not_std_", df_name = df_name)
        FuzzyNNetNotStdMatch = combine_two_matches(FuzzyStdMatch, FuzzyNNetNotStdMatch, df_name)
    
        if (len(FuzzyNNetNotStdMatch.search_df_not_matched) == 0):
            overall_toc = time.perf_counter()
            time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNNetNotStdMatch.output_summary = FuzzyNNetNotStdMatch.output_summary# + time_out
            return FuzzyNNetNotStdMatch.output_summary, FuzzyNNetNotStdMatch
    
        ''' Next on standardised addresses '''
        progress(.75, desc="Batch " + str(batch_n+1) + " of " + str(total_batches) + ". Neural net - standardised dataset")
        df_name = "Neural net standardised"
        
        FuzzyNNetStdMatch = orchestrate_single_match_batch(Matcher = copy.copy(FuzzyNNetNotStdMatch), standardise = True, nnet = True, file_stub= "nnet_std_", df_name = df_name)
        FuzzyNNetStdMatch = combine_two_matches(FuzzyNNetNotStdMatch, FuzzyNNetStdMatch, df_name)
 
        if run_fuzzy_match == False:
            overall_toc = time.perf_counter()
            time_out = f"The neural net match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNNetStdMatch.output_summary = FuzzyNNetStdMatch.output_summary + " Only Neural net match attempted. "# + time_out
            return FuzzyNNetStdMatch.output_summary, FuzzyNNetStdMatch
    
    overall_toc = time.perf_counter()
    time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"

    summary_of_summaries = FuzzyNotStdMatch.output_summary + "\n" + FuzzyStdMatch.output_summary + "\n" + FuzzyNNetStdMatch.output_summary + "\n" + time_out

    return summary_of_summaries, FuzzyNNetStdMatch

# Overarching functions
def orchestrate_single_match_batch(Matcher, standardise = False, nnet = False, file_stub= "not_std_", df_name = "Fuzzy not standardised"):

        today_rev = datetime.now().strftime("%Y%m%d")
        
        #print(Matcher.standardise)
        Matcher.standardise = standardise

        if Matcher.search_df_not_matched.empty:
            print("Nothing to match! At start of preparing run.")
            return Matcher
    
        if nnet == False:
            diag_shortlist,\
            diag_best_match,\
            match_results_output,\
            results_on_orig_df,\
            summary,\
            search_address_cols =\
        full_fuzzy_match(Matcher.search_df_not_matched.copy(),
                        Matcher.standardise, 
                        Matcher.search_df_key_field,
                        Matcher.search_address_cols,
                        Matcher.search_df_cleaned,
                        Matcher.search_df_after_stand,
                        Matcher.search_df_after_full_stand,
                        Matcher.ref_df_cleaned,
                        Matcher.ref_df_after_stand,
                        Matcher.ref_df_after_full_stand,                        
                        Matcher.fuzzy_match_limit,
                        Matcher.fuzzy_scorer_used,
                        Matcher.new_join_col)
            if match_results_output.empty: 
                print("Match results empty")
                Matcher.abort_flag = True
                return Matcher    
        
            else:
                Matcher.diag_shortlist = diag_shortlist
                Matcher.diag_best_match = diag_best_match
                Matcher.match_results_output = match_results_output      
            
        else:
            match_results_output,\
            results_on_orig_df,\
            summary,\
            predict_df_nnet =\
            full_nn_match(
                    Matcher.ref_address_cols,
                    Matcher.search_df_not_matched.copy(),
                    Matcher.search_address_cols,
                    Matcher.search_df_key_field,
                    Matcher.standardise, 
                    Matcher.exported_model[0],
                    Matcher.matching_variables,
                    Matcher.text_columns,
                    Matcher.weights,
                    Matcher.fuzzy_method,
                    Matcher.score_cut_off,
                    Matcher.match_results_output.copy(), 
                    Matcher.filter_to_lambeth_pcodes,
                    Matcher.model_type, 
                    Matcher.word_to_index, 
                    Matcher.cat_to_idx, 
                    Matcher.device,
                    Matcher.vocab,
                    Matcher.labels_list,
                    Matcher.search_df_cleaned,
                    Matcher.ref_df_after_stand,
                    Matcher.search_df_after_stand,
                    Matcher.search_df_after_full_stand,
                    Matcher.new_join_col)
            
            if match_results_output.empty: 
                print("Match results empty")
                Matcher.abort_flag = True
                return Matcher
            else:
                Matcher.match_results_output = match_results_output
                Matcher.predict_df_nnet = predict_df_nnet 
        
        # Save to file
        Matcher.results_on_orig_df = results_on_orig_df

        print("Results output in orchestrate match run shape: ", Matcher.results_on_orig_df.shape)

        Matcher.summary = summary
  
        Matcher.output_summary = create_match_summary(Matcher.match_results_output, df_name = df_name)       
        
        Matcher.match_outputs_name = output_folder + "diagnostics_" + file_stub + today_rev + ".csv"
        Matcher.results_orig_df_name = output_folder + "results_" + file_stub + today_rev + ".csv"
    
        Matcher.match_results_output.to_csv(Matcher.match_outputs_name, index = None)
        Matcher.results_on_orig_df.to_csv(Matcher.results_orig_df_name, index = None)
    
        return Matcher 

# Overarching fuzzy match function
def full_fuzzy_match(search_df:PandasDataFrame,
                     standardise:bool,
                     search_df_key_field:str,
                     search_address_cols:List[str],
                     search_df_cleaned:PandasDataFrame,
                     search_df_after_stand:PandasDataFrame,
                     search_df_after_full_stand:PandasDataFrame,
                     ref_df_cleaned:PandasDataFrame,
                     ref_df_after_stand:PandasDataFrame,
                     ref_df_after_full_stand:PandasDataFrame,                     
                     fuzzy_match_limit:float,
                     fuzzy_scorer_used:str,
                     new_join_col:List[str],
                     fuzzy_search_addr_limit:float = 100,
                     filter_to_lambeth_pcodes:bool=False):

    '''
    Compare addresses in a 'search address' dataframe with a 'reference address' dataframe by using fuzzy matching from the rapidfuzz package, blocked by postcode and then street.
    '''

    # Break if search item has length 0
    if search_df.empty:
        out_error = "Nothing to match! Just started fuzzy match."
        print(out_error)
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(), out_error,search_address_cols

    # If standardise is true, replace relevant variables with standardised versions
    if standardise == True: 
        df_name = "standardised address"
        search_df_after_stand = search_df_after_full_stand
        ref_df_after_stand = ref_df_after_full_stand
    else: 
        df_name = "non-standardised address"
    
    # RUN WITH POSTCODE AS A BLOCKER #
    # Fuzzy match against reference addresses
    
    # Remove rows from ref search series where postcode is not found in the search_df
    search_df_after_stand_series = search_df_after_stand.copy().set_index('postcode_search')['search_address_stand'].sort_index()
    ref_df_after_stand_series = ref_df_after_stand.copy().set_index('postcode_search')['ref_address_stand'].sort_index()
   
    ref_df_after_stand_series_checked = ref_df_after_stand_series.copy()[ref_df_after_stand_series.index.isin(search_df_after_stand_series.index.tolist())]

    if len(ref_df_after_stand_series_checked) == 0: 
        print("Nothing relevant in reference data to match!")
        return pd.DataFrame(), pd.DataFrame(),  pd.DataFrame(),pd.DataFrame(),"Nothing relevant in reference data to match!",search_address_cols

    # 'matched' is the list for which every single row is searched for in the reference list (the ref_df).
    
    print("Starting the fuzzy match")
    
    tic = time.perf_counter()
    results = string_match_by_post_code_multiple(match_address_series = search_df_after_stand_series.copy(),
                          reference_address_series = ref_df_after_stand_series_checked,
                          search_limit = fuzzy_search_addr_limit, 
                          scorer_name = fuzzy_scorer_used)

    toc = time.perf_counter()
    print(f"Performed the fuzzy match in {toc - tic:0.1f} seconds")


    # Create result dfs
    match_results_output, diag_shortlist, diag_best_match = _create_fuzzy_match_results_output(results, search_df_after_stand, ref_df_cleaned, ref_df_after_stand, fuzzy_match_limit, search_df_cleaned, search_df_key_field, new_join_col, standardise, blocker_col = "Postcode")
    
    match_results_output['match_method'] = "Fuzzy match - postcode"
    
    search_df_not_matched = filter_not_matched(match_results_output, search_df_after_stand, search_df_key_field)

                        
    # If nothing left to match, break
    if (sum(match_results_output['full_match']==False) == 0) | (sum(match_results_output[match_results_output['full_match']==False]['fuzzy_score'])==0):
        print("Nothing left to match!")
        
        summary = create_match_summary(match_results_output, df_name)
        
        if type(search_df) != str:
            results_on_orig_df = join_to_orig_df(match_results_output, search_df_cleaned, search_df_key_field, new_join_col)
        else: results_on_orig_df = match_results_output

        print("results_on_orig_df in fuzzy_match shape: ", results_on_orig_df.shape)
        
        return diag_shortlist, diag_best_match, match_results_output, results_on_orig_df, summary, search_address_cols
    

    # RUN WITH STREET AS A BLOCKER #
    
    ### Redo with street as blocker
    search_df_after_stand_street = search_df_not_matched.copy()
    search_df_after_stand_street['search_address_stand_w_pcode'] = search_df_after_stand_street['search_address_stand'] + " " + search_df_after_stand_street['postcode_search']
    ref_df_after_stand['ref_address_stand_w_pcode'] = ref_df_after_stand['ref_address_stand'] + " " + ref_df_after_stand['postcode_search']
        
    search_df_after_stand_street['street']= search_df_after_stand_street['full_address_search'].apply(extract_street_name)
    # Exclude non-postal addresses from street-blocked search
    search_df_after_stand_street.loc[search_df_after_stand_street['Excluded from search'] == "Excluded - non-postal address", 'street'] = ""
        
    ### Create lookup lists
    search_df_match_series_street = search_df_after_stand_street.copy().set_index('street')['search_address_stand']
    ref_df_after_stand_series_street = ref_df_after_stand.copy().set_index('Street')['ref_address_stand']
        
    # Remove rows where street is not in ref_df df
    #index_check = ref_df_after_stand_series_street.index.isin(search_df_match_series_street.index)
    #ref_df_after_stand_series_street_checked = ref_df_after_stand_series_street.copy()[index_check == True]

    ref_df_after_stand_series_street_checked = ref_df_after_stand_series_street.copy()[ref_df_after_stand_series_street.index.isin(search_df_match_series_street.index.tolist())]

    # If nothing left to match, break
    if (len(ref_df_after_stand_series_street_checked) == 0) | ((len(search_df_match_series_street) == 0)):
        
        summary = create_match_summary(match_results_output, df_name)
        
        if type(search_df) != str:
            results_on_orig_df = join_to_orig_df(match_results_output, search_df_after_stand, search_df_key_field, new_join_col)
        else: results_on_orig_df = match_results_output
        
        return diag_shortlist, diag_best_match,\
        match_results_output, results_on_orig_df, summary, search_address_cols
    
    print("Starting the fuzzy match with street as blocker")
    
    tic = time.perf_counter()
    results_st = string_match_by_post_code_multiple(match_address_series = search_df_match_series_street.copy(),
                          reference_address_series = ref_df_after_stand_series_street_checked.copy(),
                          search_limit = fuzzy_search_addr_limit, 
                          scorer_name = fuzzy_scorer_used)

    toc = time.perf_counter()

    print(f"Performed the fuzzy match in {toc - tic:0.1f} seconds")
    
    match_results_output_st, diag_shortlist_st, diag_best_match_st = _create_fuzzy_match_results_output(results_st, search_df_after_stand_street, ref_df_cleaned, ref_df_after_stand,\
    fuzzy_match_limit, search_df_cleaned, search_df_key_field, new_join_col, standardise, blocker_col = "Street")
    match_results_output_st['match_method'] = "Fuzzy match - street"

    match_results_output_st_out = combine_dfs_and_remove_dups(match_results_output, match_results_output_st, index_col = search_df_key_field)
        
    match_results_output = match_results_output_st_out
    
    summary = create_match_summary(match_results_output, df_name)

    ### Join URPN back onto orig df

    if type(search_df) != str:
        results_on_orig_df = join_to_orig_df(match_results_output, search_df_cleaned, search_df_key_field, new_join_col)
    else: results_on_orig_df = match_results_output

    print("results_on_orig_df in fuzzy_match shape: ", results_on_orig_df.shape)
        
    return diag_shortlist, diag_best_match, match_results_output, results_on_orig_df, summary, search_address_cols
 
# Overarching NN function
def full_nn_match(ref_address_cols:List[str], 
                  search_df:PandasDataFrame,
                  search_address_cols:List[str],
                  search_df_key_field:str, 
                  standardise:bool,
                  exported_model:list,
                  matching_variables:List[str],
                  text_columns:List[str],
                  weights:dict,
                  fuzzy_method:str,
                  score_cut_off:float,
                  match_results:PandasDataFrame,
                  filter_to_lambeth_pcodes:bool, 
                  model_type:str,
                  word_to_index:dict,
                  cat_to_idx:dict,
                  device:str,
                  vocab:List[str],
                  labels_list:List[str],
                  search_df_cleaned:PandasDataFrame,
                  ref_df_after_stand:PandasDataFrame,
                  search_df_after_stand:PandasDataFrame,
                  search_df_after_full_stand:PandasDataFrame,
                  new_join_col:List[str]):
    '''
    Use a neural network model to partition 'search addresses' into consituent parts in the format of UK Ordnance Survey Land Property Identifier (LPI) addresses. These address components are compared individually against reference addresses in the same format to give an overall match score using the recordlinkage package.
    '''
    
    # Break if search item has length 0
    if search_df.empty:
        out_error = "Nothing to match!"
        print(out_error)
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(), out_error, search_address_cols

    # If it is the standardisation step, or you have come from the fuzzy match area
    if (standardise == True): # | (run_fuzzy_match == True & standardise == False): 
        df_name = "standardised address"

        search_df_after_stand = search_df_after_full_stand     

    else: 
        df_name = "non-standardised address"

    print(search_df_after_stand.shape[0])
    print(ref_df_after_stand.shape[0])

    # Predict on search data to extract LPI address components

    #predict_len = len(search_df_cleaned["full_address"])
    all_columns = list(search_df_cleaned) # Creates list of all column headers
    search_df_cleaned[all_columns] = search_df_cleaned[all_columns].astype(str)
    predict_data = list(search_df_after_stand['search_address_stand'])
    
    ### Run predict function
    print("Starting neural net prediction for " + str(len(predict_data)) + " addresses")

    tic = time.perf_counter()
    
    # Determine the number of chunks
    num_chunks = math.ceil(len(predict_data) / max_predict_len)
    list_out_all = []
    predict_df_all = []
    
    for i in range(num_chunks):
        print("Starting to predict batch " + str(i+ 1) + " of " + str(num_chunks) + " batches.")
        
        start_idx = i * max_predict_len
        end_idx = start_idx + max_predict_len
        
        # Extract the current chunk of data
        chunk_data = predict_data[start_idx:end_idx]

        # Replace blank strings with a single space
        chunk_data = [" " if s in ("") else s for s in chunk_data]
        
        if (model_type == "gru") | (model_type == "lstm"):
            list_out, predict_df = full_predict_torch(model=exported_model, model_type=model_type, 
                input_text=chunk_data, word_to_index=word_to_index, 
                cat_to_idx=cat_to_idx, device=device)
        else:
            list_out, predict_df = full_predict_func(chunk_data, exported_model, vocab, labels_list)
            
        # Append the results
        list_out_all.extend(list_out)
        predict_df_all.append(predict_df)
    
    # Concatenate all the results dataframes
    predict_df_all = pd.concat(predict_df_all, ignore_index=True)
    
    toc = time.perf_counter()
    
    print(f"Performed the NN prediction in {toc - tic:0.1f} seconds")
    
    predict_df = post_predict_clean(predict_df=predict_df_all, orig_search_df=search_df_cleaned, 
        ref_address_cols=ref_address_cols, search_df_key_field=search_df_key_field)

    # Score-based matching between neural net predictions and fuzzy match results

    # Example of recordlinkage package in use: https://towardsdatascience.com/how-to-perform-fuzzy-dataframe-row-matching-with-recordlinkage-b53ca0cb944c

    ## Run with Postcode as blocker column

    blocker_column = ["Postcode"]

    scoresSBM_best_pc, matched_output_SBM_pc = score_based_match(predict_df_search = predict_df.copy(), ref_search = ref_df_after_stand.copy(),
        orig_search_df = search_df_after_stand, matching_variables = matching_variables,
                      text_columns = text_columns, blocker_column = blocker_column, weights = weights, fuzzy_method = fuzzy_method, score_cut_off = score_cut_off, search_df_key_field=search_df_key_field, standardise=standardise, new_join_col=new_join_col)

    if matched_output_SBM_pc.empty:
        error_message = "Match results empty. Exiting neural net match."
        print(error_message)

        return pd.DataFrame(),pd.DataFrame(), error_message, predict_df
    
    else:
        matched_output_SBM_pc["match_method"] = "Neural net - Postcode"
       
        match_results_output_final_pc = combine_dfs_and_remove_dups(match_results, matched_output_SBM_pc, index_col = search_df_key_field)       
        
    summary_pc = create_match_summary(match_results_output_final_pc, df_name = "NNet blocked by Postcode " + df_name)
    print(summary_pc)
    
    ## Run with Street as blocker column

    blocker_column = ["Street"]

    scoresSBM_best_st, matched_output_SBM_st = score_based_match(predict_df_search = predict_df.copy(), ref_search = ref_df_after_stand.copy(), 
                    orig_search_df = search_df_after_stand, matching_variables = matching_variables,
                      text_columns = text_columns, blocker_column = blocker_column, weights = weights, fuzzy_method = fuzzy_method, score_cut_off = score_cut_off, search_df_key_field=search_df_key_field, standardise=standardise, new_join_col=new_join_col)
    
    # If no matching pairs are found in the function above then it returns 0 - below we replace these values with the postcode blocker values (which should almost always find at least one pair unless it's a very unusual situation)
    if (type(matched_output_SBM_st) == int) | matched_output_SBM_st.empty:
        print("Nothing to match for street block")
        
        matched_output_SBM_st = matched_output_SBM_pc
        matched_output_SBM_st["match_method"] = "Neural net - Postcode" #+ standard_label
    else: matched_output_SBM_st["match_method"] = "Neural net - Street" #+ standard_label
 
    ### Join together old match df with new (model) match df

    match_results_output_final_st = combine_dfs_and_remove_dups(match_results_output_final_pc,matched_output_SBM_st, index_col = search_df_key_field)
      
    summary_street = create_match_summary(match_results_output_final_st, df_name = "NNet blocked by Street " + df_name)
    print(summary_street)

    # I decided in the end not to use PaoStartNumber as a blocker column. I get only a couple more matches in general for a big increase in processing time

    matched_output_SBM_po = matched_output_SBM_st
    matched_output_SBM_po["match_method"] = "Neural net - Street" #+ standard_label
    
    match_results_output_final_po = match_results_output_final_st
    match_results_output_final_three = match_results_output_final_po
    
    summary_three = create_match_summary(match_results_output_final_three, df_name = "fuzzy and nn model street + postcode " + df_name)
   
    ### Join URPN back onto orig df

    if type(search_df) != str:
        results_on_orig_df = join_to_orig_df(match_results_output_final_three, search_df_after_stand, search_df_key_field, new_join_col)
    else: results_on_orig_df = match_results_output_final_three
    
    return match_results_output_final_three, results_on_orig_df, summary_three, predict_df

# Combiner/summary functions
def combine_dfs_and_remove_dups(orig_df:PandasDataFrame, new_df:PandasDataFrame, index_col:str = "search_orig_address", match_address_series:str = "full_match", keep_only_duplicated:bool = False) -> PandasDataFrame:

    '''
    Combine two Pandas dataframes and remove duplicates according to a specified 'index' column. Data is sorted with matched addresses first, non-matched second, so that the duplicate removal gets rid of rows that are not matched in the case of address duplicates.
    '''

    # If one of the dataframes is empty, break
    if (orig_df.empty) & (new_df.empty):
        return orig_df
    


    combined_std_not_matches = pd.concat([orig_df, new_df])#, ignore_index=True)


    # If no results were combined
    if combined_std_not_matches.empty:
        combined_std_not_matches[match_address_series] = False

        if "full_address" in combined_std_not_matches.columns:
            combined_std_not_matches[index_col] = combined_std_not_matches["full_address"]
        combined_std_not_matches["fuzzy_score"] = 0
        return combined_std_not_matches
    
    # Convert index_col to string to ensure indexes from different sources are being compared correctly
    combined_std_not_matches[index_col] = combined_std_not_matches[index_col].astype(str)

    combined_std_not_matches = combined_std_not_matches.sort_values([index_col, match_address_series], ascending=[True, False])

    if keep_only_duplicated == True:
        combined_std_not_matches = combined_std_not_matches[combined_std_not_matches.duplicated(index_col)]

    combined_std_not_matches_no_dups = combined_std_not_matches.drop_duplicates(index_col).sort_index()
    
    return combined_std_not_matches_no_dups

def combine_two_matches(OrigMatchClass:MatcherClass, NewMatchClass:MatcherClass, df_name:str) -> MatcherClass:

        '''
        Combine two MatcherClass objects to retain newest matches and drop duplicate addresses.
        '''

        today_rev = datetime.now().strftime("%Y%m%d")

        NewMatchClass.match_results_output = combine_dfs_and_remove_dups(OrigMatchClass.match_results_output, NewMatchClass.match_results_output, index_col = NewMatchClass.search_df_key_field)

        NewMatchClass.results_on_orig_df = combine_dfs_and_remove_dups(OrigMatchClass.pre_filter_search_df, NewMatchClass.results_on_orig_df, index_col = NewMatchClass.search_df_key_field, match_address_series = 'Matched with reference address')
        
        # Filter out search results where a match was found
        NewMatchClass.pre_filter_search_df = NewMatchClass.results_on_orig_df

        found_index = NewMatchClass.results_on_orig_df.loc[NewMatchClass.results_on_orig_df["Matched with reference address"] == True, NewMatchClass.search_df_key_field].astype(int)

        key_field_values = NewMatchClass.search_df_not_matched[NewMatchClass.search_df_key_field].astype(int)  # Assuming list conversion is suitable
        rows_to_drop = key_field_values[key_field_values.isin(found_index)].tolist()
        NewMatchClass.search_df_not_matched = NewMatchClass.search_df_not_matched.loc[~NewMatchClass.search_df_not_matched[NewMatchClass.search_df_key_field].isin(rows_to_drop),:]#.drop(rows_to_drop, axis = 0)

        # Filter out rows from NewMatchClass.search_df_cleaned

        filtered_rows_to_keep = NewMatchClass.search_df_cleaned[NewMatchClass.search_df_key_field].astype(int).isin(NewMatchClass.search_df_not_matched[NewMatchClass.search_df_key_field].astype(int)).to_list()

        NewMatchClass.search_df_cleaned = NewMatchClass.search_df_cleaned.loc[filtered_rows_to_keep,:]#.drop(rows_to_drop, axis = 0)
        NewMatchClass.search_df_after_stand = NewMatchClass.search_df_after_stand.loc[filtered_rows_to_keep,:]#.drop(rows_to_drop)
        NewMatchClass.search_df_after_full_stand = NewMatchClass.search_df_after_full_stand.loc[filtered_rows_to_keep,:]#.drop(rows_to_drop)
        
        ### Create lookup lists
        NewMatchClass.search_df_after_stand_series = NewMatchClass.search_df_after_stand.copy().set_index('postcode_search')['search_address_stand'].str.lower().str.strip()
        NewMatchClass.search_df_after_stand_series_full_stand = NewMatchClass.search_df_after_full_stand.copy().set_index('postcode_search')['search_address_stand'].str.lower().str.strip()
            

        match_results_output_match_score_is_0 = NewMatchClass.match_results_output[NewMatchClass.match_results_output['fuzzy_score']==0.0][["index", "fuzzy_score"]].drop_duplicates(subset='index')
        match_results_output_match_score_is_0["index"] = match_results_output_match_score_is_0["index"].astype(str)
        #NewMatchClass.results_on_orig_df["index"] = NewMatchClass.results_on_orig_df["index"].astype(str)
        NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.merge(match_results_output_match_score_is_0, on = "index", how = "left")
    
        NewMatchClass.results_on_orig_df.loc[NewMatchClass.results_on_orig_df["fuzzy_score"] == 0.0, "Excluded from search"] = "Match score is 0"
        NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.drop("fuzzy_score", axis = 1)

        # Drop any duplicates, prioritise any matches
        NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.sort_values(by=["index", "Matched with reference address"], ascending=[True,False]).drop_duplicates(subset="index")
    
        NewMatchClass.output_summary = create_match_summary(NewMatchClass.match_results_output, df_name = df_name)
        print(NewMatchClass.output_summary)
    

        NewMatchClass.search_df_not_matched = filter_not_matched(NewMatchClass.match_results_output, NewMatchClass.search_df, NewMatchClass.search_df_key_field)

        ### Rejoin the excluded matches onto the output file
        #NewMatchClass.results_on_orig_df = pd.concat([NewMatchClass.results_on_orig_df, NewMatchClass.excluded_df])
    
        NewMatchClass.match_outputs_name = output_folder + "diagnostics_" + today_rev + ".csv" # + NewMatchClass.file_name + "_" 
        NewMatchClass.results_orig_df_name = output_folder + "results_" + today_rev + ".csv" # + NewMatchClass.file_name + "_"

        # Only keep essential columns
        essential_results_cols = [NewMatchClass.search_df_key_field, "Excluded from search", "Matched with reference address", "ref_index", "Reference matched address", "Reference file"]
        essential_results_cols.extend(NewMatchClass.new_join_col) 
    
        NewMatchClass.match_results_output.to_csv(NewMatchClass.match_outputs_name, index = None)
        NewMatchClass.results_on_orig_df[essential_results_cols].to_csv(NewMatchClass.results_orig_df_name, index = None)
        
        return NewMatchClass

def create_match_summary(match_results_output:PandasDataFrame, df_name:str) -> str:
    
    '''
    Create a text summary of the matching process results to export to a text box or log file.
    '''
    
    # Check if match_results_output is a dictionary-like object and has the key 'full_match'
   
    if not isinstance(match_results_output, dict) or 'full_match' not in match_results_output or (len(match_results_output) == 0):
        "Nothing in match_results_output"
        full_match_count = 0
        match_fail_count = 0
        records_attempted = 0
        dataset_length = 0
        records_not_attempted = 0
        match_rate = 0
        match_fail_count_without_excluded = 0
        match_fail_rate = 0
        not_attempted_rate = 0
        
    ''' Create a summary paragraph '''
    full_match_count = match_results_output['full_match'][match_results_output['full_match'] == True].count()
    match_fail_count = match_results_output['full_match'][match_results_output['full_match'] == False].count()
    records_attempted = int(sum((match_results_output['fuzzy_score']!=0.0) & ~(match_results_output['fuzzy_score'].isna())))
    dataset_length = len(match_results_output["full_match"])
    records_not_attempted = int(dataset_length - records_attempted)
    match_rate = str(round((full_match_count / dataset_length) * 100,1))
    match_fail_count_without_excluded = match_fail_count - records_not_attempted
    match_fail_rate = str(round(((match_fail_count_without_excluded) / dataset_length) * 100,1))
    not_attempted_rate = str(round((records_not_attempted / dataset_length) * 100,1))

    summary = ("For the " + df_name + " dataset (" + str(dataset_length) + " records), the fuzzy matching algorithm successfully matched " + str(full_match_count) +
               " records (" + match_rate + "%). The algorithm could not attempt to match " + str(records_not_attempted) +
               " records (" + not_attempted_rate +  "%). There are " + str(match_fail_count_without_excluded) + " records left to potentially match.")
    
    return summary
