# +
import pandas as pd
import numpy as np
from typing import TypeVar, Dict, List, Tuple
from rapidfuzz import process
from rapidfuzz import fuzz
import torch
import time
import re
import math
from datetime import datetime

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')
PandasSeries = TypeVar('pd.core.frame.Series')
MatchedResults = Dict[str,Tuple[str,int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# -

# # Load in data functions

# +
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

# Test the functions
#file_types = ["sample.csv", "sample.xlsx", "sample.parquet", "sample.txt"]
#file_types_results = {ftype: detect_file_type(ftype) if 'txt' not in ftype else "Unsupported file type." for ftype in file_types}
#file_types_results


# -

def get_file_name(in_name):
    # Corrected regex pattern
    match = re.search(r'\\(?!.*\\)(.*)', in_name)
    if match:
        matched_result = match.group(1)
    else:
        matched_result = None
    
    return matched_result


def load_matcher_data(in_text, in_file, in_ref, in_colnames, in_refcol, in_joincol, in_existing, Matcher):
        '''
        Load in user inputs from the Gradio interface. Convert all input types (single address, or csv input) into standardised data format that can be used downstream for the fuzzy matching.
        '''
        # Abort flag for if it's not even possible to attempt the first stage of the match for some reason
        Matcher.abort_flag = False

        today_rev = datetime.now().strftime("%Y%m%d")
    
        Matcher.search_df = pd.DataFrame()
        Matcher.ref = pd.DataFrame()
       
        ### Load in column names
        in_colnames_list = in_colnames#.tolist()[0]
        #in_joincol = in_joincol.tolist()[0]
        #print(in_joincol)
        #print(in_joincol[0][0])
        if not in_joincol:
            Matcher.in_joincol_list = ['UPRN']
        else:  Matcher.in_joincol_list = in_joincol # If user enters join columns then use these, otherwise assume UPRN

        ### Load in reference data
        #print(in_ref)

        Matcher.ref_name = get_file_name(in_ref[0].name)
    
        for ref_file in in_ref:
            #print(ref_file.name)
            temp_ref_file = read_file(ref_file.name)#, encoding="utf-8")  

            file_name_out = get_file_name(ref_file.name)
            temp_ref_file["Reference file"] = file_name_out
            
            Matcher.ref = pd.concat([Matcher.ref, temp_ref_file])

    
        ''' For the neural net model to work, the llpg columns have to be in the LPI format (e.g. with columns SaoText, SaoStartNumber etc. Here we check if we have that format. '''
    
        # Check if the source is the local db LPI data
        if 'Address_LPI' in Matcher.ref.columns:
            Matcher.ref = Matcher.ref.rename(columns={
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
        
        #print(Matcher.ref.columns)
    
        if 'SaoText' in Matcher.ref.columns: 
            Matcher.standard_llpg_format = True
            Matcher.ref_address_cols = ["SaoStartNumber", "SaoStartSuffix", "SaoEndNumber", "SaoEndSuffix", "SaoText", "PaoStartNumber", "PaoStartSuffix", "PaoEndNumber",
            "PaoEndSuffix", "PaoText", "Street", "PostTown", "Postcode"]
        else: 
            Matcher.standard_llpg_format = False
            Matcher.ref_address_cols = in_refcol#.tolist()[0]
            Matcher.ref = Matcher.ref.rename(columns={Matcher.ref_address_cols[-1]:"Postcode"})
            Matcher.ref_address_cols[-1] = "Postcode"

        # Reset index for ref as multiple files may have been combined with identical indices
        Matcher.ref = Matcher.ref.reset_index().drop("index", axis = 1)
        Matcher.ref.index.name = 'index'
            
        ### Load in search data ###
        #print(in_text)
        #print(in_file)

        Matcher.file_name = get_file_name(in_file[0].name)
    
        if in_file: 
            for match_file in in_file:
                match_temp_file = pd.read_csv(match_file.name, delimiter = ",", low_memory=False)#, encoding='cp1252')
                Matcher.search_df = pd.concat([Matcher.search_df, match_temp_file])
                Matcher.search_df_key_field = "index" #"property ref" #"index"#"row_number"#"property_ref" #
                Matcher.search_address_cols = in_colnames_list #["addr1", "addr2", "addr3", "addr4"] #["full_address"]#
                # If search address has multiple columns then the postcode column is the last, otherwise it is the only column
                #print(in_colnames_list)
                #print(len(in_colnames_list))
                    
                    
                if len(in_colnames_list) > 1:
                    Matcher.search_postcode_col = [in_colnames_list[-1]]
                else:
                    Matcher.search_df['full_address_postcode'] = Matcher.search_df[in_colnames_list[0]]
                    Matcher.search_postcode_col = ['full_address_postcode']
                    Matcher.search_address_cols.append('full_address_postcode')

                Matcher.existing_match_cols = in_existing

                if in_existing:
                    if "Matched with ref record" in Matcher.search_df.columns:
                        Matcher.search_df.loc[~Matcher.search_df[in_existing].isna(), "Matched with ref record"] = True
                    else: Matcher.search_df["Matched with ref record"] = ~Matcher.search_df[in_existing].isna()
                    
        else: 
                Matcher.search_df, Matcher.search_df_key_field, Matcher.search_address_cols, Matcher.search_postcode_col = prepare_search_address_string(in_text) 

        # Reset index for search_df as multiple files may have been combined with identical indices
        Matcher.search_df = Matcher.search_df.reset_index().drop(["index", "level_0"], axis = 1, errors = "ignore")
        Matcher.search_df.index.name = 'index'

        print("Shape of ref before filtering is: ")
        print(Matcher.ref.shape)
    
        print("Shape of search_df before filtering is: ")
        print(Matcher.search_df.shape)

        ### Filter addresses to those with length > 0
        zero_length_search_df = Matcher.search_df.copy()[Matcher.search_address_cols]
        zero_length_search_df = zero_length_search_df.fillna('')
        Matcher.search_df["address_cols_joined"] = zero_length_search_df.astype(str).sum(axis=1).str.strip()

        length_more_than_0 = Matcher.search_df["address_cols_joined"].str.len() > 0
    
 
        ### Filter addresses to match to postcode areas present in both search_df and ref_df only (postcode without the last three characters)
        if Matcher.filter_to_lambeth_pcodes == True:
            Matcher.search_df["postcode_search_area"] = Matcher.search_df[Matcher.search_postcode_col[0]].str.strip().str.upper().str.replace(" ", "").str[:-2]
            Matcher.ref["postcode_search_area"] = Matcher.ref["Postcode"].str.strip().str.upper().str.replace(" ", "").str[:-2]
               
            unique_ref_pcode_area = (Matcher.ref["postcode_search_area"][Matcher.ref["postcode_search_area"].str.len() > 3]).unique()
            postcode_found_in_search = Matcher.search_df["postcode_search_area"].isin(unique_ref_pcode_area)

            Matcher.search_df["Excluded from search"] = "Included in search"
            Matcher.search_df.loc[~(postcode_found_in_search), "Excluded from search"] = "Postcode area not found"
            Matcher.search_df.loc[~(length_more_than_0), "Excluded from search"] = "Address length 0"
            Matcher.pre_filter_search_df = Matcher.search_df.copy().reset_index()
            Matcher.pre_filter_search_df = Matcher.pre_filter_search_df.drop("address_cols_joined", axis = 1)

            Matcher.excluded_df = Matcher.search_df.copy()[~(postcode_found_in_search) | ~(length_more_than_0)]
            Matcher.search_df = Matcher.search_df[(postcode_found_in_search) & (length_more_than_0)]

            
            # Exclude records that have already been matched separately, i.e. if 'Matched with ref record' column exists, and has trues in it
            if "Matched with ref record" in Matcher.search_df.columns:
                previously_matched = Matcher.pre_filter_search_df["Matched with ref record"] == True 
                Matcher.pre_filter_search_df.loc[previously_matched, "Excluded from search"] = "Previously matched"

                #Matcher.pre_filter_search_df = Matcher.pre_filter_search_df.drop(["Combined address", "ref matched address", "Matched with ref record",	"UPRN", "index"],axis=1, errors="ignore")
                #Matcher.search_df = Matcher.search_df.drop(["Combined address", "ref matched address", "Matched with ref record",	"UPRN", "index"],axis=1, errors="ignore")
                
                Matcher.excluded_df = Matcher.search_df.copy()[~(postcode_found_in_search) | ~(length_more_than_0) | (previously_matched)]
                Matcher.search_df = Matcher.search_df[(postcode_found_in_search) & (length_more_than_0) & ~(previously_matched)]

           
            else:
                Matcher.excluded_df = Matcher.search_df.copy()[~(postcode_found_in_search) | ~(length_more_than_0)]
                Matcher.search_df = Matcher.search_df[(postcode_found_in_search) & (length_more_than_0)]
                    
   
            unique_search_pcode_area = (Matcher.search_df["postcode_search_area"]).unique()
            postcode_found_in_ref = Matcher.ref["postcode_search_area"].isin(unique_search_pcode_area)
            Matcher.ref = Matcher.ref[postcode_found_in_ref]

            
            Matcher.pre_filter_search_df = Matcher.pre_filter_search_df.drop("postcode_search_area", axis = 1)
            Matcher.search_df = Matcher.search_df.drop("postcode_search_area", axis = 1)
            Matcher.ref = Matcher.ref.drop("postcode_search_area", axis = 1)
            Matcher.excluded_df = Matcher.excluded_df.drop("postcode_search_area", axis = 1)
        else:
            Matcher.pre_filter_search_df = Matcher.search_df.copy()
            Matcher.search_df.loc[~(length_more_than_0), "Excluded from search"] = "Address length 0"
            
            Matcher.excluded_df = Matcher.search_df[~(length_more_than_0)]
            Matcher.search_df = Matcher.search_df[length_more_than_0]
            
        

        Matcher.search_df = Matcher.search_df.drop("address_cols_joined", axis = 1)
        Matcher.excluded_df = Matcher.excluded_df.drop("address_cols_joined", axis = 1)

        #Matcher.pre_filter_search_df.to_csv("pre_filter_search_df.csv")
        #Matcher.excluded_df.to_csv("excluded_entries.csv")
        #Matcher.search_df.to_csv("initial_search_df_pre_search.csv")

        Matcher.search_df_not_matched = Matcher.search_df

        print("Shape of ref after filtering is: ")
        print(Matcher.ref.shape)
    
        print("Shape of search_df after filtering is: ")
        print(Matcher.search_df.shape)
    
        Matcher.match_outputs_name = "diagnostics_initial_" + today_rev + ".csv" #+ Matcher.file_name + "_" + Matcher.ref_name + "_"
        Matcher.results_orig_df_name = "results_initial_" + today_rev + ".csv" #Matcher.file_name + "_" + Matcher.ref_name + "_" 
    
        Matcher.match_results_output.to_csv(Matcher.match_outputs_name, index = None)
        Matcher.pre_filter_search_df.to_csv(Matcher.results_orig_df_name, index = None)
        
        return Matcher


# # DF preparation functions

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
    #print(search_df.columns)
    #print(key_col)
    
    matched = search_df.drop('level_0', axis = 1, errors="ignore").reset_index()[key_col].astype(str).isin(matched_results_success[key_col].astype(str)) # 

    #matched_results_success.to_csv("matched_results_success.csv")
    #matched.to_csv("matched.csv")
    #search_df.to_csv("search_df_at_match_removal.csv")
    
    return search_df.iloc[np.where(~matched)[0]] # search_df[~matched] 


def prepare_search_address_string(
    search_str: str
) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Extracts address and postcode from search_str into new DataFrame"""

    # Validate input
    if not isinstance(search_str, str):
        raise TypeError("search_str must be a string")
        
    search_df = pd.DataFrame(data={"full_address":[search_str]})

    #print(search_df)
    
    # Extract postcode 
    postcode_series = extract_postcode(search_df, "full_address").dropna(axis=1)[0]

    # Remove postcode from address
    address_series = remove_postcode(search_df, "full_address")

    # Construct output DataFrame
    search_df_out = pd.DataFrame()
    search_df_out["full_address"] = address_series
    search_df_out["postcode"] = postcode_series

    # Set key field for joining
    key_field = "index"

    # Reset index to use as key field
    search_df_out = search_df_out.reset_index()

    # Define column names to return
    address_cols = ["full_address"]
    postcode_col = ["postcode"]

    return search_df_out, key_field, address_cols, postcode_col


# +
def prepare_search_address(
    search_df: pd.DataFrame, 
    address_cols: list,
    postcode_col: list,
    key_col: str
) -> Tuple[pd.DataFrame, str]:
    
    # Validate inputs
    if not isinstance(search_df, pd.DataFrame):
        raise TypeError("search_df must be a Pandas DataFrame")
        
    if not isinstance(address_cols, list):
        raise TypeError("address_cols must be a list")
        
    if not isinstance(postcode_col, list):
        raise TypeError("postcode_col must be a list")
        
    if not isinstance(key_col, str):
        raise TypeError("key_col must be a string")
        
    # Clean address columns
    clean_addresses = _clean_columns(search_df, address_cols)
    
    # Join address columns into one
    full_addresses = _join_address(clean_addresses, address_cols)
    
    # Add postcode column 
    full_df = _add_postcode_column(full_addresses, postcode_col)
    
    # Remove postcode from main address if there was only one column in the input
    if postcode_col == "full_address_postcode":
        # Remove postcode from address
        address_series = remove_postcode(search_df, "full_address")
        search_df["full_address"] == address_series
    
    # Ensure index column
    final_df = _ensure_index(full_df, key_col)
    
    return final_df, key_col

# Helper functions
def _clean_columns(df, cols):
   # Cleaning logic
   def clean_col(col):
       return col.astype(str).fillna("").str.replace("  "," ").str.replace("nan","").str.replace("  "," ").str.replace("  "," ").str.replace(","," ").str.strip()

   df[cols] = df[cols].apply(clean_col)
    
   return df
   
def _join_address(df, cols):
   # Joining logic
   full_address = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
   df["full_address"] = full_address.str.replace("  "," ").str.replace("  "," ").str.strip()
   
   return df
   
def _add_postcode_column(df, postcodes):
   # Add postcode column
   if isinstance(postcodes, list):
        postcodes = postcodes[0]
    
   if postcodes != "full_address_postcode":
        df = df.rename(columns={postcodes:"postcode"})
   else:
        #print(df["full_address_postcode"])
        #print(extract_postcode(df,"full_address_postcode"))
        df["full_address_postcode"] = extract_postcode(df,"full_address_postcode")[0] # 
        df = df.rename(columns={postcodes:"postcode"})
        #print(df)
   
   return df
   
def _ensure_index(df, index_col):
   # Ensure index column exists
   if ((index_col == "index") & ~("index" in df.columns)):        
        df = df.reset_index()

   df[index_col] = df[index_col].astype(str)

   return df


# -

def create_full_address(df):

    df = df.fillna("")

    df["full_address"] = df['SaoText'].str.replace(" - ", " REPL ").str.replace("- ", " REPLEFT ").str.replace(" -", " REPLRIGHT ") +\
            " " + df["SaoStartNumber"].astype(str) + df["SaoStartSuffix"] + "-" + df["SaoEndNumber"].astype(str) + df["SaoEndSuffix"] + " " + df["PaoText"].str.replace(" - ", " REPL ").str.replace("- ", " REPLEFT ").str.replace(" -", " REPLRIGHT ") +\
             " " + df["PaoStartNumber"].astype(str) + df["PaoStartSuffix"] + "-" + df["PaoEndNumber"].astype(str) + df["PaoEndSuffix"] + " " + df["Street"] +\
            " " + df["PostTown"] + " " + df["Postcode"]

    #.str.replace(r'(?<=[a-zA-Z])-(?![a-zA-Z])|(?<![a-zA-Z])-(?=[a-zA-Z])', ' ', regex=True)\
    
    #.str.replace(".0","", regex=False)\
    
    df["full_address"] = df["full_address"]\
    .str.replace("-999","")\
    .str.replace(" -"," ")\
    .str.replace("- "," ")\
    .str.replace(" REPL "," - ")\
    .str.replace(" REPLEFT ","- ")\
    .str.replace(" REPLRIGHT "," -")\
    .str.replace("\s+"," ", regex=True)\
    .str.strip()
    #.str.replace("  "," ")\

    #df.to_csv("ref_fulladdress_out.csv")
    
    return df["full_address"]


def prepare_ref_address(ref, ref_address_cols, new_join_col = ['UPRN'], standard_cols = True):
    #, cols = ['SaoText','SaoStartNumber','SaoStartSuffix','SaoEndNumber','SaoEndSuffix','PaoText','PaoStartNumber',
    #'PaoStartSuffix','PaoEndNumber','PaoEndSuffix','Street','PostTown','Postcode','UPRN']):
    
    if ('SaoText' in ref.columns) | ("Secondary_Name_LPI" in ref.columns): standard_cols = True
    else: standard_cols = False
    
    ref_address_cols_uprn = ref_address_cols.copy()

    ref_address_cols_uprn.extend(new_join_col)
    ref_address_cols_uprn_w_ref = ref_address_cols_uprn.copy()
    ref_address_cols_uprn_w_ref.extend(["Reference file"])
    
    ref_df = ref.copy()

    # Drop duplicates in the key field
    ref_df = ref_df.drop_duplicates(new_join_col)
    
    # In on-prem LPI db street has been excluded, so put this back in
    if ('Street' not in ref_df.columns) & ('Address_LPI' in ref_df.columns): 
            ref_df['Street'] = ref_df['Address_LPI'].str.replace("\\n", " ", regex = True).apply(extract_street_name)#
        
 
    #ref_df['PostTown'] = ''
    
    ref_df = ref_df[ref_address_cols_uprn_w_ref]

    ref_df = ref_df.fillna("")

    all_columns = list(ref_df) # Creates list of all column headers
    ref_df[all_columns] = ref_df[all_columns].astype(str).fillna('').replace('nan','')

    ref_df = ref_df.replace("\.0","",regex=True)

    # Create full address

    all_columns = list(ref_df) # Creates list of all column headers
    ref_df[all_columns] = ref_df[all_columns].astype(str)

    ref_df = ref_df.replace("nan","")
    ref_df = ref_df.replace("\.0","",regex=True)
    
    
    
    if standard_cols == True:
        ref_df= ref_df[ref_address_cols_uprn_w_ref].fillna('')

        ref_df["fulladdress"] = create_full_address(ref_df[ref_address_cols_uprn_w_ref])
    
    else: 
        ref_df= ref_df[ref_address_cols_uprn_w_ref].fillna('')
        
        full_address  = ref_df[ref_address_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1) 
        ref_df["fulladdress"] = full_address

    ref_df["fulladdress"] = ref_df["fulladdress"]\
    .str.replace("-999","")\
    .str.replace(" -"," ")\
    .str.replace("- "," ")\
    .str.replace(".0","", regex=False)\
    .str.replace("  "," ")\
    .str.replace("  "," ")\
    .str.strip()
    
    # Create a street column if it doesn't exist by extracting street from the full address
    
    if 'Street' not in ref_df.columns:        
        ref_df['Street'] = ref_df["fulladdress"].apply(extract_street_name)
        
    #ref_df.to_csv("ref_df_after_prep.csv", index = None)

    return ref_df


# # Standardisation functions

def standardise_wrapper_func(search_df:PandasDataFrame, ref_df:PandasDataFrame,\
                               standardise = False, filter_to_lambeth_pcodes = True, match_task = "fuzzy"):
    
    ## Search df - lower case addresses, replace spaces in postcode and 'AT' in addresses

    #assert not search_df['postcode'].isna().any() , "nulls in search_df subset post code"

    search_df["full_address_search"] = search_df["full_address"].str.lower()

    # Remove the 'AT's that appear everywhere
    search_df["full_address_search"] = search_df["full_address_search"]

    search_df['postcode_search'] = search_df['postcode'].str.lower().str.strip().str.replace(" ", "",regex=False)

    
    #assert not ref_df['Postcode'].isna().any() , "nulls in ref_df subset post code"
    # Remove nulls from ref postcode
    ref_df = ref_df[ref_df['Postcode'].notna()]
    
    ref_df["full_address_search"] = ref_df["fulladdress"].str.lower().str.strip()
    ref_df['postcode_search'] = ref_df['Postcode'].str.lower().str.strip().str.replace(" ", "", regex=False)
    
    # Block only on first 5 characters of postcode string - Doesn't give more matches and makes everything a bit slower
    # search_df['postcode_search'] = search_df['postcode_search'].str[:-1]

    ### Filter addresses to match to postcode areas present in both search_df and ref_df only (postcode without the last three characters)
    #if filter_to_lambeth_pcodes == True:
    #search_df["postcode_search_area"] = search_df["postcode_search"].str[:-3]
    #ref_df["postcode_search_area"] = ref_df["postcode_search"].str[:-3]

    #print("Shape of search_df before filtering is: ")
    #print(search_df.shape)

    #postcode_found = search_df["postcode_search_area"].isin(ref_df["postcode_search_area"])
        
    #search_df = search_df[postcode_found]
    #print("Shape of search_df after filtering is: ")
    #print(search_df.shape)

                                   
    ''' OLD CODE, NOT USED
    if filter_to_lambeth_pcodes == True:
        postcode_lookup_j = postcode_lookup.copy()[["pcd7","ladnm"]]

        postcode_lookup_j["pcd7"] = postcode_lookup_j["pcd7"].str.lower()

        search_df = pd.merge(search_df, postcode_lookup_j, how = "left", left_on = "postcode_search", right_on = "pcd7")

        search_df = search_df[search_df["ladnm"] == "Lambeth"]
        
        search_df = search_df.drop("ladnm", axis=1)'''

    ### Use standardise function

    ### Remove 'non-housing' places from the list - not included as want to check all
    #search_df_join = remove_non_housing(search_df, 'full_address_search')
    search_df_join, search_df_stand_col = standardise_address(search_df,# .drop_duplicates(
                                         "full_address_search", "search_address_stand", standardise = standardise, out_london = True)

    ## Standardise ref addresses

    # Block only on first 5 characters of postcode string - Doesn't give more matches and makes everything a bit slower
    # ref_df['postcode_search'] = ref_df['postcode_search'].str[:-1]

    ### Remove 'non-housing' places from the list
    #ref_df_join = remove_non_housing(ref_df, 'full_address_search')

    if match_task == "fuzzy":
        ref_join, ref_df_stand_col = standardise_address(ref_df, "full_address_search", "ref_address_stand", standardise = standardise, out_london = True)
    else:
        # For the neural network reference data there will be additional text columns that can be standardised.
        # I FOUND THAT THE STANDARDISATION PROCESS DID NOT HELP THE MODEL AT ALL, IN FACT IT REDUCED MATCHES AS STANDARDISING INDIVIDUAL REF COLUMNS GIVES YOU DIFFERENT RESULTS
        # FROM STANDARDISING THE WHOLE ADDRESS, THEN BREAKING IT DOWN. SO DON'T STANDARDISE. THE MODEL WILL JUST STANDARDISE THE INPUT ADDRESSES ONLY
        ref_join, ref_df_stand_col = standardise_address(ref_df, "full_address_search", "ref_address_stand", standardise = False, out_london = True)
        #ref_join_sao, ref_df_stand_col_sao = standardise_address(ref_df, "SaoText", "SaoText", standardise = standardise)
        #ref_join_pao, ref_df_stand_col_pao = standardise_address(ref_df, "PaoText", "PaoText", standardise = standardise)
        #ref_join_town, ref_df_stand_col_town = standardise_address(ref_df, "PostTown", "PostTown", standardise = standardise, out_london = False)
                
        #ref_join["SaoText"] = ref_df_stand_col_sao.str.upper()
        #ref_join["PaoText"] = ref_df_stand_col_pao.str.upper()
        #ref_join["PostTown"] = ref_df_stand_col_town.str.upper()
    
    ### Create lookup lists
    search_df_match_list = search_df_join.copy().set_index('postcode_search')['search_address_stand'].str.lower().str.strip()
    ref_df_match_list = ref_join.copy().set_index('postcode_search')['ref_address_stand'].str.lower().str.strip()
    
    return search_df_join, ref_join, search_df_match_list, ref_df_match_list, search_df_stand_col, ref_df_stand_col


def extract_street_name(address:str) -> str:
    """
    Extracts the street name from the given address.

    Args:
        address (str): The input address string.

    Returns:
        str: The extracted street name, or an empty string if no match is found.

    Examples:
        >>> address1 = "1 Ash Park Road SE54 3HB"
        >>> extract_street_name(address1)
        'Ash Park Road'

        >>> address2 = "Flat 14 1 Ash Park Road SE54 3HB"
        >>> extract_street_name(address2)
        'Ash Park Road'

        >>> address3 = "123 Main Blvd"
        >>> extract_street_name(address3)
        'Main Blvd'

        >>> address4 = "456 Maple AvEnUe"
        >>> extract_street_name(address4)
        'Maple AvEnUe'

        >>> address5 = "789 Oak Street"
        >>> extract_street_name(address5)
        'Oak Street'
    """
    
    import re
    
    street_types = [
        'Street', 'St', 'Boulevard', 'Blvd', 'Highway', 'Hwy', 'Broadway', 'Freeway',
        'Causeway', 'Cswy', 'Expressway', 'Way', 'Walk', 'Lane', 'Ln', 'Road', 'Rd',
        'Avenue', 'Ave', 'Circle', 'Cir', 'Cove', 'Cv', 'Drive', 'Dr', 'Parkway', 'Pkwy',
        'Park', 'Court', 'Ct', 'Square', 'Sq', 'Loop', 'Place', 'Pl', 'Parade', 'Estate',
        'Alley', 'Arcade','Avenue', 'Ave','Bay','Bend','Brae','Byway','Close','Corner','Cove',
        'Crescent', 'Cres','Cul-de-sac','Dell','Drive', 'Dr','Esplanade','Glen','Green','Grove','Heights', 'Hts',
        'Mews','Parade','Path','Piazza','Promenade','Quay','Ridge','Row','Terrace', 'Ter','Track','Trail','View','Villas',
        'Marsh', 'Embankment', 'Cut', 'Hill', 'Passage', 'Rise', 'Vale', 'Side'
    ]

    # Dynamically construct the regex pattern with all possible street types
    street_types_pattern = '|'.join(rf"{re.escape(street_type)}" for street_type in street_types)

    # The overall regex pattern to capture the street name
    pattern = rf'(?:\d+\s+|\w+\s+\d+\s+|.*\d+[a-z]+\s+|.*\d+\s+)*(?P<street_name>[\w\s]+(?:{street_types_pattern}))'

    def replace_postcode(address):
        pattern = r'\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2}|GIR ?0A{2})\b$|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$|\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\b$'
        return re.sub(pattern, "", address)

    
    modified_address = replace_postcode(address.upper())
    #print(modified_address)
    #print(address)
       
    # Perform a case-insensitive search
    match = re.search(pattern, modified_address, re.IGNORECASE)

    if match:
        street_name = match.group('street_name')
        return street_name.strip()
    else:
        return ""


def remove_flat_one_number_address(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:

    '''
    If there is only one number in the address, and there is no letter after the number,
    remove the word flat from the address
    '''
    
    df['contains_letter_after_number'] = df[col1].str.lower().str.contains(r"\d+(?:[a-z]|[A-Z])(?!.*\d+)", regex = True)
    df['contains_single_letter_before_number'] = df[col1].str.lower().str.contains(r'\b[A-Za-z]\b[^\d]* \d', regex = True)
    df['two_numbers_in_address'] =  df[col1].str.lower().str.contains(r"(?:\d+.*?)[^a-zA-Z0-9_].*?\d+", regex = True)
    df['contains_apartment'] = df[col1].str.lower().str.contains(r"\bapartment\b \w+|\bapartments\b \w+", "", regex = True)
    df['contains_flat'] = df[col1].str.lower().str.contains(r"\bflat\b \w+|\bflats\b \w+", "", regex = True)
    df['contains_room'] = df[col1].str.lower().str.contains(r"\broom\b \w+|\brooms\b \w+", "", regex = True)
      
    
    #df['selected_rows'] = (df['contains_letter_after_number'] == False) &\
    #                      (df['two_numbers_in_address'] == False) &\
    #                         (df['contains_flat'] == False) &\
    #                         (df['contains_apartment'] == False) &\
    #                         (df['contains_room'] == False)

    # remove word flat/apartment/room from addresses that only have one number in the address, and don't have a letter after a number
    
    df['selected_rows'] = (df['contains_letter_after_number'] == False) &\
                          (df['two_numbers_in_address'] == False) &\
                            (df['contains_single_letter_before_number'] == False) &\
                             ((df['contains_flat'] == True) |\
                             (df['contains_apartment'] == True) |\
                             (df['contains_room'] == True))
        
    df['one_number_no_flat'] =  df[df['selected_rows'] == True][col1]
    df['one_number_no_flat'] =  df['one_number_no_flat'].str.replace(r"(\bapartment\b)|(\bapartments\b)", "", regex=True).str.replace(r"(\bflat\b)|(\bflats\b)", "", regex=True).str.replace(r"(\broom\b)|(\brooms\b)", "", regex=True)

    
    #merge_columns(df, "new_col", col1, 'one_number_no_flat')
    df["new_col"] = merge_series(df[col1], df["one_number_no_flat"]) #merge_series(full_series: pd.Series, partially_filled_series: pd.Series)

    #print(df)
    
    return df['new_col']


def add_flat_addresses_start_with_letter(df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    df['contains_single_letter_at_start_before_number'] = df[col1].str.lower().str.contains(r'^\b[A-Za-z]\b[^\d]* \d', regex = True)

    df['selected_rows'] = (df['contains_single_letter_at_start_before_number'] == True)
    df['flat_added_to_string_start'] =  "flat " + df[df['selected_rows'] == True][col1]
    
    #merge_columns(df, "new_col", col1, 'flat_added_to_string_start')
    df["new_col"] = merge_series(df[col1], df['flat_added_to_string_start'])
    
    
    return df['new_col']


def extract_letter_one_number_address (df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    This function looks for addresses that have a letter after a number, but ONLY one number
    in the string, and doesn't already have a flat, apartment, or room number. 
        
    It then extracts this letter and returns this.
    
    This is for addresses such as '2b sycamore road', changes it to
    flat b 2 sycamore road so that 'b' is selected as the flat number

    
    '''
    
    df['contains_no_numbers_without_letter'] = df[col1].str.lower().str.contains(r"^(?:(?!\d+ ).)*$")
    df['contains_letter_after_number'] = df[col1].str.lower().str.contains(r"\d+(?:[a-z]|[A-Z])(?!.*\d+)")      
    df['contains_apartment'] = df[col1].str.lower().str.contains(r"\bapartment\b \w+|\bapartments\b \w+", "")
    df['contains_flat'] = df[col1].str.lower().str.contains(r"\bflat\b \w+|\bflats\b \w+", "")
    df['contains_room'] = df[col1].str.lower().str.contains(r"\broom\b \w+|\brooms\b \w+", "")
        
    df['selected_rows'] = (df['contains_no_numbers_without_letter'] == True) &\
                             (df['contains_letter_after_number'] == True) &\
                             (df['contains_flat'] == False) &\
                             (df['contains_apartment'] == False) &\
                             (df['contains_room'] == False)
            
    df['extract_letter'] =  df[(df['selected_rows'] == True)\
                                  ][col1].str.extract(r"\d+([a-z]|[A-Z])")
    
    df['extract_number'] =  df[(df['selected_rows'] == True)\
                                  ][col1].str.extract(r"(\d+)[a-z]|[A-Z]")
    

    df['letter_after_number'] = "flat " +\
                                df[(df['selected_rows'] == True)\
                                  ]['extract_letter'] +\
                                " " +\
                                df[(df['selected_rows'] == True)\
                                  ]['extract_number'] +\
                                " " +\
                                df[(df['selected_rows'])\
                                  ][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\d+([a-z]|[A-Z])","", regex=True).map(str)

    #merge_columns(df, "new_col", col1, 'letter_after_number')
    df["new_col"] = merge_series(df[col1], df['letter_after_number'])
    
    return df['new_col']


def replace_floor_flat (df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    ''' In references to basement, ground floor, first floor, second floor, and top floor
    flats, this function moves the word 'flat' to the front of the address. This is so that the
    following word (e.g. basement, ground floor) is recognised as the flat number in the 'extract_flat_no' function
    '''
    
    df['letter_after_number'] = extract_letter_one_number_address(df, col1)
       
   
    df['basement'] = "flat basement" + df[df[col1].str.lower().str.contains(r"basement"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bbasement\b","", regex=True).map(str)
    

    df['ground_floor'] = "flat a " + df[df[col1].str.lower().str.contains(r"\bground floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bground floor\b","", regex=True).map(str)

    df['first_floor'] = "flat b " + df[df[col1].str.lower().str.contains(r"\bfirst floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bfirst floor\b","", regex=True).map(str)

    df['ground_and_first_floor'] = "flat ab " + df[df[col1].str.lower().str.contains(r"\bground and first floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bground and first floor\b","", regex=True).map(str)

    df['basement_ground_and_first_floor'] = "flat basementab " + df[df[col1].str.lower().str.contains(r"\bbasement ground and first floors\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bbasement and ground and first floors\b","", regex=True).map(str)

    df['basement_ground_and_first_floor2'] = "flat basementab " + df[df[col1].str.lower().str.contains(r"\bbasement ground and first floors\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bbasement ground and first floors\b","", regex=True).map(str)

    df['second_floor'] = "flat c " + df[df[col1].str.lower().str.contains(r"\bsecond floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bsecond floor\b","", regex=True).map(str)

    df['first_and_second_floor'] = "flat bc " + df[df[col1].str.lower().str.contains(r"\bfirst and second floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bfirst and second floor\b","", regex=True).map(str)

    df['first1_floor'] = "flat b " + df[df[col1].str.lower().str.contains(r"\b1st floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\b1st floor\b","", regex=True).map(str)

    df['second2_floor'] = "flat c " + df[df[col1].str.lower().str.contains(r"\b2nd floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\b2nd floor\b","", regex=True).map(str)

    df['ground_first_second_floor'] = "flat abc " + df[df[col1].str.lower().str.contains(r"\bground and first and second floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bground and first and second floor\b","", regex=True).map(str)

    df['third_floor'] = "flat d " + df[df[col1].str.lower().str.contains(r"\bthird floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\bthird floor\b","", regex=True).map(str)

    df['third3_floor'] = "flat d " + df[df[col1].str.lower().str.contains(r"\b3rd floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\b3rd floor\b","", regex=True).map(str)   

    df['top_floor'] = "flat top " + df[df[col1].str.lower().str.contains(r"\btop floor\b"\
                                )][col1].str.replace(r"\bflat\b","", regex=True).str.replace(r"\btop floor\b","", regex=True).map(str)
    
    #merge_columns(df, "new_col", col1, 'letter_after_number')
    df["new_col"] = merge_series(df[col1], df['letter_after_number'])
    df["new_col"] = merge_series(df["new_col"], df['basement'])
    df["new_col"] = merge_series(df["new_col"], df['ground_floor'])
    df["new_col"] = merge_series(df["new_col"], df['first_floor'])
    df["new_col"] = merge_series(df["new_col"], df['first1_floor'])
    df["new_col"] = merge_series(df["new_col"], df['ground_and_first_floor'])
    df["new_col"] = merge_series(df["new_col"], df['basement_ground_and_first_floor'])
    df["new_col"] = merge_series(df["new_col"], df['basement_ground_and_first_floor2'])
    df["new_col"] = merge_series(df["new_col"], df['second_floor'])
    df["new_col"] = merge_series(df["new_col"], df['second2_floor'])
    df["new_col"] = merge_series(df["new_col"], df['first_and_second_floor'])
    df["new_col"] = merge_series(df["new_col"], df['ground_first_second_floor'])
    df["new_col"] = merge_series(df["new_col"], df['third_floor'])
    df["new_col"] = merge_series(df["new_col"], df['third3_floor'])
    df["new_col"] = merge_series(df["new_col"], df['top_floor'])
    
    #merge_columns(df, "new_col", col1, 'letter_after_number')
    #merge_columns(df, "new_col", "new_col", 'basement')
    #merge_columns(df, "new_col", "new_col", 'ground_floor')
    #merge_columns(df, "new_col", "new_col", 'first_floor')
    #merge_columns(df, "new_col", "new_col", 'first1_floor')
    #merge_columns(df, "new_col", "new_col", 'ground_and_first_floor') 
    #merge_columns(df, "new_col", "new_col", 'basement_ground_and_first_floor') 
    #merge_columns(df, "new_col", "new_col", 'basement_ground_and_first_floor2')
    #merge_columns(df, "new_col", "new_col", 'second_floor')
    #merge_columns(df, "new_col", "new_col", 'second2_floor')
    #merge_columns(df, "new_col", "new_col", 'first_and_second_floor')
    #merge_columns(df, "new_col", "new_col", 'ground_first_second_floor')
    #merge_columns(df, "new_col", "new_col", 'third_floor')
    #merge_columns(df, "new_col", "new_col", 'third3_floor')
    #merge_columns(df, "new_col", "new_col", 'top_floor')
    
    return df['new_col']


def remove_non_housing (df:PandasDataFrame, col1:PandasSeries) -> PandasDataFrame:
    '''
    Remove items from the housing list that are not housing. Includes addresses including
    the text 'parking', 'garage', 'store', 'visitor bay', 'visitors room', and 'bike rack',
    'yard', 'workshop'
    '''
    df_copy = df.copy()[~df[col1].str.lower().str.contains(\
    r"parking|garage|\bstore\b|\bstores\b|\bvisitor bay\b|visitors room|\bbike rack\b|\byard\b|\bworkshop\b")]
                                                                 
    return df_copy


def extract_prop_no (df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    Extract property number from an address. Remove flat/apartment/room numbers, 
    then extract the last number/number + letter in the string.
    '''
    try:
        prop_no = df[col1].str.replace(r"(^\bapartment\b \w+)|(^\bapartments\b \w+)", "", regex=True\
                                  ).str.replace(r"(^\bflat\b \w+)|(^\bflats\b \w+)", "", regex=True\
                                               ).str.replace(r"(^\broom\b \w+)|(^\brooms\b \w+)", "", regex=True\
                                                            ).str.replace(",", "", regex=True\
                                                              ).str.extract(r"(\d+\w+|\d+)(?!.*\d+)") #"(\d+\w+|\d+)(?!.*\d+)" 
    except:
        room_no = np.nan
        
    return prop_no


def extract_room_no (df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    Extract room number from an address. Find rows where the address contains 'room', then extract
    the next word after 'room' in the string.
    '''
    try:
        room_no = df[df[col1].str.lower().str.contains(r"\broom\b|\brooms\b",regex=True\
                                )][col1].str.replace("no.","").str.extract(r'room. (\w+)',regex=True\
                                                        ).rename(columns = {0:"room_number"})
    except:
        room_no = np.nan
        
    return room_no


def extract_flat_no (df:PandasDataFrame, col1:PandasSeries) -> PandasSeries:
    '''
    Extract flat number from an address. 
    It looks for letters after a property number IF THERE ARE NO MORE NUMBERS IN THE STRING,
    the words following the words 'flat' or 'apartment', or
    the last regex selects all characters in a word containing a digit if there are two numbers in the address
    # ^\d+([a-z]|[A-Z])
    '''  
    try:
        replaced_series = df[df[col1].str.lower().str.replace(r"^\bflats\b","flat", regex=True).str.contains(\
              r"^\d+([a-z]|[A-Z])(?!.*\d+)|\bflat\b|\bapartment\b|(\d+.*?)[^a-zA-Z0-9_].*?\d+")][col1].str.replace(\
             "no.","", regex=True)
        
        flat_no = replaced_series.str.extract(r'^\d+([a-z]|[A-Z])(?!.*\d+)|flat (\w+)|apartment (\w+)|(\d+.*?)[^a-zA-Z0-9_].*?\d+|\b([A-Za-z])\b[^\d]* \d|\bblock\b (\w+)'\
                                                                          ).rename(columns = {0:"prop_number", 1:"flat_number",2:"apart_number",3:"first_sec_number",4:"first_letter_flat_number", 5:"block_number"})
    except:
       
        flat_no = np.nan
        
    return flat_no


def extract_postcode(df:PandasDataFrame, col:str) -> PandasSeries:
    '''
    Extract a postcode from a string column in a dataframe
    '''
    postcode_series = df[col].str.upper().str.extract(pat = \
    "(\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2})|((GIR ?0A{2})\\b$)|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$)|(\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\\b$)")
    
    return postcode_series


def remove_postcode(df:PandasDataFrame, col:str) -> PandasSeries:
    '''
    Remove a postcode from a string column in a dataframe
    '''

    
    address_series_no_pcode = df[col].str.upper().str.replace(\
    "\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2}|GIR ?0A{2})\\b$|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$|\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\\b$","",\
                                                              regex=True
                                                               ).str.lower()
    
    return address_series_no_pcode


def replace_mistaken_dates(df:PandasDataFrame, col:str) -> PandasSeries:
    '''
    Identify addresses that mistakenly have dates in them and replace these dates with number values
    '''
    # Regex pattern to identify the date-month format
    pattern = r'(\d{2})-(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
    
    # Dictionary to map month abbreviations to numbers
    month_map = {
        'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4', 'may': '5', 'jun': '6',
        'jul': '7', 'aug': '8', 'sep': '9', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    # Custom replacement function
    def replace_month(match):
        day = match.group(1).lstrip('0')  # Get the day and remove leading zeros
        month = month_map[match.group(2)]  # Convert month abbreviation to number
        return f"{day}-{month}"
    
    # Apply the regex replacement
    corrected_addresses = df[col].str.replace(pattern, replace_month, regex = True)

    return corrected_addresses


def merge_columns(df:PandasDataFrame, new_col_name:str, full_column:PandasSeries, partially_filled_column:PandasSeries) -> PandasSeries:
    '''
    Merge two columns into a new column. The 'full column' is the column you want to replace values in
    'partially_filled_column' is the replacer column. 'new_col_name' is the name of the newly
    created column that merges details from both.
    '''
    replacer_column_is_null = df[partially_filled_column].isnull()

    #df.loc[replacer_column_is_null, new_col_name] = df[replacer_column_is_null][full_column]

    df[new_col_name] = df[full_column]

    df.loc[~replacer_column_is_null, new_col_name] = df[~replacer_column_is_null][partially_filled_column]
    
    return df[new_col_name]


def merge_series(full_series: pd.Series, partially_filled_series: pd.Series) -> pd.Series:
    '''
    Merge two series. The 'full_series' is the series you want to replace values in
    'partially_filled_series' is the replacer series.
    '''
    replacer_series_is_null = partially_filled_series.isnull()

    # Start with full_series values
    merged_series = full_series.copy()

    # Replace values in merged_series where partially_filled_series is not null
    merged_series[~replacer_series_is_null] = partially_filled_series.dropna()

    return merged_series


def clean_cols(col:str) -> str:
    return col.lower().strip().replace(r" ", "_").strip()

def string_match_array(to_match:array, choices:array,
                      index_name:str, matched_name:str) -> PandasDataFrame:
    
    temp = {name: process.extractOne(name,choices) 
            for name in to_match}
    
    return _create_frame(matched_results=temp, index_name=index_name,
                        matched_name=matched_name)

def standardise_address(df:PandasDataFrame, col:str, out_col:str, standardise:bool = True, out_london = True) -> PandasDataFrame:
    
    ''' 
    This function takes a 'full address' column and then standardises so that extraneous
    information is removed (i.e. postcodes & London, as this algorithm is used for London
    addresses only), and so that room/flat/property numbers can be accurately extracted. The
    standardised addresses can then be used for the fuzzy matching functions later in this 
    notebook.
    
    The function does the following:
    
    - Removes the post code and 'london' (if not dealing with addresses outside of london)
      from the address to reduce the text the algorithm has to search.
      Postcode removal uses regex to extract a UK postcode.
      
    - Remove the word 'flat' or 'apartment' from an address that has only one number in it
    
    - Add 'flat' to the start of any address that contains 'house' or 'court' (which are generally housing association buildings)
      This is because in the housing list, these addresses never have the word flat in front of them

    - Replace any addresses that don't have a space between the comma and the next word or double spaces
    
    - Replace 'number / number' and 'number-number' with 'number' (the first number in pair)
    
    - Add 'flat' to the start of addresses that include ground floor/first floor etc. flat 
      in the text. Replace with flat a,b,c etc.
    
    - Pull out property, flat, and room numbers from the address text
    
    - Return the data frame with the new columns included
    
    '''
   
    df_copy = df.copy(deep=True)
    
    # Trim the address to remove leading and tailing spaces
    df_copy[col] = df_copy[col].str.strip()
    
    ''' Remove the post code and 'london' from the address to reduce the text the algorithm has to search
    Using a regex to extract a UK postcode. I got the regex from the following. Need to replace their \b in the solution with \\b
    https://stackoverflow.com/questions/51828712/r-regular-expression-for-extracting-uk-postcode-from-an-address-is-not-ordered
        
    The following will pick up whole postcodes, postcodes with just the first part, and postcodes with the first
    part and the first number of the second half
    '''
    
    
    df_copy['add_no_pcode'] = remove_postcode(df_copy, col)
    
    #df_copy[col].str.upper().str.replace(\
   # "\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2}|GIR ?0A{2})\\b$|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$|\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\\b$",""\
    #                                                           ).str.lower()
    
    if out_london == False:
        df_copy['add_no_pcode'] = df_copy['add_no_pcode'].str.replace("london","").str.replace(r",,|, ,","", regex=True)
    
    # If the user wants to standardise the address
    if standardise:
        
        df_copy['add_no_pcode'] = df_copy['add_no_pcode'].str.lower()

        # If there are dates at the start of the address, change this
        df_copy['add_no_pcode'] = replace_mistaken_dates(df_copy, 'add_no_pcode')

        # Replace flat name variations with flat, abbreviations with full name of item (e.g. rd to road)
        df_copy['add_no_pcode'] = df_copy['add_no_pcode'].str.replace(r"\brd\b","road", regex=True).\
                                                str.replace(r"\bst\b","street", regex=True).\
                                                str.replace(r"\bave\b","avenue", regex=True).\
                                                str.replace("'", "", regex=False).\
                                                str.replace(r"\bat\b ", " ",regex=True).\
                                                str.replace("apartment", "flat",regex=False).\
                                                str.replace("studio flat", "flat",regex=False).\
                                                str.replace("cluster flat", "flats",regex=False).\
                                                str.replace(r"\bflr\b", "floor", regex=True).\
                                                str.replace(r"\bflrs\b", "floors", regex=True).\
                                                str.replace(r"\blwr\b", "lower", regex=True).\
                                                str.replace(r"\bgnd\b", "ground", regex=True).\
                                                str.replace(r"\blgnd\b", "lower ground", regex=True).\
                                                str.replace(r"\bgrd\b", "ground", regex=True).\
                                                str.replace(r"\bmais\b", "flat", regex=True).\
                                                str.replace(r"\bmaisonette\b", "flat", regex=True).\
                                                str.replace(r"\bpt\b", "penthouse", regex=True).\
                                                str.replace(r"\bbst\b","basement", regex=True).\
                                                str.replace(r"\bbsmt\b","basement", regex=True)
        
    
        # Remove the word flat or apartment from addresses that have only one number in it. 'Flat' will be re-added later to relevant addresses 
        # that need it (replace_floor_flat)
        df_copy['flat_removed'] = remove_flat_one_number_address(df_copy, 'add_no_pcode')
    
    
        ''' Remove 'flat' from any address that contains 'house' or 'court'
         From the df_copy address, remove the word 'flat' from any address that contains the word 'house' or 'court'
         This is because in the housing list, these addresses never have the word flat in front of them
        '''
        remove_flat_house = df_copy['flat_removed'].str.lower().str.contains(r"\bhouse\b")#(?=\bhouse\b)(?!.*house road)")
        remove_flat_court = df_copy['flat_removed'].str.lower().str.contains(r"\bcourt\b")#(?=\bcourt\b)(?!.*court road)")
        remove_flat_terrace = df_copy['flat_removed'].str.lower().str.contains(r"\bterrace\b")#(?=\bterrace\b)(?!.*terrace road)")
        remove_flat_house_or_court = (remove_flat_house | remove_flat_court | remove_flat_terrace == 1)

        df_copy['remove_flat_house_or_court'] = remove_flat_house_or_court
        df_copy['house_court_replacement'] = "flat " + df_copy[df_copy['remove_flat_house_or_court'] == True]['flat_removed'].str.replace(r"\bflat\b","", regex=True\
                                                                                                                                         ).str.strip().map(str)       
        #df_copy["add_no_pcode_house"] = merge_columns(df_copy, "add_no_pcode_house", 'flat_removed', "house_court_replacement")

        #merge_columns(df, "new_col", col1, 'letter_after_number')
        df_copy["add_no_pcode_house"] = merge_series(df_copy['flat_removed'], df_copy["house_court_replacement"])

        #df_copy["add_no_pcode_house"] = df_copy['flat_removed'].fillna(df_copy['house_court_replacement'])
    
        # Replace any addresses that don't have a space between the comma and the next word. and double spaces # df_copy['add_no_pcode_house']
        df_copy['add_no_pcode_house_comma'] = df_copy['add_no_pcode_house'].str.replace(r',(\w)', r', \1', regex=True).str.replace('  ', ' ', regex=False)

        # Replace number / number and number-number with number
        df_copy['add_no_pcode_house_comma_no'] = df_copy['add_no_pcode_house_comma'].str.replace(r'(\d+)\/(\d+)', r'\1', regex=True\
                                                                                                ).str.replace(r'(\d+)-(\d+)', r'\1', regex=True\
                                                                                                 ).str.replace(r'(\d+) - (\d+)', r'\1', regex=True)

        # Add 'flat' to the start of addresses that include ground/first/second etc. floor flat in the text
        df_copy['floor_replacement'] = replace_floor_flat(df_copy, 'add_no_pcode_house_comma_no')
        df_copy['flat_added_to_start_addresses_begin_letter'] = add_flat_addresses_start_with_letter(df_copy, 'floor_replacement')

        #merge_columns(df, "new_col", col1, 'letter_after_number')
        #df["new_col"] = merge_series(df[col1], df['letter_after_number'])

        #df_copy[out_col] = merge_columns(df_copy, out_col, 'add_no_pcode_house_comma_no', 'flat_added_to_start_addresses_begin_letter')
        df_copy[out_col] = merge_series(df_copy['add_no_pcode_house_comma_no'], df_copy['flat_added_to_start_addresses_begin_letter'])
        

        # Write stuff back to the original df
        df[out_col] = df_copy[out_col]
    
    else:
        df_copy[out_col] = df_copy['add_no_pcode']
        df[out_col] = df_copy['add_no_pcode']
        
    df[out_col] = df[out_col].str.strip()
    
    # Pull out property, flat, and room numbers from the address text
    df['property_number'] = extract_prop_no(df_copy, out_col)
    
    #try:    
    df[['prop_number','flat_number', 'apart_number','first_sec_number', 'first_letter_flat_number', 'block_number']] = extract_flat_no(df_copy, out_col)
    #except:
    #   print("error in one of the number columns")
    #  df['prop_number'] = np.nan
    # df['flat_number'] = np.nan
    #df['apart_number'] = np.nan
    #df['first_sec_number'] = np.nan

    #merge_columns(df, "new_col", col1, 'letter_after_number')
    #df["new_col"] = merge_series(df[col1], df['letter_after_number'])

    df_copy['flat_number'] = merge_series(df['flat_number'], df['apart_number'])
    df_copy['flat_number'] = merge_series(df['flat_number'], df['prop_number'])
    df_copy['flat_number'] = merge_series(df['flat_number'], df['first_sec_number'])
    df_copy['flat_number'] = merge_series(df['flat_number'], df['first_letter_flat_number'])
        
    #df_copy['flat_number'] = merge_columns(df, 'flat_number', 'flat_number', 'apart_number')
    #df_copy['flat_number'] = merge_columns(df, 'flat_number', 'flat_number', 'prop_number')
    #df_copy['flat_number'] = merge_columns(df, 'flat_number', 'flat_number', 'first_sec_number')
    #df_copy['flat_number'] = merge_columns(df, 'flat_number', 'flat_number', 'first_letter_flat_number')
    
    df_copy['block_number'] = df['block_number']
    
    df['room_number'] = extract_room_no(df_copy, out_col)

    #df_copy['room_number'] = extract_room_no(df, out_col)
    
    return df, df[out_col]
    #df_copy, df_copy[out_col]


# # Overarching function

def create_match_summary(match_results_output, df_name):
    
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
    records_attempted = int(sum(match_results_output['fuzzy_score']!=0.0))
    dataset_length = len(match_results_output["full_match"])
    records_not_attempted = int(dataset_length - records_attempted)
    match_rate = str(round((full_match_count / dataset_length) * 100,1))
    match_fail_count_without_excluded = match_fail_count - records_not_attempted
    match_fail_rate = str(round(((match_fail_count_without_excluded) / dataset_length) * 100,1))
    not_attempted_rate = str(round((records_not_attempted / dataset_length) * 100,2))

    summary = ("For the " + df_name + " dataset (" + str(dataset_length) + " records), the fuzzy matching algorithm successfully matched " + str(full_match_count) +
               " records (" + match_rate + "%). The algorithm was not able to match " + str(records_not_attempted) +
               " records (" + not_attempted_rate +  "%). This leaves " + str(match_fail_count_without_excluded) + " records left to match.")
    
    return summary


def combine_std_df_remove_dups(df_not_std, df_std, orig_addr_col = "search_orig_address", match_col = "full_match",
                              keep_only_duplicated = False):

    #df_not_std = df_not_std.reset_index(drop=True)
    #df_std = df_std.reset_index(drop=True)
                           
    combined_std_not_matches = pd.concat([df_not_std, df_std]).sort_values([orig_addr_col, match_col], ascending = False)
    
    if keep_only_duplicated == True:
        combined_std_not_matches = combined_std_not_matches[combined_std_not_matches.duplicated(orig_addr_col)]
    
    combined_std_not_matches_no_dups = combined_std_not_matches.drop_duplicates(orig_addr_col)
    
    return combined_std_not_matches_no_dups


def run_fuzzy_match(Matcher, standardise = False, nnet = False, file_stub= "not_std_", df_name = "Fuzzy not standardised"):

        today_rev = datetime.now().strftime("%Y%m%d")
        
        #print(Matcher.standardise)
        Matcher.standardise = standardise
    
        if nnet == False:
            compare_all_candidates,\
            diag_shortlist,\
            diag_best_match,\
            match_results_output,\
            results_on_orig_df,\
            summary,\
            search_address_cols =\
                             full_fuzzy_match(Matcher.search_df_not_matched.copy(),
                                                          Matcher.ref.copy(), 
                                                          Matcher.standardise, 
                                                          Matcher.ref_address_cols,
                                                          Matcher.search_df_key_field,
                                                          Matcher.search_address_cols,
                                                          Matcher.search_postcode_col,
                                                          Matcher.fuzzy_match_limit,
                                                          Matcher.fuzzy_scorer_used, 
                                                          Matcher.fuzzy_search_addr_limit,
                                                          Matcher.filter_to_lambeth_pcodes, 
                                                          Matcher.in_joincol_list)
            if match_results_output.empty: 
                print("Match results empty")
                Matcher.abort_flag = True
                return Matcher    
        
            else:
            
                Matcher.compare_all_candidates = compare_all_candidates
                Matcher.diag_shortlist = diag_shortlist
                Matcher.diag_best_match = diag_best_match
                Matcher.match_results_output = match_results_output
                #Matcher.match_results_output["match_method"] = df_name

            
            
            
            
        else:
            match_results_output,\
            results_on_orig_df,\
            summary,\
            predict_df_nnet =\
                             perform_full_nn_match(
                                                          Matcher.ref.copy(),
                                                          Matcher.ref_address_cols,
                                                          Matcher.search_df_not_matched.copy(),
                                                          Matcher.search_address_cols,
                                                          Matcher.search_postcode_col,
                                                          Matcher.search_df_key_field,
                                                          Matcher.standardise, 
                                                          Matcher.exported_model,
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
                                                          Matcher.labels_list)
            
            if match_results_output.empty: 
                print("Match results empty")
                Matcher.abort_flag = True
                return Matcher
            else:
                Matcher.match_results_output = match_results_output
                Matcher.predict_df_nnet = predict_df_nnet
            
        
        
        Matcher.results_on_orig_df = results_on_orig_df
        Matcher.summary = summary
  
        Matcher.output_summary = create_match_summary(Matcher.match_results_output, df_name = df_name)       
        
        Matcher.match_outputs_name = "diagnostics_" + file_stub + today_rev + ".csv" #+ Matcher.file_name + "_" + Matcher.ref_name + "_"
        Matcher.results_orig_df_name = "results_" + file_stub + today_rev + ".csv" #Matcher.file_name + "_" + Matcher.ref_name + "_" 
    
        Matcher.match_results_output.to_csv(Matcher.match_outputs_name, index = None)
        Matcher.results_on_orig_df.to_csv(Matcher.results_orig_df_name, index = None)
    
        return Matcher #summary, output_summary, search_df_not_matched, match_outputs_name, results_orig_df_name 


def combine_two_matches(OrigMatchClass, NewMatchClass, df_name):

        today_rev = datetime.now().strftime("%Y%m%d")

        NewMatchClass.match_results_output = combine_std_df_remove_dups(OrigMatchClass.match_results_output, NewMatchClass.match_results_output, orig_addr_col = NewMatchClass.search_df_key_field)    
        NewMatchClass.results_on_orig_df = combine_std_df_remove_dups(OrigMatchClass.pre_filter_search_df, NewMatchClass.results_on_orig_df, orig_addr_col = NewMatchClass.search_df_key_field, match_col = 'Matched with ref record') #OrigMatchClass.results_on_orig_df
        NewMatchClass.pre_filter_search_df = NewMatchClass.results_on_orig_df
        
        # Identify records where the match score was 0
        match_results_output_match_score_is_0 = NewMatchClass.match_results_output[NewMatchClass.match_results_output['fuzzy_score']==0.0][["index", "fuzzy_score"]]
        match_results_output_match_score_is_0["index"] = match_results_output_match_score_is_0["index"].astype(int)
        #NewMatchClass.results_on_orig_df["index"] = NewMatchClass.results_on_orig_df["index"].astype(str)
        NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.merge(match_results_output_match_score_is_0, on = "index", how = "left")
    
        NewMatchClass.results_on_orig_df.loc[NewMatchClass.results_on_orig_df["fuzzy_score"] == 0.0, "Excluded from search"] = "Match score is 0"
        NewMatchClass.results_on_orig_df = NewMatchClass.results_on_orig_df.drop("fuzzy_score", axis = 1)

    
        NewMatchClass.output_summary = create_match_summary(NewMatchClass.match_results_output, df_name = df_name)
        print(NewMatchClass.output_summary)
    

        NewMatchClass.search_df_not_matched = filter_not_matched(NewMatchClass.match_results_output, NewMatchClass.search_df, NewMatchClass.search_df_key_field)

        ### Rejoin the excluded matches onto the output file
        #NewMatchClass.results_on_orig_df = pd.concat([NewMatchClass.results_on_orig_df, NewMatchClass.excluded_df])
    
        NewMatchClass.match_outputs_name = "match_results_output_std_" + today_rev + ".csv" # + NewMatchClass.file_name + "_" 
        NewMatchClass.results_orig_df_name = "results_on_orig_df_std_" + today_rev + ".csv" # + NewMatchClass.file_name + "_" 
    
        NewMatchClass.match_results_output.to_csv(NewMatchClass.match_outputs_name, index = None)
        NewMatchClass.results_on_orig_df.to_csv(NewMatchClass.results_orig_df_name, index = None)
        
        return NewMatchClass


# # Fuzzy match functions

# ## Fuzzy match algorithm

def string_match_by_post_code_multiple(match_col:PandasSeries, reference_col:PandasSeries,
                                       matched_index_df:PandasDataFrame,
                              reference_output_name='ref_list_address',
                              matched_output_name='search_df_prep_list_address',
                              search_limit=10, scorer_name="token_set_ratio")-> MatchedResults:
    '''
    Matches by Series values; for example idx is post code and 
    values address. Search field is reduced by comparing same post codes address reference_col.
    
    Default scorer is fuzz.Wratio. This tries to weight the different algorithms
    to give the best score.
    Choice of ratio type seems to make a big difference. Looking at this link:
    https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
    and this one: 
    https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings    
    
    'partial_token_sort_ratio' seems like a good option based on my tests in
    the housing to ref match file. It accounts for words in a different order,
    and scores strings badly if they are significantly different lengths. It also
    doesn't depend a lot on full matches of just partial text in the string. 
    See the seetgeek link above for more details.
    '''
    def _create_frame_multiple(matched_results:MatchedResults,
                 index_name:str, matched_name:str) -> PandasDataFrame:
        """
        Helper function to convert results matched into desired frame format, 
        when multiple outputs are allowable in the string matching function
        """
            
        output = pd.DataFrame(matched_results).T.reset_index().melt(id_vars = "index").rename(columns= {'index':matched_name})
            
        output[[index_name, 'score', 'ref_index']] = pd.DataFrame(output["value"].tolist(), index=output.index)   
        output = output.drop(["variable", "value", "ref_index"], axis = 1).sort_values(matched_name)
    
        return output

    scorer = getattr(fuzz, scorer_name)
                     
    results= {}
    counter = 0
    for pc_match, add_match in match_col.items(): # .iteritems() is deprecated, so replaced
        
        # Link the search to the relevant ID in the original df
        current_index = matched_index_df["search_address_stand"].iloc[counter]
           
        try:
            lookup = reference_col.loc[pc_match]
            
            if isinstance(lookup, str): # lookup can be a str-> 1 address per postcode
                matched = process.extract(add_match,[lookup], scorer= getattr(fuzz, scorer_name), limit = search_limit)
            
            else: # 1+ addresses
                matched = process.extract(add_match,lookup.values, scorer= getattr(fuzz, scorer_name), limit = search_limit)
            
            # If less matches than search limit found, add on NA entries to the search limit
            if len(matched) < search_limit:
                for x in range(len(matched),search_limit):
                    matched.append(("NA",0))
                
               # matched = matched + [pc_match]
                                
            results[current_index] = matched 
                        
            
        except KeyError: # no addresses for postcode, gives list of NA tuples           
            matched = []
            
            for x in range(1,search_limit+1):
                    matched.append(("NA",0))
                        
            results[current_index] = matched
        
        counter = counter + 1
    return _create_frame_multiple(matched_results = results,index_name=reference_output_name,
                         matched_name=matched_output_name)


# def string_match_by_post_code_multiple(match_col:PandasSeries, reference_col:PandasSeries,
#                                        matched_index_df:PandasDataFrame,
#                               reference_output_name='ref_list_address',
#                               matched_output_name='search_df_prep_list_address',
#                               search_limit=10, scorer_name="token_set_ratio")-> MatchedResults:
#     '''
#     Matches by Series values; for example idx is post code and 
#     values address. Search field is reduced by comparing same post codes address reference_col.
#     
#     Default scorer is fuzz.Wratio. This tries to weight the different algorithms
#     to give the best score.
#     Choice of ratio type seems to make a big difference. Looking at this link:
#     https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
#     and this one: 
#     https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings    
#     
#     'partial_token_sort_ratio' seems like a good option based on my tests in
#     the housing to ref match file. It accounts for words in a different order,
#     and scores strings badly if they are significantly different lengths. It also
#     doesn't depend a lot on full matches of just partial text in the string. 
#     See the seetgeek link above for more details.
#     '''
#     def _create_frame_multiple(matched_results:MatchedResults,
#                  index_name:str, matched_name:str) -> PandasDataFrame:
#         """
#         Helper function to convert results matched into desired frame format, 
#         when multiple outputs are allowable in the string matching function
#         """
#             
#         output = pd.DataFrame(matched_results).T.reset_index().melt(id_vars = "index").rename(columns= {'index':matched_name})
#             
#         output[[index_name, 'score', 'ref_index']] = pd.DataFrame(output["value"].tolist(), index=output.index)   
#         output = output.drop(["variable", "value", "ref_index"], axis = 1).sort_values(matched_name)
#     
#         return output
#     
#     results= {}
#     counter = 0
#
#     scorer = getattr(fuzz, scorer_name)
#                                   
#     for current_index, (pc_match, add_match) in enumerate(match_col.items()): # .iteritems() is deprecated, so replaced
#         
#         # Link the search to the relevant ID in the original df
#         #current_index = matched_index_df["search_address_stand"].iloc[counter]
#            
#         try:
#             lookup = reference_col.loc[pc_match]
#             
#             if isinstance(lookup, str): # lookup can be a str-> 1 address per postcode
#                 matched = process.extract(add_match,[lookup], scorer= scorer, limit = search_limit)
#             
#             else: # 1+ addresses
#                 matched = process.extract(add_match,lookup.values, scorer=scorer, limit = search_limit)
#             
#             # If less matches than search limit found, add on NA entries to the search limit
#             # Pad results with "NA" if less than search_limit
#             matched += [("NA", 0)] * (search_limit - len(matched))
#                 
#             # matched = matched + [pc_match]
#                                 
#             results[current_index] = matched 
#                         
#             
#         except KeyError: # no addresses for postcode, gives list of NA tuples           
#                         
#             matched = [("NA", 0)] * search_limit
#                         
#             results[current_index] = matched
#         
#         #counter = counter + 1
#     return _create_frame_multiple(matched_results = results,index_name=reference_output_name,
#                          matched_name=matched_output_name)

# ##### Define the type for clearer annotations
# MatchedResults = List[Tuple[str, int]]
#
# def string_match_by_post_code_multiple(
#         match_col: pd.Series, 
#         reference_col: pd.Series,
#         matched_index_df: pd.DataFrame,
#         reference_output_name: str = 'ref_list_address',
#         matched_output_name: str = 'search_df_prep_list_address',
#         search_limit: int = 10, 
#         scorer_name: str = "token_set_ratio"
#     ) -> pd.DataFrame:
#
#     '''
#     Fuzzy matches two series.
#     
#     Default scorer is fuzz.Wratio. This tries to weight the different algorithms
#     to give the best score.
#     Choice of ratio type seems to make a big difference. Looking at this link:
#     https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
#     and this one: 
#     https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings
#     '''
#     
#     def _create_frame_multiple(
#             matched_results: Dict[str, MatchedResults],
#             index_name: str, 
#             matched_name: str
#         ) -> pd.DataFrame:
#         output = pd.DataFrame(matched_results).T.reset_index().melt(id_vars = "index").rename(columns= {'index': matched_name})
#         output[[index_name, 'score', 'ref_index']] = pd.DataFrame(output["value"].tolist(), index=output.index)   
#         output = output.drop(["variable", "value", "ref_index"], axis = 1).sort_values(matched_name)
#         return output
#     
#     # Get the scoring function once
#     scorer = getattr(fuzz, scorer_name)
#     
#     results = {}
#     for current_index, (pc_match, add_match) in enumerate(match_col.items()):
#         
#         # Check if the postal code exists in reference_col
#         if pc_match in reference_col:
#             lookup = reference_col.loc[pc_match]
#             
#             # Depending on the type of lookup, use appropriate method to match
#             if isinstance(lookup, str):
#                 matched = process.extract(add_match, [lookup], scorer=scorer, limit=search_limit)
#             else:
#                 matched = process.extract(add_match, lookup.values, scorer=scorer, limit=search_limit)
#             
#             # Pad results with "NA" if less than search_limit
#             matched += [("NA", 0)] * (search_limit - len(matched))
#         
#         else:
#             matched = [("NA", 0)] * search_limit
#         
#         results[matched_index_df["search_address_stand"].iloc[current_index]] = matched
#
#     return _create_frame_multiple(matched_results=results, index_name=reference_output_name, matched_name=matched_output_name)
#
# ##### The function has been optimized to some extent but needs testing with actual data.

# ## Overarching fuzzy match function

def full_fuzzy_match(search_df, ref, standardise, ref_address_cols,\
                     search_df_key_field, search_address_cols, search_postcode_col,\
                     fuzzy_match_limit, fuzzy_scorer_used, fuzzy_search_addr_limit = 100,
                    filter_to_lambeth_pcodes=False, new_join_col=["UPRN"]):

    
    # Break if search item has length 0
    if len(search_df) == 0: 
        print("Nothing to match!")
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),\
        pd.DataFrame(),pd.DataFrame(),"Nothing to match!",search_address_cols

        compare_all_candidates, diag_shortlist, diag_best_match,\
    match_results_output, results_on_orig_df, summary, search_address_cols
    
    
    if standardise == True: df_name = "standardised address"
    else: df_name = "non-standardised address"
    
    
    # Prepare address format

    if type(search_df) == str: search_df_prep, search_df_key_field, search_address_cols = prepare_search_address_string(search_df)
    else: search_df_prep, search_df_key_field = prepare_search_address(search_df, search_address_cols, search_postcode_col, search_df_key_field)
  
    ref_df = prepare_ref_address(ref, ref_address_cols, new_join_col)
    
   

    # Standardise addresses if required

    search_df_prep_join, ref_join, search_df_prep_match_list, ref_df_match_list, search_df_stand_col, ref_df_stand_col =\
                                                standardise_wrapper_func(search_df_prep.copy(), ref_df, standardise = standardise, filter_to_lambeth_pcodes=filter_to_lambeth_pcodes,
                                                                        match_task="fuzzy")

    
    # RUN WITH POSTCODE AS A BLOCKER #
    # Fuzzy match against reference addresses
    
    # Remove rows where postcode is not in ref df
    index_check = ref_df_match_list.index.isin(search_df_prep_match_list.index)
    ref_df_match_list = ref_df_match_list[index_check == True]

    if len(ref_df_match_list) == 0: 
        print("Nothing relevant in reference data to match!")
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),\
        pd.DataFrame(),pd.DataFrame(),"Nothing relevant in reference data to match!",search_address_cols
    
    #search_df_prep_match_list.to_csv("search_df_prep_match_list.csv")
    #ref_df_match_list.to_csv("ref_df_match_list.csv")

    ''' matched is the list for which every single row is searched for in the reference list (the ref).'''
    
    print("Starting the fuzzy match")
    
    tic = time.perf_counter()
    results = string_match_by_post_code_multiple(match_col = search_df_prep_match_list,
                          reference_col = ref_df_match_list,
                          matched_index_df = search_df_prep_join,
                          search_limit = fuzzy_search_addr_limit, scorer_name = fuzzy_scorer_used)

    toc = time.perf_counter()

    print(f"Performed the fuzzy match in {toc - tic:0.1f} seconds")


    # Create result dfs
    #ref_df.to_csv("ref_df_before_fuzzy_output.csv")

    match_results_output, compare_all_candidates, diag_shortlist, diag_best_match = _create_fuzzy_match_results_output(results, search_df_prep_join, ref_df, ref_join,\
                                                                                                 fuzzy_match_limit, search_df_prep, search_df_key_field, new_join_col, standardise, blocker_col = "Postcode")
    
    match_results_output['match_method'] = "Fuzzy match - postcode"
    
    search_df_not_matched = filter_not_matched(match_results_output, search_df_prep_join, search_df_key_field)

                        
    # If nothing left to match, break
    if (sum(match_results_output['full_match']==False) == 0) | (sum(match_results_output[match_results_output['full_match']==False]['fuzzy_score'])==0):
        print("Trying to break")
        
        summary = create_match_summary(match_results_output, df_name)
        
        if type(search_df) != str:
            results_on_orig_df = join_to_orig_df(match_results_output, search_df, search_df_key_field, new_join_col)
        else: results_on_orig_df = match_results_output
        
        return compare_all_candidates, diag_shortlist, diag_best_match,\
        match_results_output, results_on_orig_df, summary, search_address_cols
    
    
    
    # RUN WITH STREET AS A BLOCKER #
    
    ### Redo with street as blocker
    search_df_prep_join_street = search_df_not_matched.copy() #search_df_prep_join.copy()
    search_df_prep_join_street['search_address_stand_w_pcode'] = search_df_prep_join_street['search_address_stand'] + " " + search_df_prep_join_street['postcode_search']
    
    search_df_prep_join_street['street']= search_df_prep_join_street['full_address_search'].apply(extract_street_name)       
    ref_join['ref_address_stand_w_pcode'] = ref_join['ref_address_stand'] + " " + ref_join['postcode_search']
        
        
    ### Create lookup lists
    search_df_match_list_street = search_df_prep_join_street.set_index('street')['search_address_stand'].str.lower().str.strip() # 'search_address_stand'
    ref_df_match_list_street = ref_join.copy().set_index('Street')['ref_address_stand'].str.lower().str.strip() # 'ref_address_stand'
        
    
    # Remove rows where street is not in ref df
    index_check = ref_df_match_list_street.index.isin(search_df_match_list_street.index)
    ref_df_match_list_street = ref_df_match_list_street[index_check == True]
    
    #search_index_check = search_df_match_list_street.index.isin(ref_df_match_list_street.index)
    #search_df_match_list_street = search_df_match_list_street[search_index_check == True]

    # If nothing left to match, break
    if (len(ref_df_match_list_street) == 0) | ((len(search_df_match_list_street) == 0)):
        
        summary = create_match_summary(match_results_output, df_name)
        
        if type(search_df) != str:
            results_on_orig_df = join_to_orig_df(match_results_output, search_df, search_df_key_field, new_join_col)
        else: results_on_orig_df = match_results_output
        
        return compare_all_candidates, diag_shortlist, diag_best_match,\
        match_results_output, results_on_orig_df, summary, search_address_cols
    
    print("Starting the fuzzy match with street as blocker")
    
    tic = time.perf_counter()
    results_st = string_match_by_post_code_multiple(match_col = search_df_match_list_street,
                          reference_col = ref_df_match_list_street,
                          matched_index_df = search_df_prep_join_street,
                          search_limit = fuzzy_search_addr_limit, scorer_name = fuzzy_scorer_used)

    toc = time.perf_counter()

    print(f"Performed the fuzzy match in {toc - tic:0.1f} seconds")
    
    match_results_output_st, compare_all_candidates_st, diag_shortlist_st, diag_best_match_st = _create_fuzzy_match_results_output(results_st, search_df_prep_join_street, ref_df, ref_join,\
                                                                                                 fuzzy_match_limit, search_df_prep, search_df_key_field, new_join_col, standardise, blocker_col = "Street")
    
    match_results_output_st['match_method'] = "Fuzzy match - street"

    
    match_results_output_st_out = combine_std_df_remove_dups(match_results_output, match_results_output_st, orig_addr_col = search_df_key_field)
        
    match_results_output = match_results_output_st_out
    
    
    summary = create_match_summary(match_results_output, df_name)
    #print(summary)
    #print(match_results_output['match_method'].value_counts())
    
    
    ### Join URPN back onto orig df

    if type(search_df) != str:
        results_on_orig_df = join_to_orig_df(match_results_output, search_df, search_df_key_field, new_join_col)
    else: results_on_orig_df = match_results_output
    


    
    return compare_all_candidates, diag_shortlist, diag_best_match, match_results_output, results_on_orig_df, summary, search_address_cols


def _create_fuzzy_match_results_output(results, search_df_prep_join, ref_df, ref_join, fuzzy_match_limit, search_df_prep, search_df_key_field, new_join_col, standardise, blocker_col):

        ## Diagnostics

        compare_all_candidates, diag_shortlist, diag_best_match =\
                                      refine_export_results(results_df=results,\
                                      matched_df = search_df_prep_join, ref_list_df = ref_join,
                                      fuzzy_match_limit = fuzzy_match_limit, blocker_col=blocker_col)
        
        ## Fuzzy search results

        # Join results data onto the original housing list to create the full output
        match_results_output = pd.merge(search_df_prep[[search_df_key_field, "full_address","postcode"]],\
                             diag_best_match[['search_orig_address','reference_orig_address', 'full_match',\
                                            'fuzzy_score_match', 'property_number_match','fuzzy_score','search_mod_address',\
                                            'reference_mod_address']], how = "left", left_on = "full_address", right_on = "search_orig_address").\
                                                drop(["postcode", "search_orig_address"], axis = 1).rename(columns={"full_address":"search_orig_address"})

        #print(len(match_results_output))
        
        # Join UPRN back onto the data from reference data
        joined_ref_cols = ["fulladdress", "Reference file"]
        joined_ref_cols.extend(new_join_col)


        #ref_df[joined_ref_cols].to_csv("ref_df_joined_ref_cols.csv")
    
        match_results_output = pd.merge(match_results_output,ref_df[joined_ref_cols].drop_duplicates("fulladdress"), how = "left",\
                             left_on = "reference_orig_address",right_on = "fulladdress").drop("fulladdress", axis = 1)

        # Convert long keys to string to avoid data loss
        match_results_output[search_df_key_field] = match_results_output[search_df_key_field].astype("str")
        match_results_output[new_join_col] = match_results_output[new_join_col].astype("string")
        match_results_output["standardised_address"] = standardise
    
        match_results_output = match_results_output.sort_values(search_df_key_field, ascending = True)
        
        #print(len(match_results_output))
        
        return match_results_output, compare_all_candidates, diag_shortlist, diag_best_match


# ## Export diagnostics function

def refine_export_results(results_df:PandasDataFrame, 
                           matched_df:PandasDataFrame, ref_list_df:PandasDataFrame,
                           matched_col="search_df_prep_list_address", ref_list_col="ref_list_address",\
                           final_matched_address_col="search_address_stand", final_ref_address_col="ref_address_stand",\
                           orig_matched_address_col = "full_address", orig_ref_address_col = "fulladdress",\
                           fuzzy_match_limit=75, blocker_col="Postcode") -> PandasDataFrame:
    '''
    This function takes a result file from the fuzzy search, then refines the 'matched results' according
    the score limit specified by the user and exports results list, matched and unmatched files.
    '''
    
    results_join = results_df.reset_index()
    
    # Rename score column
    results_join = results_join.rename(columns = {"score":"fuzzy_score"})
    
    # Create new df for export at end of function
    compare_all_candidates = results_join.copy()
        
    # Remove empty addresses
    results_join = results_join[results_join[matched_col] !=0 ]

    ### Join property number and flat/room number onto results_df

    reference_j = ref_list_df[[final_ref_address_col, "property_number","flat_number","room_number","block_number",\
                               orig_ref_address_col,"Postcode"\
                      ]].rename(columns={"property_number": "reference_property_number",\
                                         "flat_number":"reference_flat_number",\
                                         "room_number":"reference_room_number",
                                         "block_number":"reference_block_number",\
                                         orig_ref_address_col: "reference_orig_address",\
                                        final_ref_address_col:'reference_list_address'                                         
                                        })

    results_join = results_join.merge(reference_j, how = "left", left_on = ref_list_col, right_on = "reference_list_address")

    matched_j = matched_df[[final_matched_address_col,"property_number","flat_number","room_number", "block_number",
                            orig_matched_address_col, "postcode",\
                             ]].rename(columns={"property_number": "matched_property_number",\
                                                "flat_number":"matched_flat_number",\
                                                "room_number":"matched_room_number",\
                                                "block_number":"matched_block_number",\
                                               orig_matched_address_col:"search_orig_address",\
                                               final_matched_address_col:'search_mod_address'
                                               })

    diag_j = results_join.merge(matched_j, how = "left", left_on = matched_col, right_on = "search_mod_address")
        

    ## Calculate highest fuzzy score from all candidates, keep all candidates with matching highest fuzzy score
    results_max_fuzzy_score = diag_j.groupby(matched_col)["fuzzy_score"].max().reset_index().rename(columns={"fuzzy_score": "max_fuzzy_score"})
    
    diag_shortlist = pd.merge(diag_j, results_max_fuzzy_score, how = "left", left_on = matched_col, right_on = matched_col)
    diag_shortlist = diag_shortlist[diag_shortlist["fuzzy_score"] == diag_shortlist["max_fuzzy_score"]]

    # Fuzzy match limit for records with no numbers in it is 0.95 or the provided fuzzy_match_limit, whichever is higher
    diag_shortlist["fuzzy_score_match"] = diag_shortlist['fuzzy_score'] >= fuzzy_match_limit

    if fuzzy_match_limit > 0.95: no_number_fuzzy_match_limit = fuzzy_match_limit
    else: no_number_fuzzy_match_limit = fuzzy_match_limit

    ### Count number of numbers in search string
    diag_shortlist["number_count_search_string"] =  diag_shortlist["search_mod_address"].str.count(r'\d')
    diag_shortlist["no_numbers_in_search_string"] = diag_shortlist["number_count_search_string"] == 0

    # Replace fuzzy_score_match values for addresses with no numbers in them
    diag_shortlist.loc[(diag_shortlist["no_numbers_in_search_string"]==True) & (diag_shortlist['fuzzy_score'] >= no_number_fuzzy_match_limit), "fuzzy_score_match"] = True
    diag_shortlist.loc[(diag_shortlist["no_numbers_in_search_string"]==True) & (diag_shortlist['fuzzy_score'] < no_number_fuzzy_match_limit), "fuzzy_score_match"] = False

    # If blocking on street, don't match addresses with 0 numbers in. There are too many options and the matches are rarely good
    if blocker_col == "Street":
        diag_shortlist.loc[(diag_shortlist["no_numbers_in_search_string"]==True), "fuzzy_score_match"] = False
                               
    diag_shortlist = diag_shortlist.fillna("").drop(["number_count_search_string", "no_numbers_in_search_string"], axis = 1)

    # Following considers full matches to be those that match on property number and flat number, and the postcode is relatively close.
 
   
    diag_shortlist["property_number_match"] = (diag_shortlist["reference_property_number"] == diag_shortlist["matched_property_number"])
    diag_shortlist["flat_number_match"] = (diag_shortlist['matched_flat_number'] == diag_shortlist['reference_flat_number'])
    diag_shortlist["room_number_match"] = (diag_shortlist['matched_room_number'] == diag_shortlist['reference_room_number'])
    diag_shortlist["block_number_match"] = (diag_shortlist['matched_block_number'] == diag_shortlist['reference_block_number'])

    # Full number match is currently considered only a match between property number and flat number
                               
    diag_shortlist['full_number_match'] = (diag_shortlist["property_number_match"] == True) & (diag_shortlist["flat_number_match"] == True) &\
                                          (diag_shortlist["block_number_match"] == True)
   
    
    ### Postcodes need to be close together, so all the characters should match apart from the last two 
    diag_shortlist['close_postcode_match'] = diag_shortlist['postcode'].str[:-1] == diag_shortlist['Postcode'].str[:-1]
    
    
    
    diag_shortlist["full_match"] = (diag_shortlist["fuzzy_score_match"] == True) &\
                                    (diag_shortlist['full_number_match'] == True) &\
                                    (diag_shortlist['close_postcode_match'] == True)



    
    #diag_shortlist = diag_shortlist.sort_values(by = [matched_col, 'full_number_match'],ascending =False)
    
    #diag_shortlist.to_csv("diag_shortlist_interim.csv")
    
    diag_shortlist = diag_shortlist[['search_orig_address','reference_orig_address',
            'full_match',
            'full_number_match',
            'flat_number_match',
            'room_number_match',
            'block_number_match',
            'property_number_match',
            'close_postcode_match',
            'fuzzy_score_match',
            'fuzzy_score', 
            'matched_property_number', 'reference_property_number',  
            'matched_flat_number', 'reference_flat_number', 
            'matched_room_number', 'reference_room_number',
            'matched_block_number', 'reference_block_number',
            'search_mod_address', 'reference_list_address','Postcode']] # , 'uprn'
    
    diag_shortlist = diag_shortlist.rename(columns = {"matched_flat_number":"search_flat_number",
                                                      "matched_room_number":"search_room_number",
                                                      "matched_block_number":"search_block_number",
                                                      "matched_property_number":"search_property_number",
                                                     "reference_list_address":"reference_mod_address"})
 

    '''
    If a matched address is duplicated, choose the version that has a number match, 
    if there is no number match, then property match, room number match, then best fuzzy score, then show all options
    '''  
    
    
    ### Dealing with tie breaks ##
    # Do a backup simple Wratio search on the open text to act as a tie breaker when the fuzzy scores are identical
    def compare_strings_wratio(row, scorer = fuzz.WRatio):
        search_score = process.cdist([row["search_mod_address"]], [row["reference_mod_address"]], scorer=scorer)
        return search_score[0][0]

    diag_shortlist_dups = diag_shortlist.loc[diag_shortlist.duplicated(subset= ["search_mod_address", 'full_number_match', "search_room_number", "fuzzy_score"], keep=False)]
    diag_shortlist_dups["wratio_score"] = diag_shortlist_dups.apply(compare_strings_wratio, axis=1)
                               
    diag_shortlist = diag_shortlist.merge(diag_shortlist_dups[["wratio_score"]], left_index=True, right_index=True, how = "left")
                               
    # Choose the best match
    diag_shortlist = diag_shortlist.sort_values(["search_mod_address", 'full_number_match', "search_room_number", "fuzzy_score", "wratio_score"], ascending = False)
    diag_best_match = diag_shortlist[["search_orig_address",'reference_orig_address', 'full_match',
                                          'full_number_match', 'room_number_match','flat_number_match', 'block_number_match',
                                            'property_number_match', 'close_postcode_match','fuzzy_score_match','fuzzy_score',
                                           "search_mod_address","reference_mod_address",'Postcode']].drop_duplicates("search_mod_address")


    #diag_shortlist.to_csv("diagnostics_shortlist_" + today_rev + ".csv", index=None)
   
    return compare_all_candidates, diag_shortlist, diag_best_match

# # Neural net functions

import recordlinkage
import tensorflow as tf


def vocab_lookup(characters: str, vocab) -> (int, np.ndarray):
    """
    Taken from the function from the addressnet package by Jason Rigby
    
    Converts a string into a list of vocab indices
    :param characters: the string to convert
    :param training: if True, artificial typos will be introduced
    :return: the string length and an array of vocab indices
    """
    result = list()
    for c in characters.lower():
        try:
            result.append(vocab.index(c) + 1)
        except ValueError:
            result.append(0)
    return len(characters), np.array(result, dtype=np.int64)


# ## Neural net predictor functions

def text_to_model_input_local(in_text, vocab, model_type = "estimator"):
    addresses_out = []
    model_input_out = []
    encoded_text = []
    
    # Calculate longest string length
    import heapq
 
    # get the index of the largest element in the list
    index = heapq.nlargest(1, range(len(in_text)), key=lambda x: len(in_text[x]))[0]
 
    # use the index to get the corresponding string
    longest_string = len(in_text[index])
 
    #print("Longest string is: " + str(longest_string))

    for x in range(0, len(in_text)):
        
        out = vocab_lookup(in_text[x], vocab)
        addresses_out.append(out)
        
        #print(out)
        
        if model_type == "estimator":
            model_input_add= tf.train.Example(features=tf.train.Features(feature={
            'lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=[out[0]])),
            'encoded_text': tf.train.Feature(int64_list=tf.train.Int64List(value=out[1].tolist()))  
            })).SerializeToString()

            model_input_out.append(model_input_add)
        
        if model_type == "keras":
            encoded_text.append(out[1])
            
            
    if model_type == "keras":
        # Pad out the strings so they're all the same length. 69 seems to be the value for spaces
        model_input_out = tf.keras.utils.pad_sequences(encoded_text, maxlen=longest_string, padding="post", truncating="post", value=0)#69)
        
        
    return addresses_out, model_input_out


def reformat_predictions_local(predict_out):

    predictions_list_reformat = []

    for x in range(0,len(predict_out['pred_output_classes'])):

        new_entry = {'class_ids': predict_out['pred_output_classes'][x], 'probabilities': predict_out['probabilities'][x]}
        predictions_list_reformat.append(new_entry)
        
    return predictions_list_reformat


def predict_serve_conv_local(in_text:List[str], labels_list, predictions) -> List[Dict[str, str]]:
 
    class_names = [l.replace("_code", "") for l in labels_list]
    class_names = [l.replace("_abbreviation", "") for l in class_names]
    
    #print(input_text)
    
    #print(list(zip(input_text, predictions)))
    
    for addr, res in zip(in_text, predictions):
        
        #print(zip(input_text, predictions))
        
        mappings = dict()
        
        
        #print(addr.upper())
        #print(res['class_ids'])
        
        for char, class_id in zip(addr.upper(), res['class_ids']):
            #print(char)
            if class_id == 0:
                continue
            cls = class_names[class_id - 1]
            mappings[cls] = mappings.get(cls, "") + char
            
        
        #print(mappings)
        yield mappings
        #return mappings


def prep_predict_export(prediction_outputs, in_text):
    
    out_list = list(prediction_outputs)
    
    df_out = pd.DataFrame(out_list)
    
    #print(in_text)
    #print(df_out)
    
    df_out["address"] = in_text
    
    return out_list, df_out


# +
### Predict function for imported .pb files
    
def full_predict_func(list_to_predict, model, vocab, labels_list):
    
    if hasattr(model, "summary"): # Indicates this is a keras model rather than an estimator
        model_type = "keras"
    else: model_type = "estimator"
    
    list_to_predict = [x.upper() for x in list_to_predict]
    
    addresses_out, model_input = text_to_model_input_local(list_to_predict, vocab, model_type) 

    if hasattr(model, "summary"):
        probs = model.predict(model_input, use_multiprocessing=True)

        classes = probs.argmax(axis=-1)

        predictions = {'pred_output_classes':classes, 'probabilities':probs}
        
    else:
        #predictions = model.signatures["predict_output"](predictor_inputs=tf.constant(model_input)) # This was for when using the contrib module
        predictions = model.signatures["serving_default"](predictor_inputs=tf.constant(model_input))
    
    predictions_list_reformat = reformat_predictions_local(predictions)
    

    #### Final output as list or dataframe

    output = predict_serve_conv_local(list(list_to_predict), labels_list, predictions_list_reformat)

    list_out, predict_df = prep_predict_export(output, list_to_predict)
    
    return list_out, predict_df


# -

def predict_torch(model, model_type, input_text, word_to_index, device):
    #print(device)
    
    # Convert input_text to tensor of character indices
    indexed_texts = [[word_to_index.get(char, word_to_index['<UNK>']) for char in text] for text in input_text]
    
    # Calculate max_len based on indexed_texts
    max_len = max(len(text) for text in indexed_texts)
    
    # Pad sequences and convert to tensor
    padded_texts = torch.tensor([text + [word_to_index['<pad>']] * (max_len - len(text)) for text in indexed_texts])
    
    with torch.no_grad():
        texts = padded_texts.to(device)
        
        if (model_type == "lstm") | (model_type == "gru"):
            text_lengths = texts.ne(word_to_index['<pad>']).sum(dim=1)
            predictions = model(texts, text_lengths)
        
        if model_type == "transformer":
            # Call model with texts and pad_idx
            predictions = model(texts, word_to_index['<pad>'])
        
    # Convert predictions to most likely category indices
    _, predicted_indices = predictions.max(2)
    return predicted_indices


def torch_predictions_to_dicts(input_text, predicted_indices, index_to_category):
    results = []
    for i, text in enumerate(input_text):
        # Treat each character in the input text as a "token"
        tokens = list(text)  # Convert string to a list of characters
        
        # Create a dictionary for the current text
        curr_dict = {}
        
        # Iterate over the predicted categories and the tokens together
        for category_index, token in zip(predicted_indices[i], tokens):
            # Convert the category index to its name
            category_name = index_to_category[category_index.item()]
            
            # Append the token to the category in the dictionary (or create the category if it doesn't exist)
            if category_name in curr_dict:
                curr_dict[category_name] += token  # No space needed between characters
            else:
                curr_dict[category_name] = token
        
        results.append(curr_dict)
    
    return results


def torch_prep_predict_export(prediction_outputs, in_text):
    
    #out_list = list(prediction_outputs)
    
    df_out = pd.DataFrame(prediction_outputs).drop("IGNORE", axis = 1)
    
    #print(in_text)
    #print(df_out)
    
    df_out["address"] = in_text
    
    return df_out


def full_predict_torch(model,  model_type, input_text, word_to_index, cat_to_idx, device):
    
    input_text = [x.upper() for x in input_text]
    
    predicted_indices = predict_torch(model, model_type, input_text, word_to_index, device)
    
    index_to_category = {v: k for k, v in cat_to_idx.items()}

    results_dict = torch_predictions_to_dicts(input_text, predicted_indices, index_to_category)
    
    df_out = torch_prep_predict_export(results_dict, input_text)
       
    return results_dict, df_out


def post_predict_clean(predict_df, orig_search_df, ref_address_cols, search_df_key_field):

    
    # Add address to ref_address_cols
    ref_address_cols_add = ref_address_cols.copy()
    ref_address_cols_add.extend(['address'])                
    
    # Create column if it doesn't exist
    for x in ref_address_cols:

        predict_df[x] = predict_df.get(x, np.nan)
    
    predict_df = predict_df[ref_address_cols_add]
    
    #Columns that are in the ref and model, but are not matched in this instance, need to be filled in with blanks

    predict_cols_match = list(predict_df.drop(["address"],axis=1).columns)
    predict_cols_match_uprn = predict_cols_match.copy()
    predict_cols_match_uprn.append("UPRN")

    pred_output_missing_cols = list(set(ref_address_cols) - set(predict_cols_match))
    predict_df[pred_output_missing_cols] = np.nan
    predict_df = predict_df.fillna("")

    #Convert all columns to string

    all_columns = list(predict_df) # Creates list of all column headers
    predict_df[all_columns] = predict_df[all_columns].astype(str)

    predict_df = predict_df.replace("\.0","",regex=True)

    #When comparing with ref, the postcode existing in the data will be used to compare rather than the postcode predicted by the model. This is to minimise errors in matching

    predict_df = predict_df.rename(columns={"Postcode":"Postcode_predict"})

    orig_search_df_pc = orig_search_df[["postcode"]].rename(columns={"postcode":"Postcode"}).reset_index().drop("index", axis=1)

    predict_df = pd.concat([predict_df, orig_search_df_pc], axis = 1)
    
    predict_df[search_df_key_field] = orig_search_df[search_df_key_field]

    #predict_df = predict_df.drop("index", axis=1)

    #predict_df.to_csv("predict_end_of_clean.csv")
    
    return predict_df


def create_fuzzy_matched_col(df, orig_match_col, pred_match_col, fuzzy_method:"WRatio", match_score=95):

    results = []

    for orig_index, orig_string in df[orig_match_col].items():
        
        predict_string = df[pred_match_col][orig_index] 
        
        if (orig_string == '') and (predict_string == ''):
            results.append(np.nan)
            
        else:
            fuzz_score = process.extract(orig_string, [predict_string], scorer= getattr(fuzz, fuzzy_method))
            results.append(fuzz_score[0][1])

    new_result_col_score = (orig_match_col + "_fuzz_score")
    new_result_col_match = (orig_match_col + "_fuzz_match") 

    df[new_result_col_score] = results
    df[new_result_col_match] = df[new_result_col_score] >= match_score
    #df[new_result_col_match][df[new_result_col_score].isna()] = np.nan
    df.loc[df[new_result_col_score].isna(), new_result_col_match] = np.nan
    
    return df


def join_to_orig_df(match_results_output, search_df, search_df_key_field, new_join_col):
    
    match_results_output_success = match_results_output[match_results_output["full_match"]==True]

    # If you're joining to the original df on index you will need to recreate the index again
    #print(match_results_output_success.columns)

    match_results_output_success = match_results_output_success.rename(columns={"reference_orig_address":"ref matched address",\
                                      "full_match":"Matched with ref record",\
                                        'uprn':'UPRN'                                                                             
                                     })
    
    ref_join_cols = ["ref matched address","Matched with ref record", "Reference file", search_df_key_field]
    ref_join_cols.extend(new_join_col)

    
    
    if (search_df_key_field == "index"):
        #print("Doing index match")
        
        match_results_output_success[search_df_key_field] = match_results_output_success[search_df_key_field].astype(float).astype(int)


        #search_df.to_csv("search_df_pre_merge.csv")
        
        search_df_j = search_df.merge(match_results_output_success[ref_join_cols], left_index=True, right_on = search_df_key_field, how = "left",
                                                                            suffixes = ('', '_y')) #.reset_index().drop("index", axis=1)
        #search_df_j.to_csv("search_df_j.csv")
        
    else:
        search_df_j = search_df.merge(match_results_output_success[ref_join_cols],how = "left", on = search_df_key_field, suffixes = ('', '_y'))


    # If the join columns already exist in the search_df, then use the new column to fill in the NAs in the original column, then delete the new column
    if "ref matched address_y" in search_df_j.columns: search_df_j['ref matched address'] = search_df_j['ref matched address'].fillna(search_df_j['ref matched address_y'])
    if "Matched with ref record_y" in search_df_j.columns: search_df_j['Matched with ref record'] = pd.Series(np.where(search_df_j['Matched with ref record_y'].notna(), search_df_j['Matched with ref record_y'], search_df_j['Matched with ref record']))
    if "Reference file_y" in search_df_j.columns: search_df_j['Reference file'] = search_df_j['Reference file'].fillna(search_df_j['Reference file_y'])
    if "UPRN_y" in search_df_j.columns: search_df_j['UPRN'] = search_df_j['UPRN'].fillna(search_df_j['UPRN_y'])
    #search_df_j['UPRN'] = search_df_j['UPRN'].fillna(search_df_j['UPRN_y'])
    #search_df_j[search_df_key_field] = search_df_j[search_df_key_field].fillna(search_df_j[search_df_key_field + '_y'])

    search_df_j = search_df_j.drop(['ref matched address_y', 'Matched with ref record_y', 'Reference file_y', 'search_df_key_field_y', 'UPRN_y', 'index_y'], axis = 1, errors = "ignore")

    #search_df_j.set_index('index', inplace=True)

    
    # Drop columns that aren't useful
    search_df_j = search_df_j.drop(["full_address_search","postcode_search", "full_address_1", "full_address_2", "full_address",
                                   "address_stand", "property_number","prop_number" "flat_number" "apart_number" "first_sec_number" "room_number"],
                               axis=1, errors='ignore')

    # Replace blanks with NA, fix UPRNs

    search_df_j = search_df_j.replace(r'^\s*$', np.nan, regex=True)

    

    search_df_j[new_join_col] = search_df_j[new_join_col].astype(str).replace(".0","", regex=False).replace("nan","", regex=False)
    
    # Replace cells with only 'nan' with blank
    search_df_j = search_df_j.replace(r'^nan$', "", regex=True)

    #search_df_j = search_df_j.rename(columns={"full_address":"Combined address"})

    #print(search_df_j.index)

    # Only keep 
    #search_df_j = search_df_j[search_df_j.index.notna() & (search_df_j.index != '')]
    
    
    return search_df_j


# ## Recordlinkage matching functions

def score_based_match(predict_df_search, ref_search, orig_search_df, matching_variables,
                      text_columns, blocker_column,  weights, fuzzy_method, score_cut_off, search_df_key_field, standardise, new_join_col):
    
    # Use the merge command to match group1 and group2
    predict_df_search[matching_variables] = predict_df_search[matching_variables].astype(str)
    ref_search[matching_variables] = ref_search[matching_variables].astype(str).replace("-999","")

    # SaoText needs to be exactly the same to get a 'full' match. So I moved that to the exact match group
    exact_columns = main_list = list(set(matching_variables) - set(text_columns))

    # Replace all blanks with a space, so they can be included in the fuzzy match searches

    for column in text_columns:
        predict_df_search[column][predict_df_search[column] == ''] = ' '
        ref_search[column][ref_search[column] == ''] = ' '

    # Score based match functions
    
    # Create an index of all pairs
    indexer = recordlinkage.Index()

    # Block on selected blocker column
    
    ## Remove all NAs from predict_df blocker column
    if blocker_column[0] == "PaoStartNumber":
        predict_df_search = predict_df_search[~(predict_df_search[blocker_column[0]].isna()) & ~(predict_df_search[blocker_column[0]] == '')& ~(predict_df_search[blocker_column[0]].str.contains(r'^\s*$', na=False))]
    
    
    indexer.block(blocker_column) #matchkey.block(["Postcode", "PaoStartNumber"])

    # Generate candidate pairs

    pairsSBM = indexer.index(predict_df_search,ref_search)

    print('Running with ' + blocker_column[0] + ' as blocker has created', len(pairsSBM), 'pairs.')
    
    # If no pairs are found, break
    if len(pairsSBM) == 0: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Call the compare class from the toolkit 
    compareSBM = recordlinkage.Compare()

    # Assign variables to matching technique - exact
    for columns in exact_columns:
        compareSBM.exact(columns, columns, label = columns, missing_value = 0)

    # Assign variables to matching technique - fuzzy
    for columns in text_columns:
        compareSBM.string(columns, columns, label = columns, missing_value = 0, method = fuzzy_method)

    ## Run the match - compare each column within the blocks according to exact or fuzzy matching (defined in cells above)

    scoresSBM = compareSBM.compute(pairs = pairsSBM, x = predict_df_search, x_link = ref_search)

    
    
    scoresSBM_w = scoresSBM.copy()

    # Fill in the 'NA' searches so that they are not counted in the scoring

    #scoresSBM_w = scoresSBM_w.replace(-999, np.nan)

    # Establish a fuzzy cut off - anything above this value is a 'match', anything below is a miss

    #fuzzy_cut_off = 0.98

    #for column in text_columns:
    #    scoresSBM_w[column][scoresSBM_w[column] >= fuzzy_cut_off] = 1 
    #    scoresSBM_w[column][scoresSBM_w[column] < fuzzy_cut_off] = 0

    #Modify the output scores by the weights set at the start of the code

    scoresSBM_w = scoresSBM_w*weights

    ### Determine matched roles that score above a threshold

    # Sum all columns 
    scoresSBM_r = scoresSBM_w.copy()

    scoresSBM_r['score'] = scoresSBM_r[matching_variables].sum(axis = 1)
    #scoresSBM_r.loc[scoresSBM_r['score'].isna(),'score'] = 0
    scoresSBM_r['score_max'] = sum(weights.values()) # + 2 for the additional scoring from the weighted variables a couple of cells above
    #- scoresSBM_r[matching_variables].isna().sum(axis = 1)
    scoresSBM_r['score_perc'] = scoresSBM_r['score'] / scoresSBM_r['score_max']
    #scoresSBM_r.loc[scoresSBM_r['score_perc'].isna(),'score_perc'] = 0

    # Sort by score, highest first 
    #print(scoresSBM_r)
                          
    scoresSBM_r = scoresSBM_r.reset_index()
                          
    #print(scoresSBM_r)
    # Rename the index if misnamed
    scoresSBM_r = scoresSBM_r.rename(columns={"index":"level_1"}, errors = "ignore")
    scoresSBM_r = scoresSBM_r.sort_values(by=["level_0","score_perc"], ascending = False)
    

    
    

    # Within each search, order descending by score and remove anything below the max
    #scoresSBM_r.to_csv("scoresSBM_r.csv")
    scoresSBM_g = scoresSBM_r.reset_index()

    
                          
    
    scoresSBM_g = scoresSBM_g.groupby("level_0").max("score_perc").reset_index()[["level_0", "score_perc"]]
    scoresSBM_g =scoresSBM_g.rename(columns={"score_perc":"score_perc_max"})

    scoresSBM_search_m = scoresSBM_r.merge(scoresSBM_g, on = "level_0", how="left")

    scoresSBM_search_m = scoresSBM_search_m[scoresSBM_search_m["score_perc"] == scoresSBM_search_m["score_perc_max"]]
 

    ## Join back search addresses onto matching df

    scoresSBM_search_m_j = scoresSBM_search_m.merge(ref_search, left_on="level_1", right_index=True, how = "left", suffixes=("", "_ref"))

    scoresSBM_search_m_j = scoresSBM_search_m_j.merge(predict_df_search, left_on="level_0", right_index=True,how="left", suffixes=("", "_pred"))

    scoresSBM_search_m_j = scoresSBM_search_m_j.reindex(sorted(scoresSBM_search_m_j.columns), axis=1)

                          
    
    ## Join on ref full address

    #scoresSBM_search_m_j = scoresSBM_search_m_j.merge(ref_search[["UPRN", "fulladdress"]].drop_duplicates("UPRN"), on="UPRN", how="left")

    ### Label rows that are above threshold score, reorder df

    # When blocking by street, need to have an increased threshold as this is more prone to making mistakes
    if blocker_column[0] == "Street": scoresSBM_search_m_j['full_match_score_based'] = (scoresSBM_search_m_j['score_perc'] >= 0.987)#0.9955)

    else: scoresSBM_search_m_j['full_match_score_based'] = (scoresSBM_search_m_j['score_perc'] >= score_cut_off)
    


    
    ### Reorder some columns
    
    start_columns = new_join_col.copy()

    start_columns.extend(["address", "fulladdress", "level_0", "level_1","score","score_max","score_perc","score_perc_max"])
    
    other_columns = list(set(scoresSBM_search_m_j.columns) - set(start_columns))

    all_columns_order = start_columns.copy()
    all_columns_order.extend(sorted(other_columns))
    

    # Place important columns at start

    scoresSBM_search_m_j = scoresSBM_search_m_j.reindex(all_columns_order, axis=1)

    scoresSBM_search_m_j = scoresSBM_search_m_j.rename(columns={'address':'address_pred',
     'fulladdress':'address_ref',
     'level_0':'index_pred',
     'level_1':'index_ref',
     'score':'match_score',
     'score_max':'max_possible_score',
     'score_perc':'perc_weighted_columns_matched',
     'score_perc_max':'perc_weighted_columns_matched_max_for_pred_address'})

    scoresSBM_search_m_j = scoresSBM_search_m_j.sort_values("index_pred", ascending = True)
    
    #search_df_j = orig_search_df[["full_address_search", search_df_key_field]]

    #scoresSBM_out = scoresSBM_search_m_j.merge(search_df_j, left_on = "address_pred", right_on = "full_address_search", how = "left")

    final_cols = new_join_col.copy()
    final_cols.extend([search_df_key_field, 'full_match_score_based', 'address_pred', 'address_ref',\
                                                  'match_score', 'max_possible_score', 'perc_weighted_columns_matched',\
                                                   'perc_weighted_columns_matched_max_for_pred_address',\
                                                   'SaoText', 'SaoText_ref', 'SaoText_pred',\
                                                   'SaoStartNumber', 'SaoStartNumber_ref', 'SaoStartNumber_pred',\
                                                   'SaoStartSuffix', 'SaoStartSuffix_ref', 'SaoStartSuffix_pred',\
                                                   'SaoEndNumber', 'SaoEndNumber_ref', 'SaoEndNumber_pred',\
                                                   'SaoEndSuffix', 'SaoEndSuffix_ref', 'SaoEndSuffix_pred',\
                                                   'PaoStartNumber', 'PaoStartNumber_ref', 'PaoStartNumber_pred',\
                                                   'PaoStartSuffix', 'PaoStartSuffix_ref', 'PaoStartSuffix_pred',\
                                                   'PaoEndNumber', 'PaoEndNumber_ref', 'PaoEndNumber_pred',\
                                                   'PaoEndSuffix', 'PaoEndSuffix_ref', 'PaoEndSuffix_pred',\
                                                   'PaoText', 'PaoText_ref', 'PaoText_pred',\
                                                   'Street', 'Street_ref', 'Street_pred',\
                                                   'PostTown', 'PostTown_ref', 'PostTown_pred',\
                                                   'Postcode', 'Postcode_ref', 'Postcode_pred', 'Postcode_predict',\
                                                   'index_pred', 'index_ref', 'Reference file'
                                                  ])
    
    scoresSBM_out = scoresSBM_search_m_j[final_cols]

    #scoresSBM_out.to_csv("scoresSBM_out" + "_" + blocker_column[0] + "_" + str(standardise) + ".csv")
    
    ''' Create 'best' results df '''

    scoresSBM_best = scoresSBM_out.sort_values([search_df_key_field, "perc_weighted_columns_matched"]).drop_duplicates(search_df_key_field)

    ### Make the final 'matched output' file

    scoresSBM_best_pred_cols = scoresSBM_best.filter(regex='_pred$').iloc[:,1:-1]

    scoresSBM_best["search_orig_address"] = (scoresSBM_best_pred_cols.agg(' '.join, axis=1)).str.strip().str.replace("  ", " ").str.replace("  ", " ").str.replace("  ", " ")

    scoresSBM_best_ref_cols = scoresSBM_best.filter(regex='_ref$').iloc[:,1:-1]

    scoresSBM_best['reference_mod_address'] = (scoresSBM_best_ref_cols.agg(' '.join, axis=1)).str.strip().str.replace("  ", " ").str.replace("  ", " ").str.replace("  ", " ")

    ## Create matched output df
    matched_output_SBM = orig_search_df[[search_df_key_field, "full_address"]]
    matched_output_SBM[search_df_key_field] = matched_output_SBM[search_df_key_field].astype(str)

    matched_output_SBM = matched_output_SBM.merge(scoresSBM_best[[search_df_key_field, 'address_ref',
                                                                  'full_match_score_based', 'Reference file']], on = search_df_key_field, how = "left").\
                                                                rename(columns={"full_address":"search_orig_address"})

    matched_output_SBM = matched_output_SBM.rename(columns={"full_match_score_based":"full_match"})

    matched_output_SBM['property_number_match'] = matched_output_SBM['full_match']
    
    scores_SBM_best_cols = [search_df_key_field, 'full_match_score_based',  'perc_weighted_columns_matched',
                                     'address_pred', "reference_mod_address"]
    scores_SBM_best_cols.extend(new_join_col)

    matched_output_SBM_b = scoresSBM_best[scores_SBM_best_cols]

    matched_output_SBM = matched_output_SBM.merge(matched_output_SBM_b, on = search_df_key_field,  how = "left")

    #matched_output_SBM["UPRN"] = scoresSBM_best['UPRN']

    matched_output_SBM['standardised_address'] = standardise

    matched_output_SBM = matched_output_SBM.rename(columns={"address_pred":"search_mod_address",
                                                        "address_ref":"reference_orig_address",
                                                        "full_match_score_based":"fuzzy_score_match",                                                        
                                                       'perc_weighted_columns_matched':"fuzzy_score"})

    matched_output_SBM_cols = [search_df_key_field, 'search_orig_address', 'reference_orig_address',
       'full_match', 'fuzzy_score_match', 'property_number_match',
       'fuzzy_score', 'search_mod_address', 'reference_mod_address', 'Reference file']
    
    matched_output_SBM_cols.extend(new_join_col)
    matched_output_SBM_cols.extend(['standardised_address'])
    matched_output_SBM = matched_output_SBM[matched_output_SBM_cols]
    
    matched_output_SBM = matched_output_SBM.sort_values(search_df_key_field, ascending=True)
    
    return scoresSBM, scoresSBM_out, scoresSBM_best, matched_output_SBM

def check_matches_against_fuzzy(match_results, scoresSBM, search_df_key_field):

    match_results = match_results.add_prefix("fuzz_").rename(columns={"fuzz_"+search_df_key_field:search_df_key_field})

    #Keep only full matches to compare with model output

    match_results_t = match_results[match_results["fuzz_full_match"] == True]

    scoresSBM_t = scoresSBM[scoresSBM["full_match_score_based"]==True]

    #Merge fuzzy match full matches onto model data

    scoresSBM_m = scoresSBM.merge(match_results.drop_duplicates(search_df_key_field), on = search_df_key_field, how = "left")

    ### Create a df of matches the model finds that the fuzzy matching work did not

    scoresSBM_m_model_add_matches = scoresSBM_m[(scoresSBM_m["full_match_score_based"] == True) &\
                                                         (scoresSBM_m["fuzz_full_match"] == False)]

    # Drop some irrelevant columns

    first_cols = ['UPRN', search_df_key_field, 'full_match_score_based', 'fuzz_full_match', 'fuzz_fuzzy_score_match', 'fuzz_property_number_match',\
                                   'fuzz_fuzzy_score', 'match_score', 'max_possible_score', 'perc_weighted_columns_matched',\
                                   'perc_weighted_columns_matched_max_for_pred_address', 'address_pred',\
                                   'address_ref', 'fuzz_reference_orig_address']

    last_cols = [col for col in scoresSBM_m_model_add_matches.columns if col not in first_cols]

    scoresSBM_m_model_add_matches = scoresSBM_m_model_add_matches[first_cols+last_cols].drop(['fuzz_search_mod_address',
       'fuzz_reference_mod_address', 'fuzz_fulladdress', 'fuzz_UPRN'], axis=1, errors="ignore")

    ### Create a df for matches the fuzzy matching found that the neural net model does not

    scoresSBM_t_model_failed = match_results[(~match_results[search_df_key_field].isin(scoresSBM_t[search_df_key_field])) &\
                                                      (match_results["fuzz_full_match"] == True)]

    scoresSBM_t_model_failed = scoresSBM_t_model_failed.\
        merge(scoresSBM.drop_duplicates(search_df_key_field), on = search_df_key_field, how = "left")

    scoresSBM_t_model_failed = scoresSBM_t_model_failed[first_cols+last_cols].drop(['fuzz_search_mod_address',
       'fuzz_reference_mod_address', 'fuzz_fulladdress', 'fuzz_UPRN'], axis=1, errors="ignore")
    
    ## Join back onto original results file and export

    scoresSBM_new_matches_from_model = scoresSBM_m_model_add_matches.drop_duplicates(search_df_key_field)

    match_results_out = match_results.merge(scoresSBM_new_matches_from_model[[search_df_key_field, 'full_match_score_based', 'address_pred',
       'address_ref']], on = search_df_key_field, how = "left")

    match_results_out.loc[match_results_out['full_match_score_based'].isna(),'full_match_score_based'] = False

    #match_results_out['full_match_score_based'].value_counts()

    match_results_out["full_match_fuzzy_or_score_based"] = (match_results_out["fuzz_full_match"] == True) |\
    (match_results_out["full_match_score_based"] == True)

    return scoresSBM_m_model_add_matches, scoresSBM_t_model_failed, match_results_out
# ## Overarching neural net matcher/standardisation function

# ## Overarching NN function

def perform_full_nn_match(ref, ref_address_cols, search_df, search_address_cols,
                       search_postcode_col, search_df_key_field, 
                      standardise, exported_model, matching_variables,
                       text_columns, weights, fuzzy_method, score_cut_off,
                         match_results, filter_to_lambeth_pcodes, 
                          model_type, word_to_index, cat_to_idx, device, vocab, labels_list, new_join_col=["UPRN"]):

    # Break if search item has length 0
    if len(search_df) == 0: 
        print("Nothing to match!")
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),\
        pd.DataFrame(),pd.DataFrame(),"Nothing to match!",search_address_cols
    
    ### Prepare ref data    
    ref_df = prepare_ref_address(ref, ref_address_cols, new_join_col)

    ## Prepare search_df data
    search_df_prep, search_df_key_field = prepare_search_address(search_df, search_address_cols, search_postcode_col, search_df_key_field)

    search_df_prep, ref_df, search_df_match_list, ref_df_match_list, search_df_stand_col, ref_df_stand_col =\
                                                standardise_wrapper_func(search_df_prep.copy(), ref_df,\
                                                standardise = standardise,
                                                filter_to_lambeth_pcodes = filter_to_lambeth_pcodes, match_task = "nnet")

    # Predict on search data to extract LPI address components

    predict_len = len(search_df_prep["full_address"])
    all_columns = list(search_df_prep) # Creates list of all column headers
    search_df_prep[all_columns] = search_df_prep[all_columns].astype(str)
    predict_data = list(search_df_stand_col)
    
    ### Run predict function
    print("Starting neural net prediction")

       
    # Commented out text was an attempt to save having to do the neural net predict every single time
    #predict_df = pd.read_csv(file_path)
    #print(pd.Series(predict_data).str.upper())
    #print(predict_df['address'])
    #print(predict_df['address'].isin(pd.Series(predict_data).str.upper()).value_counts())
    #predict_df = predict_df[predict_df['address'].isin(pd.Series(predict_data).str.upper())]
    #print(predict_df)


    max_predict_len = 12000

    tic = time.perf_counter()
    
    # Determine the number of chunks
    num_chunks = math.ceil(len(predict_data) / max_predict_len)
    list_out_all = []
    predict_df_all = []
    
    for i in range(num_chunks):
        #print("Starting to predict chunk " + str(i) + " of " + str(num_chunks) + " chunks.")
        
        start_idx = i * max_predict_len
        end_idx = start_idx + max_predict_len
        
        # Extract the current chunk of data
        chunk_data = predict_data[start_idx:end_idx]
        
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
    
    predict_df = post_predict_clean(predict_df=predict_df_all, orig_search_df=search_df_prep, 
                                       ref_address_cols=ref_address_cols, search_df_key_field=search_df_key_field)
    
    #if ~file_path.exists():
    #    print(f"'{file_path}' does not exist.")
    #    predict_df.to_csv(file_path, index = None)

    #print(predict_df)

    # Score-based matching between neural net predictions and fuzzy match results

    '''Example of recordlinkage package in use: https://towardsdatascience.com/how-to-perform-fuzzy-dataframe-row-matching-with-recordlinkage-b53ca0cb944c

    Make copies of the dfs for matching'''
    
    #

    #ref_df.to_csv("ref_df_before_nnet_output.csv")

    #predict_df_search = predict_df.copy()#.replace('',np.nan)
    #ref_search = ref_df.copy()#.replace('',np.nan)
    
    
    
    if standardise == True: standard_label = " standardised"
    else: standard_label = " not standardised"
    
    ## Run with Postcode as blocker column

    blocker_column = ["Postcode"]

    all_scores_pc, scoresSBM_out_pc, scoresSBM_best_pc, matched_output_SBM_pc = score_based_match(predict_df_search = predict_df.copy(), ref_search = ref_df.copy(),
                                                                                                        orig_search_df = search_df_prep, matching_variables = matching_variables,
                      text_columns = text_columns, blocker_column = blocker_column, weights = weights, fuzzy_method = fuzzy_method, score_cut_off = score_cut_off,
                                               search_df_key_field=search_df_key_field, standardise=standardise, new_join_col=new_join_col)


    #print(matched_output_SBM_pc)

    if matched_output_SBM_pc.empty:
        print("Match results empty")

        return pd.DataFrame(),pd.DataFrame(), "Nothing to match!", predict_df

        #scoresSBM_new_matches_from_model_pc, scoresSBM_fuzzy_matches_not_found_by_model_pc, match_results_out_pc = \
        #                                    check_matches_against_fuzzy(match_results=matched_output_SBM_pc,#match_results,
        #                                                                     scoresSBM=scoresSBM_best_pc, search_df_key_field=search_df_key_field)

        #match_results_output_final_pc = matched_output_SBM_pc
        #match_results_output_final_pc = combine_std_df_remove_dups(match_results, matched_output_SBM_pc, orig_addr_col = search_df_key_field) # match_results
        

    else:
        matched_output_SBM_pc["match_method"] = "Neural net - Postcode" #+ standard_label
        
        summary_base = create_match_summary(match_results, df_name = "baseline model (pre neural net matching)")
        print(summary_base)
        
        scoresSBM_new_matches_from_model_pc, scoresSBM_fuzzy_matches_not_found_by_model_pc, match_results_out_pc = \
                                            check_matches_against_fuzzy(match_results=match_results,
                                                                              scoresSBM=scoresSBM_best_pc, search_df_key_field=search_df_key_field)
    
        match_results_output_final_pc = combine_std_df_remove_dups(match_results, matched_output_SBM_pc, orig_addr_col = search_df_key_field) # match_results             
        
      
    summary_pc = create_match_summary(match_results_output_final_pc, df_name = "NNet blocked by Postcode" + standard_label)
    print(summary_pc)
    
    ## Run with Street as blocker column

    blocker_column = ["Street"]

    all_scores_st, scoresSBM_out_st, scoresSBM_best_st, matched_output_SBM_st = score_based_match(predict_df_search = predict_df.copy(), ref_search = ref_df.copy(), 
                                                                                                  orig_search_df = search_df_prep, matching_variables = matching_variables,
                      text_columns = text_columns, blocker_column = blocker_column, weights = weights, fuzzy_method = fuzzy_method, score_cut_off = score_cut_off,
                                               search_df_key_field=search_df_key_field, standardise=standardise, new_join_col=new_join_col)
    
    # If no matching pairs are found in the function above then it returns 0 - below we replace these values with the postcode blocker values
    # (which should almost always find at least one pair unless it's a very unusual situation
    if (type(matched_output_SBM_st) == int) | matched_output_SBM_st.empty:
        print("Nothing to match for street block")
        
        all_scores_st = all_scores_pc
        scoresSBM_out_st = scoresSBM_out_pc
        scoresSBM_best_st = scoresSBM_best_pc
        matched_output_SBM_st = matched_output_SBM_pc
        matched_output_SBM_st["match_method"] = "Neural net - Postcode" #+ standard_label
    else: matched_output_SBM_st["match_method"] = "Neural net - Street" #+ standard_label

    scoresSBM_new_matches_from_model_st, scoresSBM_fuzzy_matches_not_found_by_model_st, match_results_out_st = \
                                            check_matches_against_fuzzy(match_results=match_results_output_final_pc, 
                                                                        scoresSBM=scoresSBM_best_st, search_df_key_field=search_df_key_field)
    

    ### Join together old match df with new (model) match df

    match_results_output_final_st = combine_std_df_remove_dups(match_results_output_final_pc,matched_output_SBM_st, orig_addr_col = search_df_key_field)
      
    summary_street = create_match_summary(match_results_output_final_st, df_name = "NNet blocked by Street" + standard_label)
    print(summary_street)

    ''' I decided in the end not to use PaoStartNumber as a blocker column. I get only a couple more matches in general for a big increase in processing time '''
    
    ## Run with PaoStartNumber as blocker column
    
    #blocker_column = ["PaoStartNumber"]

    #all_scores_po, scoresSBM_out_po, scoresSBM_best_po, matched_output_SBM_po = score_based_match(predict_df_search = predict_df_search.copy(), ref_search = ref_search,
    #                                                    orig_search_df = search_df_prep, matching_variables = matching_variables,
    #                                                    text_columns = text_columns, blocker_column = blocker_column, weights = weights,
    #                                                    fuzzy_method = fuzzy_method, score_cut_off = score_cut_off, search_df_key_field=search_df_key_field,
    #                                                                                             standardise=standardise, new_join_col=new_join_col)

    # If no matching pairs are found in the function above then it returns 0 - below we replace these values with the postcode blocker values
    # (which should almost always find at least one pair unless it's a very unusual situation
    #if type(matched_output_SBM_po) == int:
    #    all_scores_po = all_scores_pc
    #    scoresSBM_out_po = scoresSBM_out_pc
     #   scoresSBM_best_po = scoresSBM_best_pc
    #    matched_output_SBM_po = matched_output_SBM_pc
     #   matched_output_SBM_po["match_method"] = "Neural net - Postcode"
    #else: matched_output_SBM_po["match_method"] = "Neural net - PaoStartNumber"
    
    #scoresSBM_new_matches_from_model_po, scoresSBM_fuzzy_matches_not_found_by_model_po, match_results_out_po = \
    #                                        check_matches_against_fuzzy(match_results=match_results_output_final_st,
    #                                                                          scoresSBM=scoresSBM_best_po, search_df_key_field=search_df_key_field)

    #match_results_output_final_po = combine_std_df_remove_dups(match_results_output_final_st, matched_output_SBM_po, orig_addr_col = search_df_key_field)
        
    #summary_po = create_match_summary(match_results_output_final_po, df_name = "NNet blocked by PaoStartNumber" + standard_label)
    #print(summary_po)
    
    all_scores_po = all_scores_st
    scoresSBM_out_po = scoresSBM_out_st
    scoresSBM_best_po = scoresSBM_best_st
    matched_output_SBM_po = matched_output_SBM_st
    matched_output_SBM_po["match_method"] = "Neural net - Street" #+ standard_label
    
    scoresSBM_new_matches_from_model_po = scoresSBM_new_matches_from_model_st
    scoresSBM_fuzzy_matches_not_found_by_model_po = scoresSBM_fuzzy_matches_not_found_by_model_st
    match_results_out_po = match_results_out_st
    match_results_output_final_po = match_results_output_final_st
    summary_po = summary_street
    
    ### Combine Street, Postcode, and PaoStartNumber blocker outputs together to get a combined output

    #match_results_output_final_both = combine_std_df_remove_dups(match_results_output_final_st, match_results_output_final_pc, orig_addr_col = search_df_key_field)
    #match_results_output_final_three = combine_std_df_remove_dups(match_results_output_final_both, match_results_output_final_po, orig_addr_col = search_df_key_field)

    match_results_output_final_three = match_results_output_final_po
    
    #summary_three = create_match_summary(match_results_output_final_three, df_name = "fuzzy and nn model street + postcode + paostartnumber" + standard_label)
    summary_three = create_match_summary(match_results_output_final_three, df_name = "fuzzy and nn model street + postcode" + standard_label)
   
    #print(summary_three)
    
    ### Join URPN back onto orig df

    if type(search_df) != str:
        results_on_orig_df = join_to_orig_df(match_results_output_final_three, search_df, search_df_key_field, new_join_col)
    else: results_on_orig_df = match_results_output_final_three
    
    return match_results_output_final_three, results_on_orig_df, summary_three, predict_df
