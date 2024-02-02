import os
import pandas as pd
import numpy as np
from typing import TypeVar, Dict, List, Tuple

import time
import re
import math
from datetime import datetime
import copy
import gradio as gr

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')
PandasSeries = TypeVar('pd.core.frame.Series')
MatchedResults = Dict[str,Tuple[str,int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# Constants
run_match = True
run_nnet_match = True
run_standardise = True

# Load in data functions

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

def get_file_name(in_name):
    # Corrected regex pattern
    match = re.search(r'\\(?!.*\\)(.*)', in_name)
    if match:
        matched_result = match.group(1)
    else:
        matched_result = None
    
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
    #print(search_df.columns)
    #print(key_col)
    
    matched = search_df.drop('level_0', axis = 1, errors="ignore").reset_index()[key_col].astype(str).isin(matched_results_success[key_col].astype(str)) # 

    #matched_results_success.to_csv("matched_results_success.csv")
    #matched.to_csv("matched.csv")
    #search_df.to_csv("search_df_at_match_removal.csv")
    
    return search_df.iloc[np.where(~matched)[0]] # search_df[~matched] 

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
            Matcher.ref_address_cols = ["Organisation", "SaoStartNumber", "SaoStartSuffix", "SaoEndNumber", "SaoEndSuffix", "SaoText", "PaoStartNumber", "PaoStartSuffix", "PaoEndNumber",
            "PaoEndSuffix", "PaoText", "Street", "PostTown", "Postcode"]
        else: 
            Matcher.standard_llpg_format = False
            Matcher.ref_address_cols = in_refcol#.tolist()[0]
            Matcher.ref = Matcher.ref.rename(columns={Matcher.ref_address_cols[-1]:"Postcode"})
            Matcher.ref_address_cols[-1] = "Postcode"

        # Reset index for ref as multiple files may have been combined with identical indices
        Matcher.ref = Matcher.ref.reset_index().drop("index", axis = 1)
        Matcher.ref.index.name = 'index'       
    
        if in_file: 
            Matcher.file_name = get_file_name(in_file[0].name)
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

from tools.preparation import prepare_search_address_string, prepare_search_address,  prepare_ref_address
from tools.standardise import standardise_wrapper_func, extract_street_name, remove_non_postal, check_no_number_addresses
from tools.fuzzy_match import string_match_by_post_code_multiple, _create_fuzzy_match_results_output, join_to_orig_df

# Neural network functions
### Predict function for imported model
from tools.model_predict import full_predict_func, full_predict_torch, post_predict_clean
from tools.recordlinkage_funcs import score_based_match, check_matches_against_fuzzy


# Run batch of matches
def run_match_batch(InitialMatch, batch_n, total_batches, progress=gr.Progress()):
    if run_match == True:
    
        overall_tic = time.perf_counter()
        
        progress(0, desc= "Batch " + str(batch_n+1) + " of " + str(total_batches) + ". Fuzzy match - non-standardised dataset")
        df_name = "Fuzzy not standardised"
                                    
        ''' FUZZY MATCHING '''
            
        ''' Run fuzzy match on non-standardised dataset '''
        
        FuzzyNotStdMatch = orchestrate_match_run(Matcher = copy.copy(InitialMatch), standardise = False, nnet = False, file_stub= "not_std_", df_name = df_name)

        if FuzzyNotStdMatch.abort_flag == True:
            print("Nothing to match!")
            return "Nothing to match!", InitialMatch

        FuzzyNotStdMatch = combine_two_matches(InitialMatch, FuzzyNotStdMatch, df_name)
        
        if (len(FuzzyNotStdMatch.search_df_not_matched) == 0) | (sum(FuzzyNotStdMatch.match_results_output[FuzzyNotStdMatch.match_results_output['full_match']==False]['fuzzy_score'])==0): 
            overall_toc = time.perf_counter()
            time_out = f"The fuzzy match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNotStdMatch.output_summary = FuzzyNotStdMatch.output_summary + " Neural net match not attempted. "# + time_out
            return FuzzyNotStdMatch.output_summary, FuzzyNotStdMatch
    
        ''' Run fuzzy match on standardised dataset '''
        
        progress(.25, desc="Batch " + str(batch_n+1) + " of " + str(total_batches) + ". Fuzzy match - standardised dataset")
        df_name = "Fuzzy standardised"
        
        FuzzyStdMatch = orchestrate_match_run(Matcher = copy.copy(FuzzyNotStdMatch), standardise = True, nnet = False, file_stub= "std_", df_name = df_name)
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

        if run_match == False:
            FuzzyStdMatch = copy.copy(InitialMatch)
            overall_tic = time.perf_counter()
    
        ''' First on non-standardised addresses '''
        progress(.50, desc="Batch " + str(batch_n+1) + " of " + str(total_batches) + ". Neural net - non-standardised dataset")
        df_name = "Neural net not standardised"
        
        FuzzyNNetNotStdMatch = orchestrate_match_run(Matcher = copy.copy(FuzzyStdMatch), standardise = False, nnet = True, file_stub= "nnet_not_std_", df_name = df_name)
        FuzzyNNetNotStdMatch = combine_two_matches(FuzzyStdMatch, FuzzyNNetNotStdMatch, df_name)
    
        if (len(FuzzyNNetNotStdMatch.search_df_not_matched) == 0):
            overall_toc = time.perf_counter()
            time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNNetNotStdMatch.output_summary = FuzzyNNetNotStdMatch.output_summary# + time_out
            return FuzzyNNetNotStdMatch.output_summary, FuzzyNNetNotStdMatch
    
        ''' Next on standardised addresses '''
        progress(.75, desc="Batch " + str(batch_n+1) + " of " + str(total_batches) + ". Neural net - standardised dataset")
        df_name = "Neural net standardised"
        
        FuzzyNNetStdMatch = orchestrate_match_run(Matcher = copy.copy(FuzzyNNetNotStdMatch), standardise = True, nnet = True, file_stub= "nnet_std_", df_name = df_name)
        FuzzyNNetStdMatch = combine_two_matches(FuzzyNNetNotStdMatch, FuzzyNNetStdMatch, df_name)
 
        if run_match == False:
            overall_toc = time.perf_counter()
            time_out = f"The neural net match script took {overall_toc - overall_tic:0.1f} seconds"
            FuzzyNNetStdMatch.output_summary = FuzzyNNetStdMatch.output_summary + " Only Neural net match attempted. "# + time_out
            return FuzzyNNetStdMatch.output_summary, FuzzyNNetStdMatch
    
    overall_toc = time.perf_counter()
    time_out = f"The whole match script took {overall_toc - overall_tic:0.1f} seconds"

    summary_of_summaries = FuzzyNotStdMatch.output_summary + "\n" + FuzzyStdMatch.output_summary + "\n" + FuzzyNNetStdMatch.output_summary + "\n" + time_out

    return summary_of_summaries, FuzzyNNetStdMatch

# Overarching functions
def orchestrate_match_run(Matcher, standardise = False, nnet = False, file_stub= "not_std_", df_name = "Fuzzy not standardised"):

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
                             full_nn_match(
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

# Overarching fuzzy match function
def full_fuzzy_match(search_df, ref, standardise, ref_address_cols, search_df_key_field, search_address_cols, search_postcode_col, fuzzy_match_limit, fuzzy_scorer_used, fuzzy_search_addr_limit = 100, filter_to_lambeth_pcodes=False, new_join_col=["UPRN"]):

    
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

    # Remove addresses that are not postal addresses
    search_df_prep = remove_non_postal(search_df_prep, "full_address")

    # Remove addresses that have no numbers in from consideration
    search_df_prep = check_no_number_addresses(search_df_prep, "full_address")

    #ref.to_csv("ref.csv")

    ref_df = prepare_ref_address(ref, ref_address_cols, new_join_col)
    
    #ref_df.to_csv("ref_df.csv")

    # Standardise addresses if required

    search_df_prep_join, ref_join, search_df_prep_match_list, ref_df_match_list, search_df_stand_col, ref_df_stand_col =\
                                                standardise_wrapper_func(search_df_prep.copy(), ref_df, standardise = standardise, filter_to_lambeth_pcodes=filter_to_lambeth_pcodes,
                                                                        match_task="fuzzy")

    
    # RUN WITH POSTCODE AS A BLOCKER #
    # Fuzzy match against reference addresses
    #pd.Series(ref_df_match_list).to_csv("ref_df_match_list.csv")
    #pd.Series(search_df_prep_match_list).to_csv("search_df_prep_match_list.csv")
    
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

    match_results_output, compare_all_candidates, diag_shortlist, diag_best_match = _create_fuzzy_match_results_output(results, search_df_prep_join, ref_df, ref_join, fuzzy_match_limit, search_df_prep, search_df_key_field, new_join_col, standardise, blocker_col = "Postcode")
    
    #compare_all_candidates.to_csv("compare_all_candidates.csv")

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
    ref_join['ref_address_stand_w_pcode'] = ref_join['ref_address_stand'] + " " + ref_join['postcode_search']
        
    search_df_prep_join_street['street']= search_df_prep_join_street['full_address_search'].apply(extract_street_name)
    # Exclude non-postal addresses from street-blocked search
    search_df_prep_join_street.loc[search_df_prep_join_street['Excluded from search'] == "Excluded - non-postal address", 'street'] = ""
        
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

    ### Join URPN back onto orig df

    if type(search_df) != str:
        results_on_orig_df = join_to_orig_df(match_results_output, search_df, search_df_key_field, new_join_col)
    else: results_on_orig_df = match_results_output
        
    return compare_all_candidates, diag_shortlist, diag_best_match, match_results_output, results_on_orig_df, summary, search_address_cols

# Overarching NN function
def full_nn_match(ref, ref_address_cols, search_df, search_address_cols,
                       search_postcode_col, search_df_key_field, 
                      standardise, exported_model, matching_variables,
                       text_columns, weights, fuzzy_method, score_cut_off, match_results, filter_to_lambeth_pcodes, 
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

    # Remove non-postal addresses from consideration
    search_df_prep = remove_non_postal(search_df_prep, "full_address")

    # Remove addresses that have no numbers in from consideration
    search_df_prep = check_no_number_addresses(search_df_prep, "full_address")


    search_df_prep, ref_df, search_df_match_list, ref_df_match_list, search_df_stand_col, ref_df_stand_col =\
        standardise_wrapper_func(search_df_prep.copy(), ref_df,\
        standardise = standardise,
        filter_to_lambeth_pcodes = filter_to_lambeth_pcodes, match_task = "nnet")
    
    #search_df_prep.to_csv("search_df_prep_nnet.csv")

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


    # Score-based matching between neural net predictions and fuzzy match results

    '''Example of recordlinkage package in use: https://towardsdatascience.com/how-to-perform-fuzzy-dataframe-row-matching-with-recordlinkage-b53ca0cb944c

    Make copies of the dfs for matching'''
    
    #
    
    if standardise == True: standard_label = " standardised"
    else: standard_label = " not standardised"
    
    ## Run with Postcode as blocker column

    blocker_column = ["Postcode"]

    all_scores_pc, scoresSBM_out_pc, scoresSBM_best_pc, matched_output_SBM_pc = score_based_match(predict_df_search = predict_df.copy(), ref_search = ref_df.copy(),
        orig_search_df = search_df_prep, matching_variables = matching_variables,
                      text_columns = text_columns, blocker_column = blocker_column, weights = weights, fuzzy_method = fuzzy_method, score_cut_off = score_cut_off, search_df_key_field=search_df_key_field, standardise=standardise, new_join_col=new_join_col)


    #print(matched_output_SBM_pc)

    if matched_output_SBM_pc.empty:
        print("Match results empty")

        return pd.DataFrame(),pd.DataFrame(), "Nothing to match!", predict_df

    else:
        matched_output_SBM_pc["match_method"] = "Neural net - Postcode" #+ standard_label
        
        #summary_base = create_match_summary(match_results, df_name = "baseline model (pre neural net matching)")
        #print(summary_base)
        
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
                      text_columns = text_columns, blocker_column = blocker_column, weights = weights, fuzzy_method = fuzzy_method, score_cut_off = score_cut_off, search_df_key_field=search_df_key_field, standardise=standardise, new_join_col=new_join_col)
    
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
    
    match_results_output_final_three = match_results_output_final_po
    
    summary_three = create_match_summary(match_results_output_final_three, df_name = "fuzzy and nn model street + postcode" + standard_label)
   
    ### Join URPN back onto orig df

    if type(search_df) != str:
        results_on_orig_df = join_to_orig_df(match_results_output_final_three, search_df, search_df_key_field, new_join_col)
    else: results_on_orig_df = match_results_output_final_three
    
    return match_results_output_final_three, results_on_orig_df, summary_three, predict_df

# Combiner/summary functions
def combine_std_df_remove_dups(df_not_std, df_std, orig_addr_col = "search_orig_address", match_col = "full_match",
                              keep_only_duplicated = False):

    if (df_not_std.empty) & (df_std.empty):
        return df_not_std

    combined_std_not_matches = pd.concat([df_not_std, df_std])#, ignore_index=True)

    if combined_std_not_matches.empty: #| ~(match_col in combined_std_not_matches.columns) | ~(orig_addr_col in combined_std_not_matches.columns):
        combined_std_not_matches[match_col] = False

        if "full_address" in combined_std_not_matches.columns:
            combined_std_not_matches[orig_addr_col] = combined_std_not_matches["full_address"]
        combined_std_not_matches["fuzzy_score"] = 0
        return combined_std_not_matches

    combined_std_not_matches = combined_std_not_matches.sort_values([orig_addr_col, match_col], ascending=False)

    if keep_only_duplicated == True:
        combined_std_not_matches = combined_std_not_matches[combined_std_not_matches.duplicated(orig_addr_col)]
    
    combined_std_not_matches_no_dups = combined_std_not_matches.drop_duplicates(orig_addr_col)
    
    return combined_std_not_matches_no_dups

def combine_two_matches(OrigMatchClass, NewMatchClass, df_name):

        today_rev = datetime.now().strftime("%Y%m%d")

        NewMatchClass.match_results_output = combine_std_df_remove_dups(OrigMatchClass.match_results_output, NewMatchClass.match_results_output, orig_addr_col = NewMatchClass.search_df_key_field)    
        NewMatchClass.results_on_orig_df = combine_std_df_remove_dups(OrigMatchClass.pre_filter_search_df, NewMatchClass.results_on_orig_df, orig_addr_col = NewMatchClass.search_df_key_field, match_col = 'Matched with ref record') #OrigMatchClass.results_on_orig_df
        NewMatchClass.pre_filter_search_df = NewMatchClass.results_on_orig_df
        
        # Identify records where the match score was 0
        #if ~('fuzzy_score' in NewMatchClass.match_results_output.columns):
        #    NewMatchClass.match_results_output['fuzzy_score'] = pd.NA
        #    return NewMatchClass

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
    records_attempted = int(sum((match_results_output['fuzzy_score']!=0.0) & ~(match_results_output['fuzzy_score'].isna())))
    dataset_length = len(match_results_output["full_match"])
    records_not_attempted = int(dataset_length - records_attempted)
    match_rate = str(round((full_match_count / dataset_length) * 100,1))
    match_fail_count_without_excluded = match_fail_count - records_not_attempted
    match_fail_rate = str(round(((match_fail_count_without_excluded) / dataset_length) * 100,1))
    not_attempted_rate = str(round((records_not_attempted / dataset_length) * 100,1))

    summary = ("For the " + df_name + " dataset (" + str(dataset_length) + " records), the fuzzy matching algorithm successfully matched " + str(full_match_count) +
               " records (" + match_rate + "%). The algorithm had no success matching " + str(records_not_attempted) +
               " records (" + not_attempted_rate +  "%). There are " + str(match_fail_count_without_excluded) + " records left to potentially match.")
    
    return summary
