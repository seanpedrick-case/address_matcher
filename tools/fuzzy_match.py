import pandas as pd
import numpy as np
from typing import TypeVar, Dict, List, Tuple
from datetime import datetime
from rapidfuzz import fuzz, process

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')
PandasSeries = TypeVar('pd.core.frame.Series')
MatchedResults = Dict[str,Tuple[str,int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

def string_match_array(to_match:array, choices:array,
                      index_name:str, matched_name:str) -> PandasDataFrame:
    
    temp = {name: process.extractOne(name,choices) 
            for name in to_match}
    
    return _create_frame(matched_results=temp, index_name=index_name,
                        matched_name=matched_name)

# Fuzzy match algorithm
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

def _create_fuzzy_match_results_output(results, search_df_prep_join, ref_df, ref_join, fuzzy_match_limit, search_df_prep, search_df_key_field, new_join_col, standardise, blocker_col):

        ## Diagnostics

        compare_all_candidates, diag_shortlist, diag_best_match =\
                                      refine_export_results(results_df=results,\
                                      matched_df = search_df_prep_join, ref_list_df = ref_join,
                                      fuzzy_match_limit = fuzzy_match_limit, blocker_col=blocker_col)
        
        ## Fuzzy search results

        match_results_cols = ['search_orig_address','reference_orig_address',
        'full_match',
        'full_number_match',
        'flat_number_match',
        'room_number_match',
        'block_number_match',
        'unit_number_match',
        'property_number_match',
        'close_postcode_match',
        'house_court_name_match',
        'fuzzy_score_match',
        "fuzzy_score",
        "wratio_score",
        'property_number_search', 'property_number_reference',  
        'flat_number_search', 'flat_number_reference', 
        'room_number_search', 'room_number_reference',
        'unit_number_search', 'unit_number_reference',
        'block_number_search', 'block_number_reference',
        'house_court_name_search', 'house_court_name_reference',
        "search_mod_address", 'reference_mod_address','Postcode']

        # Join results data onto the original housing list to create the full output
        match_results_output = pd.merge(search_df_prep[[search_df_key_field, "full_address","postcode"]],\
                             diag_best_match[match_results_cols], how = "left", left_on = "full_address", right_on = "search_orig_address").\
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

def create_diag_shortlist(diag_j, matched_col, fuzzy_match_limit, blocker_col, fuzzy_col="fuzzy_score", search_mod_address = "search_mod_address", resolve_tie_breaks=True):
        '''
        Create a shortlist of the best matches from a list of suggested matches
        '''

        ## Calculate highest fuzzy score from all candidates, keep all candidates with matching highest fuzzy score
        results_max_fuzzy_score = diag_j.groupby(matched_col)[fuzzy_col].max().reset_index().rename(columns={fuzzy_col: "max_fuzzy_score"})

        diag_shortlist = pd.merge(diag_j, results_max_fuzzy_score, how = "left", on = matched_col)

        diag_shortlist = diag_shortlist[(diag_shortlist[fuzzy_col] == diag_shortlist["max_fuzzy_score"])]# | (diag_shortlist["max_fuzzy_score"].isnull())]

        #diag_shortlist.to_csv("diag_shortlist.csv")

        # Fuzzy match limit for records with no numbers in it is 0.95 or the provided fuzzy_match_limit, whichever is higher
        diag_shortlist["fuzzy_score_match"] = diag_shortlist[fuzzy_col] >= fuzzy_match_limit

        if fuzzy_match_limit > 95: no_number_fuzzy_match_limit = fuzzy_match_limit
        else: no_number_fuzzy_match_limit = fuzzy_match_limit

        ### Count number of numbers in search string
        diag_shortlist["number_count_search_string"] =  diag_shortlist[search_mod_address].str.count(r'\d')
        diag_shortlist["no_numbers_in_search_string"] = diag_shortlist["number_count_search_string"] == 0

        # Replace fuzzy_score_match values for addresses with no numbers in them
        diag_shortlist.loc[(diag_shortlist["no_numbers_in_search_string"]==True) & (diag_shortlist[fuzzy_col] >= no_number_fuzzy_match_limit), "fuzzy_score_match"] = True
        diag_shortlist.loc[(diag_shortlist["no_numbers_in_search_string"]==True) & (diag_shortlist[fuzzy_col] < no_number_fuzzy_match_limit), "fuzzy_score_match"] = False

        # If blocking on street, don't match addresses with 0 numbers in. There are too many options and the matches are rarely good
        if blocker_col == "Street":
            diag_shortlist.loc[(diag_shortlist["no_numbers_in_search_string"]==True), "fuzzy_score_match"] = False
                                
        diag_shortlist = diag_shortlist.fillna("").drop(["number_count_search_string", "no_numbers_in_search_string"], axis = 1)

        # Following considers full matches to be those that match on property number and flat number, and the postcode is relatively close.
        #print(diag_shortlist.columns) 
        diag_shortlist["property_number_match"] = (diag_shortlist["property_number_search"] == diag_shortlist["property_number_reference"])
        diag_shortlist["flat_number_match"] = (diag_shortlist['flat_number_search'] == diag_shortlist['flat_number_reference'])
        diag_shortlist["room_number_match"] = (diag_shortlist['room_number_search'] == diag_shortlist['room_number_reference'])
        diag_shortlist["block_number_match"] = (diag_shortlist['block_number_search'] == diag_shortlist['block_number_reference'])
        diag_shortlist["unit_number_match"] = (diag_shortlist['unit_number_search'] == diag_shortlist['unit_number_reference'])
        diag_shortlist["house_court_name_match"] = (diag_shortlist['house_court_name_search'] == diag_shortlist['house_court_name_reference'])

        # Full number match is currently considered only a match between property number and flat number
                                
        diag_shortlist['full_number_match'] = (diag_shortlist["property_number_match"] == True) &\
            (diag_shortlist["flat_number_match"] == True) &\
            (diag_shortlist["room_number_match"] == True) &\
            (diag_shortlist["block_number_match"] == True) &\
            (diag_shortlist["unit_number_match"] == True) &\
            (diag_shortlist["house_court_name_match"] == True)
    
        
        ### Postcodes need to be close together, so all the characters should match apart from the last two 
        diag_shortlist['close_postcode_match'] = diag_shortlist['postcode'].str.lower().str.replace(" ","").str[:-2] == diag_shortlist['Postcode'].str.lower().str.replace(" ","").str[:-2]
          
        
        diag_shortlist["full_match"] = (diag_shortlist["fuzzy_score_match"] == True) &\
            (diag_shortlist['full_number_match'] == True) &\
            (diag_shortlist['close_postcode_match'] == True)
        
        #diag_shortlist = diag_shortlist.rename(columns = {"flat_number_search":"search_flat_number",
        #    "room_number_search":"search_room_number",
        #    "block_number_search":"search_block_number",
        #    "property_number_search":"search_property_number",
        diag_shortlist = diag_shortlist.rename(columns = {"reference_list_address":"reference_mod_address"})

        '''
        If a matched address is duplicated, choose the version that has a number match, 
        if there is no number match, then property match, room number match, then best fuzzy score, then show all options
        '''
        
        ### Dealing with tie breaks ##
        # Do a backup simple Wratio search on the open text to act as a tie breaker when the fuzzy scores are identical
        # fuzz.WRatio
        if resolve_tie_breaks == True:
            def compare_strings_wratio(row, scorer = fuzz.ratio, fuzzy_col = fuzzy_col):
                search_score = process.cdist([row[search_mod_address]], [row["reference_mod_address"]], scorer=scorer)
                return search_score[0][0]

            diag_shortlist_dups = diag_shortlist[diag_shortlist['full_number_match'] == True]
            diag_shortlist_dups = diag_shortlist_dups.loc[diag_shortlist_dups.duplicated(subset= [search_mod_address, 'full_number_match', "room_number_search", fuzzy_col], keep=False)]

            if not diag_shortlist_dups.empty:
                diag_shortlist_dups["wratio_score"] = diag_shortlist_dups.apply(compare_strings_wratio, axis=1)
                                    
                diag_shortlist = diag_shortlist.merge(diag_shortlist_dups[["wratio_score"]], left_index=True, right_index=True, how = "left")
                                    
                # Choose the best match
                diag_shortlist = diag_shortlist.sort_values([search_mod_address, 'full_number_match', "room_number_search", fuzzy_col, "wratio_score"], ascending = False)

        if 'wratio_score' not in diag_shortlist.columns:
            diag_shortlist['wratio_score'] = ''

        return diag_shortlist

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

    reference_j = ref_list_df[[final_ref_address_col, "property_number","flat_number","room_number","block_number", "unit_number", 'house_court_name',
            orig_ref_address_col,"Postcode"\
    ]].rename(columns={"property_number": "property_number_reference",\
                        "flat_number":"flat_number_reference",\
                        "room_number":"room_number_reference",
                        "block_number":"block_number_reference",\
                        "unit_number":"unit_number_reference",\
                        'house_court_name':'house_court_name_reference',\
                        orig_ref_address_col: "reference_orig_address",\
                    final_ref_address_col:'reference_list_address'             
                    })

    results_join = results_join.merge(reference_j, how = "left", left_on = ref_list_col, right_on = "reference_list_address")

    matched_j = matched_df[[final_matched_address_col,"property_number","flat_number","room_number", "block_number", "unit_number", 'house_court_name',
        orig_matched_address_col, "postcode",\
            ]].rename(columns={"property_number": "property_number_search",\
                            "flat_number":"flat_number_search",\
                            "room_number":"room_number_search",\
                            "block_number":"block_number_search",\
                            "unit_number":"unit_number_search",\
                            'house_court_name':'house_court_name_search',\
                            orig_matched_address_col:"search_orig_address",\
                            final_matched_address_col:'search_mod_address'
                            })

    diag_j = results_join.merge(matched_j, how = "left", left_on = matched_col, right_on = "search_mod_address")

    #diag_j.to_csv("diag_j.csv")
    
    diag_shortlist = create_diag_shortlist(diag_j, matched_col, fuzzy_match_limit, blocker_col)

    match_results_cols = ['search_orig_address','reference_orig_address',
        'full_match',
        'full_number_match',
        'flat_number_match',
        'room_number_match',
        'block_number_match',
        'unit_number_match',
        'house_court_name_match',
        'property_number_match',
        'close_postcode_match',
        'fuzzy_score_match',
        "fuzzy_score",
        "wratio_score",
        'property_number_search', 'property_number_reference',  
        'flat_number_search', 'flat_number_reference', 
        'room_number_search', 'room_number_reference',
        'block_number_search', 'block_number_reference',
        'unit_number_search', 'unit_number_reference',
        'house_court_name_search', 'house_court_name_reference',
        "search_mod_address", 'reference_mod_address', 'postcode','Postcode']

    diag_shortlist = diag_shortlist[match_results_cols] # , 'uprn'

    # Choose best match from the shortlist that has been ordered according to score descending
    diag_best_match = diag_shortlist[match_results_cols].drop_duplicates("search_mod_address")


    diag_shortlist.to_csv("diagnostics_shortlist_" + today_rev + ".csv", index=None)
   
    return compare_all_candidates, diag_shortlist, diag_best_match

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

    #search_df_j = search_df_j.rename(columns={"full_address":"Combined search address"})

    #print(search_df_j.index)

    # Only keep 
    #search_df_j = search_df_j[search_df_j.index.notna() & (search_df_j.index != '')]
    
    
    return search_df_j