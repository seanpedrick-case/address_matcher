import pandas as pd
import polars as pl
import numpy as np
from typing import Type, Dict, List, Tuple
from datetime import datetime

PandasDataFrame = Type[pd.DataFrame]
PandasSeries = Type[pd.Series]
MatchedResults = Dict[str,Tuple[str,int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

# # Standardisation functions

def standardise_wrapper_func(search_df, ref_df,\
                               standardise = False, filter_to_lambeth_pcodes = True, match_task = "fuzzy"):
    
    ## Search df - lower case addresses, replace spaces in postcode and 'AT' in addresses

    #assert not search_df['postcode'].isna().any() , "nulls in search_df subset post code"

    search_df["full_address_search"] = search_df["full_address"].str.lower()

    # Remove the 'AT's that appear everywhere
    search_df["full_address_search"] = search_df["full_address_search"]

    search_df['postcode_search'] = search_df['postcode'].str.lower().str.strip().str.replace(" ", "",regex = False)

    # Filter out records where 'Excluded from search' is not a postal address by making the postcode blank
    search_df.loc[search_df['Excluded from search'] == "Excluded - non-postal address", 'postcode_search'] = ""

    #assert not ref_df['Postcode'].isna().any() , "nulls in ref_df subset post code"
    # Remove nulls from ref postcode
    ref_df = ref_df[ref_df['Postcode'].notna()]
    
    ref_df["full_address_search"] = ref_df["fulladdress"].str.lower().str.strip()
    ref_df['postcode_search'] = ref_df['Postcode'].str.lower().str.strip().str.replace(" ", "", regex = False)
    
    # Block only on first 5 characters of postcode string - Doesn't give more matches and makes everything a bit slower
    # search_df['postcode_search'] = search_df['postcode_search'].str[:-1]


    ### Use standardise function

    ### Remove 'non-housing' places from the list - not included as want to check all
    #search_df_join = remove_non_housing(search_df, 'full_address_search')
    search_df_join, search_df_stand_col = standardise_address(search_df, "full_address_search", "search_address_stand", standardise = standardise, out_london = True)

    ## Standardise ref addresses

    # Block only on first 5 characters of postcode string - Doesn't give more matches and makes everything a bit slower
    # ref_df['postcode_search'] = ref_df['postcode_search'].str[:-1]

    ### Remove 'non-housing' places from the list
    #ref_df_join = remove_non_housing(ref_df, 'full_address_search')

    if match_task == "fuzzy":
        ref_join, ref_df_stand_col = standardise_address(ref_df, "full_address_search", "ref_address_stand", standardise = standardise, out_london = True)
    else:
        # I FOUND THAT THE STANDARDISATION PROCESS ON REF FOR THE NEURAL NET PART DID NOT HELP THE MODEL AT ALL, IN FACT IT REDUCED MATCHES AS STANDARDISING INDIVIDUAL REF COLUMNS GIVES YOU DIFFERENT RESULTS
        # FROM STANDARDISING THE WHOLE ADDRESS, THEN BREAKING IT DOWN. SO DON'T STANDARDISE. THE MODEL WILL JUST STANDARDISE THE INPUT ADDRESSES ONLY
        ref_join, ref_df_stand_col = standardise_address(ref_df, "full_address_search", "ref_address_stand", standardise = False, out_london = True)
    
    ### Create lookup lists
    search_df_match_list = search_df_join.copy().set_index('postcode_search')['search_address_stand'].str.lower().str.strip()
    ref_df_match_list = ref_join.copy().set_index('postcode_search')['ref_address_stand'].str.lower().str.strip()
    
    return search_df_join, ref_join, search_df_match_list, ref_df_match_list, search_df_stand_col, ref_df_stand_col


def standardise_address(df, col:str, out_col:str, standardise:bool = True, out_london = True):
    
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

    # Convert to polars dataframe
    df_copy = pl.from_pandas(df_copy)
    
    # Trim the address to remove leading and tailing spaces
    df_copy = df_copy.with_columns(pl.col(col).str.strip_chars())
    
    
    ''' Remove the post code and 'london' from the address to reduce the text the algorithm has to search
    Using a regex to extract a UK postcode. I got the regex from the following. Need to replace their \b in the solution with \\b
    https://stackoverflow.com/questions/51828712/r-regular-expression-for-extracting-uk-postcode-from-an-address-is-not-ordered
        
    The following will pick up whole postcodes, postcodes with just the first part, and postcodes with the first
    part and the first number of the second half
    '''
    
    
    #df_copy['add_no_pcode'] = remove_postcode(df_copy, col)
    #df_copy =  df_copy.with_columns(remove_postcode(df_copy, col), 'add_no_pcode')

    df_copy =  df_copy.with_columns(add_no_pcode = remove_postcode(df_copy, col))

    # Use remove_postcode() function to process the column
    #processed_column = remove_postcode(df_copy, col)
    # Create a named expression with the processed column and the desired column name
    
    #     df.with_columns(
    #     new_column = pl.lit('some_text')
    # )
    
    #named_expr = (processed_column, 'add_no_pcode')

    # Use with_columns() method with the named expression
    #df_copy = df_copy.with_columns(named_expr)
    
    if out_london == False:
        df_copy['add_no_pcode'] = df_copy.with_columns(pl.col('add_no_pcode').str.replace("london","").str.replace(r",,|, ,","", ))
    
    # If the user wants to standardise the address
    if standardise:
        
        df_copy['add_no_pcode'] = df_copy.with_columns(pl.col('add_no_pcode').str.to_lowercase())

        # If there are dates at the start of the address, change this
        #df_copy['add_no_pcode'] = replace_mistaken_dates(df_copy, 'add_no_pcode')
        df_copy =  df_copy.with_columns(replace_mistaken_dates(df_copy, 'add_no_pcode'), 'add_no_pcode')

        # Replace flat name variations with flat, abbreviations with full name of item (e.g. rd to road)
        df_copy['add_no_pcode'] = df_copy.with_columns(pl.col('add_no_pcode').str.replace(r"\brd\b","road", ).\
                                                str.replace(r"\bst\b","street", ).\
                                                str.replace(r"\bave\b","avenue", ).\
                                                str.replace("'", "", literal=True).\
                                                str.replace(r"\bat\b ", " ",).\
                                                str.replace("apartment", "flat",literal=True).\
                                                str.replace("studio flat", "flat",literal=True).\
                                                str.replace("cluster flat", "flats",literal=True).\
                                                str.replace(r"\bflr\b", "floor", ).\
                                                str.replace(r"\bflrs\b", "floors", ).\
                                                str.replace(r"\blwr\b", "lower", ).\
                                                str.replace(r"\bgnd\b", "ground", ).\
                                                str.replace(r"\blgnd\b", "lower ground", ).\
                                                str.replace(r"\bgrd\b", "ground", ).\
                                                str.replace(r"\bmais\b", "flat", ).\
                                                str.replace(r"\bmaisonette\b", "flat", ).\
                                                str.replace(r"\bpt\b", "penthouse", ).\
                                                str.replace(r"\bbst\b","basement", ).\
                                                str.replace(r"\bbsmt\b","basement", ))
        
        #df_copy["add_no_pcode_house"] = move_flat_house_court(df_copy)
        df_copy = df_copy.with_columns(move_flat_house_court(df_copy), 'add_no_pcode_house')
    
        # Replace any addresses that don't have a space between the comma and the next word. and double spaces # df_copy['add_no_pcode_house']
        df_copy['add_no_pcode_house_comma'] = df_copy.with_columns(pl.col('add_no_pcode').str.replace(r',(\w)', r', \1', ).str.replace('  ', ' ', literal=True))

        # Replace number / number and number-number with number
        df_copy['add_no_pcode_house_comma_no'] = df_copy.with_columns(pl.col('add_no_pcode_comma').str.replace(r'(\d+)\/(\d+)', r'\1', \
                                                                                                ).str.replace(r'(\d+)-(\d+)', r'\1', \
                                                                                                ).str.replace(r'(\d+) - (\d+)', r'\1', ))

        # Add 'flat' to the start of addresses that include ground/first/second etc. floor flat in the text
        #df_copy['floor_replacement'] = replace_floor_flat(df_copy, 'add_no_pcode_house_comma_no')
        df_copy = df_copy.with_columns(replace_floor_flat(df_copy, 'add_no_pcode_house_comma_no'), 'floor_replacement')


        #df_copy['flat_added_to_start_addresses_begin_letter'] = add_flat_addresses_start_with_letter(df_copy, 'floor_replacement')
        df_copy = df_copy.with_columns(add_flat_addresses_start_with_letter(df_copy, 'floor_replacement'), 'flat_added_to_start_addresses_begin_letter')

        #df_copy[out_col] = merge_series(df_copy['add_no_pcode_house_comma_no'], df_copy['flat_added_to_start_addresses_begin_letter'])
        df_copy = df_copy.with_columns(merge_series(pl.col('add_no_pcode_house_comma_no'), pl.col('flat_added_to_start_addresses_begin_letter')), out_col)

        # Write stuff back to the original df
        #df[out_col] = df_copy[out_col]
        df.with_columns(pl.Series(name="out_col", values=df_copy[out_col])) 

    
    else:
        #df_copy[out_col] = df_copy['add_no_pcode']
        #df_copy.with_columns(pl.Series(name=out_col, values=df_copy['add_no_pcode'])) 

        #df_copy[out_col] = df_copy.with_columns(pl.col('add_no_pcode'))
        #df_copy = df_copy.with_columns(df_copy[out_col])
        df_copy = df_copy.with_columns(out_col, pl.col('add_no_pcode'))

        #df[out_col] = df_copy['add_no_pcode']
        #df.with_columns(pl.Series(name=out_col, values=df_copy['add_no_pcode'])) 
        #processed_series = df_copy['add_no_pcode']
        df = df.with_columns(df_copy['add_no_pcode'].alias(out_col))

    ## POST STANDARDISATION CLEANING AND INFORMATION EXTRACTION  
    # Remove trailing spaces
    df[out_col] = df.with_columns(pl.col(out_col).str.strip_chars())
    
    # Pull out property, flat, and room numbers from the address text
    #df['property_number'] = extract_prop_no(df_copy, out_col)
    df = df_copy.with_columns(extract_prop_no(df_copy, out_col), 'property_number')

    # Extract flat, apartment numbers
    #df = extract_flat_and_other_no(df, out_col)
    processed_df = extract_flat_and_other_no(df, out_col)
    # Adding the new columns to the original DataFrame
    df = df.with_columns(processed_df)
    
    # df['flat_number'] = merge_series(df['flat_number'], df['apart_number'])
    # df['flat_number'] = merge_series(df['flat_number'], df['prop_number'])
    # df['flat_number'] = merge_series(df['flat_number'], df['first_sec_number'])
    # df['flat_number'] = merge_series(df['flat_number'], df['first_letter_flat_number'])
    # df['flat_number'] = merge_series(df['flat_number'], df['first_letter_no_more_numbers'])

    df = df.with_columns(merge_series(pl.col('flat_number'), pl.col('apart_number')), 'flat_number')
    df = df.with_columns(merge_series(pl.col('flat_number'), pl.col('prop_number')), 'flat_number')
    df = df.with_columns(merge_series(pl.col('flat_number'), pl.col('first_sec_number')), 'flat_number')
    df = df.with_columns(merge_series(pl.col('flat_number'), pl.col('first_letter_flat_number')), 'flat_number')
    df = df.with_columns(merge_series(pl.col('flat_number'), pl.col('first_letter_no_more_numbers')), 'flat_number')
    
    # Extract room numbers
    #df['room_number'] = extract_room_no(df, out_col)
    df = df_copy.with_columns(extract_room_no(df, out_col), 'room_number')

    # Extract block and unit names
    #df = extract_block_and_unit_name(df, out_col)
    processed_df = extract_block_and_unit_name(df, out_col)
    # Adding the new columns to the original DataFrame
    df = df.with_columns(processed_df)

    # Extract house or court name
    #df['house_court_name'] = extract_house_or_court_name(df, out_col)
    df = df_copy.with_columns(extract_house_or_court_name(df, out_col), 'house_court_name')

    # convert back to pandas
    df = df.to_pandas()

    return df, df[out_col]



def move_flat_house_court(df_copy):
            ''' Remove 'flat' from any address that contains 'house' or 'court'
            From the df_copy address, remove the word 'flat' from any address that contains the word 'house' or 'court'
            This is because in the housing list, these addresses never have the word flat in front of them
            '''
            
            # Remove the word flat or apartment from addresses that have only one number in it. 'Flat' will be re-added later to relevant addresses 
            # that need it (replace_floor_flat)
            df_copy['flat_removed'] = remove_flat_one_number_address(df_copy, 'add_no_pcode')
        
        
            
            remove_flat_house = df_copy['flat_removed'].str.to_lowercase().str.contains(r"\bhouse\b")#(?=\bhouse\b)(?!.*house road)")
            remove_flat_court = df_copy['flat_removed'].str.to_lowercase().str.contains(r"\bcourt\b")#(?=\bcourt\b)(?!.*court road)")
            remove_flat_terrace = df_copy['flat_removed'].str.to_lowercase().str.contains(r"\bterrace\b")#(?=\bterrace\b)(?!.*terrace road)")
            remove_flat_house_or_court = (remove_flat_house | remove_flat_court | remove_flat_terrace == 1)

            df_copy['remove_flat_house_or_court'] = remove_flat_house_or_court
            df_copy['house_court_replacement'] = "flat " + df_copy[df_copy['remove_flat_house_or_court'] == True]['flat_removed'].str.replace(r"\bflat\b","", \
                                                                                                                                            ).str.strip_chars().map(str)       
            #df_copy["add_no_pcode_house"] = merge_columns(df_copy, "add_no_pcode_house", 'flat_removed', "house_court_replacement")

            #merge_columns(df, "new_col", col, 'letter_after_number')
            df_copy["add_no_pcode_house"] = merge_series(df_copy['flat_removed'], df_copy["house_court_replacement"])

            return df_copy["add_no_pcode_house"]

def remove_postcode(df, col: str):
    '''
    Remove a postcode from a string column in a dataframe
    '''

    # Remove postcodes from the specified column
    address_series_no_pcode = (
        df[col]
        .str
        .to_uppercase()
        .str
        .replace(
            "\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9][A-Z]{2}|GIR ?0A{2})\\b$|(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]? ?[0-9]{1}?)$|\\b(?:[A-Z][A-HJ-Y]?[0-9][0-9A-Z]?)\\b$",
            ""
        )
        .str
        .to_lowercase()
    )

    return address_series_no_pcode



def add_flat_addresses_start_with_letter(df, col):
    df['contains_single_letter_at_start_before_number'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r'^\b[A-Za-z]\b[^\d]* \d'))

    df['selected_rows'] = (df['contains_single_letter_at_start_before_number'] == True)
    df['flat_added_to_string_start'] =  "flat " + df[df['selected_rows'] == True][col]
    
    #merge_columns(df, "new_col", col, 'flat_added_to_string_start')
    df["new_col"] = merge_series(df[col], df['flat_added_to_string_start'])
    
    
    return df['new_col']

def extract_letter_one_number_address(df, col):
    '''
    This function looks for addresses that have a letter after a number, but ONLY one number
    in the string, and doesn't already have a flat, apartment, or room number. 
        
    It then extracts this letter and returns this.
    
    This is for addresses such as '2b sycamore road', changes it to
    flat b 2 sycamore road so that 'b' is selected as the flat number

    
    '''
    
    df['contains_no_numbers_without_letter'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r"^(?:(?!\d+ ).)*$"))
    df['contains_letter_after_number'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r"\d+(?:[a-z]|[A-Z])(?!.*\d+)")  )    
    df['contains_apartment'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r"\bapartment\b \w+|\bapartments\b \w+"))
    df['contains_flat'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r"\bflat\b \w+|\bflats\b \w+"))
    df['contains_room'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r"\broom\b \w+|\brooms\b \w+"))
        
    df['selected_rows'] = (df['contains_no_numbers_without_letter'] == True) &\
                             (df['contains_letter_after_number'] == True) &\
                             (df['contains_flat'] == False) &\
                             (df['contains_apartment'] == False) &\
                             (df['contains_room'] == False)
            
    df['extract_letter'] =  df[(df['selected_rows'] == True)\
                                  ][col].str.extract(r"\d+([a-z]|[A-Z])")
    
    df['extract_number'] =  df[(df['selected_rows'] == True)\
                                  ][col].str.extract(r"(\d+)[a-z]|[A-Z]")
    

    df['letter_after_number'] = "flat " +\
                                df[(df['selected_rows'] == True)\
                                  ]['extract_letter'] +\
                                " " +\
                                df[(df['selected_rows'] == True)\
                                  ]['extract_number'] +\
                                " " +\
                                df[(df['selected_rows'])\
                                  ][col].str.replace(r"\bflat\b","", ).str.replace(r"\d+([a-z]|[A-Z])","", ).map(str)

    #merge_columns(df, "new_col", col, 'letter_after_number')
    df["new_col"] = merge_series(df[col], df['letter_after_number'])
    
    return df['new_col']

def replace_floor_flat(df, col):
    ''' 
    In references to basement, ground floor, first floor, second floor, and top floor
    flats, this function moves the word 'flat' to the front of the address. This is so that the
    following word (e.g. basement, ground floor) is recognised as the flat number in the 
    'extract_flat_and_other_no' function.
    '''

    # Helper function to apply replacements
    def apply_replacement(condition, replacement):
        return pl.when(condition).then(replacement).otherwise(pl.col(col))

    # Extracting series with the letter after number
    letter_after_number_series = extract_letter_one_number_address(df, col)

    # Replace values based on conditions
    df = df.with_columns(
        pl.col(col), apply_replacement(
            df[col].str.to_lowercase().str.contains(r"\bflat\b"),
            pl.col(col).str.replace(r"\bflat\b", "")
        )
    )

    # Define replacements for different floor types
    replacements = [
        ('basement', 'flat basement'),
        ('ground floor', 'flat a'),
        ('first floor|1st floor', 'flat b'),
        ('ground and first floor', 'flat ab'),
        ('basement ground and first floors', 'flat basementab'),
        ('second floor|2nd floor', 'flat c'),
        ('first and second floor', 'flat bc'),
        ('ground and first and second floor', 'flat abc'),
        ('third floor|3rd floor', 'flat d'),
        ('top floor', 'flat top')
    ]

    # Apply replacements
    for floor_type, replacement in replacements:
        df = df.with_columns(
            pl.col(col), apply_replacement(
                df[col].str.to_lowercase().str.contains(fr"\b{floor_type}\b"),
                pl.col(col).str.replace(fr"\b{floor_type}\b", replacement)
            )
        )

    # Merge the letter_after_number_series
    df = df.with_columns(
        "new_col", pl.when(letter_after_number_series.is_not_null()).then(letter_after_number_series).otherwise(pl.col(col))
    )

    return df["new_col"]

def remove_non_housing(df, col):
    '''
    Remove items from the housing list that are not housing. Includes addresses including
    the text 'parking', 'garage', 'store', 'visitor bay', 'visitors room', and 'bike rack',
    'yard', 'workshop'
    '''
    df_copy = df.copy()[~df[col].str.to_lowercase().str.contains(\
    r"parking|garage|\bstore\b|\bstores\b|\bvisitor bay\b|visitors room|\bbike rack\b|\byard\b|\bworkshop\b")]
                                                                 
    return df_copy

def extract_prop_no(df, col):
    '''
    Extract property number from an address. Remove flat/apartment/room numbers, 
    then extract the last number/number + letter in the string.
    '''
    try:
        prop_no = df.with_columns(pl.col(col).str.replace(r"(^\bapartment\b \w+)|(^\bapartments\b \w+)", "", \
                                  ).str.replace(r"(^\bflat\b \w+)|(^\bflats\b \w+)", "", \
                                               ).str.replace(r"(^\broom\b \w+)|(^\brooms\b \w+)", "", \
                                                            ).str.replace(",", "", \
                                                              ).str.extract(r"(\d+\w+|\d+)(?!.*\d+)")) #"(\d+\w+|\d+)(?!.*\d+)" 
    except:
        prop_no = np.nan
        
    return prop_no

def extract_room_no(df, col):
    '''
    Extract room number from an address. Find rows where the address contains 'room', then extract
    the next word after 'room' in the string.
    '''
    try:
        # Extract room numbers
        room_no = df.with_columns(
            pl.col(col)
            .str.to_lowercase()
            .str.contains(r"\broom\b|\brooms\b")
            .str.replace("no.", "")
            .str.extract(r'room. (\w+)')
        )
        room_no = room_no.rename([(0, "room_number")])  # Renaming extracted column
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error extracting room numbers: {e}")
        room_no = pl.DataFrame()  # Returning an empty DataFrame in case of error
    
    return room_no


def extract_flat_and_other_no(df, col):
    '''
    Extract flat number from an address. 
    It looks for letters after a property number IF THERE ARE NO MORE NUMBERS IN THE STRING,
    the words following the words 'flat' or 'apartment', or
    the last regex selects all characters in a word containing a digit if there are two numbers in the address
    '''
    
    # Regex patterns
    prop_number_pattern = r'^\d+([a-z]|[A-Z])(?!.*\d+)'
    flat_pattern = r'(?i)(?:flat|flats) (\w+)'
    apartment_pattern = r'(?i)(?:apartment|apartments) (\w+)'
    first_sec_number_pattern = r'(\d+.*?)[^a-zA-Z0-9_].*?\d+'
    first_letter_flat_number_pattern = r'\b([A-Za-z])\b[^\d]* \d'
    first_letter_no_more_numbers_pattern = r'^([a-z] |[A-Z] )(?!.*\d+)'
    
    # Extracting data
    replaced_series = (
        df.with_columns(
            pl.col(col)
            .str
            .to_lowercase()
            .str
            .replace(r"^\bflats\b", "flat")
            .str
            .contains(
                r"^\d+([a-z]|[A-Z])(?!.*\d+)|^([a-z] |[A-Z] )(?!.*\d+)|\bflat\b|\bapartment\b|(\d+.*?)[^a-zA-Z0-9_].*?\d+"
            )
        )[col]
        .str
        .replace("no.", "")
    )
    
    df = pl.DataFrame()

    # Extracting and assigning individual columns
    df["prop_number"] = replaced_series.str.extract(prop_number_pattern)
    
    extracted_series = replaced_series.str.extract(flat_pattern)
    df["flat_number"] = extracted_series[0].fillna(extracted_series[1]) if 1 in extracted_series.columns else extracted_series[0]
    
    extracted_series = replaced_series.str.extract(apartment_pattern)
    df["apart_number"] = extracted_series[0].fillna(extracted_series[1]) if 1 in extracted_series.columns else extracted_series[0]
    
    df["first_sec_number"] = replaced_series.str.extract(first_sec_number_pattern)
    df["first_letter_flat_number"] = replaced_series.str.extract(first_letter_flat_number_pattern)
    df["first_letter_no_more_numbers"] = replaced_series.str.extract(first_letter_no_more_numbers_pattern)
    
    return df

def extract_house_or_court_name(df, col):
    '''
    Extract house or court name. Extended to include estate, buildings, and mansions
    '''
    extracted_series = df.with_columns(pl.col(col).str.extract(r"(\w+)\s+(house|court|estate|buildings|mansions)"))
    if 1 in  extracted_series.columns:  
        df["house_court_name"] = extracted_series[0].fillna(extracted_series[1])
    else: 
        df["house_court_name"] = extracted_series[0]

    return df["house_court_name"]

def extract_block_and_unit_name(df, col):
    '''
    Extract house or court name. Extended to include estate, buildings, and mansions
    '''

    extracted_series = df.with_columns(pl.col(col).str.extract(r'(?i)(?:block|blocks) (\w+)'))
    if 1 in  extracted_series.columns:  
        df["block_number"] = extracted_series[0].fillna(extracted_series[1])
    else: 
        df["block_number"] = extracted_series[0]

    extracted_series = df.with_columns(pl.col(col).str.extract(r'(?i)(?:unit|units) (\w+)'))
    if 1 in  extracted_series.columns:  
        df["unit_number"] = extracted_series[0].fillna(extracted_series[1])
    else: 
        df["unit_number"] = extracted_series[0]

    return df

def replace_mistaken_dates(df, col:str):
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
    corrected_addresses = df.with_columns(pl.col(col).str.replace(pattern, replace_month))

    return corrected_addresses

def merge_series(full_series, partially_filled_series):
    '''
    Merge two series. The 'full_series' is the series you want to replace values in
    'partially_filled_series' is the replacer series.
    '''
    # Check if values in partially_filled_series are null
    replacer_series_is_null = partially_filled_series.is_null()

    # Start with full_series values
    merged_series = full_series.clone()

    # Replace values in merged_series where partially_filled_series is not null
    merged_series = pl.when(replacer_series_is_null)\
        .then(full_series)\
        .otherwise(partially_filled_series)

    return merged_series

def clean_cols(col:str) -> str:
    return col.lower().strip_chars().replace(r" ", "_").strip_chars()


def remove_flat_one_number_address(df, col):

    '''
    If there is only one number in the address, and there is no letter after the number,
    remove the word flat from the address
    '''

    df['contains_letter_after_number'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r"\d+(?:[a-z]|[A-Z])(?!.*\d+)"))
    df['contains_single_letter_before_number'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r'\b[A-Za-z]\b[^\d]* \d'))
    df['two_numbers_in_address'] =  df[col].str.to_lowercase().str.contains(r"(?:\d+.*?)[^a-zA-Z0-9_).*?\d+")
    df['contains_apartment'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r"\bapartment\b \w+|\bapartments\b \w+"))
    df['contains_flat'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r"\bflat\b \w+|\bflats\b \w+"))
    df['contains_room'] = df.with_columns(pl.col(col).str.to_lowercase().str.contains(r"\broom\b \w+|\brooms\b \w+"))
        

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
        
    df['one_number_no_flat'] =  df[df['selected_rows'] == True][col]
    df['one_number_no_flat'] =  df['one_number_no_flat'].str.replace(r"(\bapartment\b)|(\bapartments\b)", "", ).\
        str.replace(r"(\bflat\b)|(\bflats\b)", "", ).str.replace(r"(\broom\b)|(\brooms\b)", "", )


    #merge_columns(df, "new_col", col, 'one_number_no_flat')
    df["new_col"] = merge_series(df[col], df["one_number_no_flat"]) #merge_series(full_series: pd.Series, partially_filled_series: pd.Series)

    #print(df)

    return df['new_col']