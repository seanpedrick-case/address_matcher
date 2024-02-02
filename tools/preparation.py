import pandas as pd
from typing import TypeVar, Dict, List, Tuple
from datetime import datetime

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')
PandasSeries = TypeVar('pd.core.frame.Series')
MatchedResults = Dict[str,Tuple[str,int]]
array = List[str]

today = datetime.now().strftime("%d%m%Y")
today_rev = datetime.now().strftime("%Y%m%d")

from tools.standardise import extract_postcode, remove_postcode, extract_street_name

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

def create_full_address(df):

    df = df.fillna("")

    if "Organisation" not in df.columns:
        df["Organisation"] = ""

    df["full_address"] = df['Organisation'] + " " + df['SaoText'].str.replace(" - ", " REPL ").str.replace("- ", " REPLEFT ").str.replace(" -", " REPLRIGHT ") + " " + df["SaoStartNumber"].astype(str) + df["SaoStartSuffix"] + "-" + df["SaoEndNumber"].astype(str) + df["SaoEndSuffix"] + " " + df["PaoText"].str.replace(" - ", " REPL ").str.replace("- ", " REPLEFT ").str.replace(" -", " REPLRIGHT ") + " " + df["PaoStartNumber"].astype(str) + df["PaoStartSuffix"] + "-" + df["PaoEndNumber"].astype(str) + df["PaoEndSuffix"] + " " + df["Street"] + " " + df["PostTown"] + " " + df["Postcode"]

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

    #print(ref_address_cols_uprn_w_ref)
    
    ref_df = ref.copy()

    # Drop duplicates in the key field - not necessary?
    #ref_df = ref_df.drop_duplicates(new_join_col)

    #print(ref_df)

      
    # In on-prem LPI db street has been excluded, so put this back in
    if ('Street' not in ref_df.columns) & ('Address_LPI' in ref_df.columns): 
            ref_df['Street'] = ref_df['Address_LPI'].str.replace("\\n", " ", regex = True).apply(extract_street_name)#
        
    if ('Organisation' not in ref_df.columns) & ('SaoText' in ref_df.columns):
        ref_df['Organisation'] = ""
     
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
