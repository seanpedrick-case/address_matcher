import os
import pandas as pd

# +
''' Fuzzywuzzy/Rapidfuzz scorer to use. Options are: ratio, partial_ratio, token_sort_ratio, partial_token_sort_ratio,
token_set_ratio, partial_token_set_ratio, QRatio, UQRatio, WRatio (default), UWRatio
details here: https://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings'''

fuzzy_scorer_used = "token_set_ratio"

# +
fuzzy_match_limit = 45

fuzzy_search_addr_limit = 20

filter_to_lambeth_pcodes= False 
# -

standardise = False

# +
if standardise == True:
    std = "_std"
if standardise == False:
    std = "_not_std"
    
dataset_name = "ctax" + std
    
suffix_used = dataset_name + "_" + fuzzy_scorer_used
file_name = "search"

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Neural net variables and models

# +
# https://stackoverflow.com/questions/59221557/tensorflow-v2-replacement-for-tf-contrib-predictor-from-saved-model

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
print(ROOT_DIR)

#model_stub = "addr_model_out_lon_keras"
model_stub = "addr_model_out_lon"
#model_stub = "two_epoch"
#model_stub = "two_epoch_128"
#model_stub = "two_epoch_64_perf_3"
#model_stub = "tenK"

# I add a suffix to output files to be able to separate comparisons of test data from the same model with different steps
# e.g. '350' indicates a model that has been through 350,000 steps of training

#file_step_suffix = "2"
#file_step_suffix = "3"
file_step_suffix = "550"

model_dir_name = os.path.join(ROOT_DIR, "nnet_model" , model_stub , "00000001")
print(model_dir_name)

model_path = os.path.join(model_dir_name, "saved_model.zip")
print("model path: ")
print(model_path)

if os.path.exists(model_path):

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Better to go without GPU to avoid 'out of memory' issues

    import tensorflow as tf
    import recordlinkage

    tf.config.list_physical_devices('GPU')

    ## The labels_list object defines the structure of the prediction outputs. It must be the same as what the model was originally trained on
    
    ''' Load pre-trained model '''
    
    import zipfile

    with zipfile.ZipFile(model_path,"r") as zip_ref:
        zip_ref.extractall(model_dir_name)
        
    if model_stub == "addr_model_out_lon":
        exported_model = tf.saved_model.load(model_dir_name)

    else:
        exported_model = tf.keras.models.load_model(model_dir_name, compile=False)
        # Compile the model with a loss function and an optimizer
        exported_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])

else: exported_model = []
# +
''' Fuzzy method '''

''' https://recordlinkage.readthedocs.io/en/latest/ref-compare.html#recordlinkage.compare.String
 The Python Record Linkage Toolkit uses the jellyfish package for the Jaro, Jaro-Winkler, Levenshtein and Damerau- Levenshtein algorithms.
 Options are [‘jaro’, ‘jarowinkler’, ‘levenshtein’, ‘damerau_levenshtein’, ‘qgram’, ‘cosine’, ‘smith_waterman’, ‘lcs’]

 Comparison of some of the Jellyfish string comparison methods: https://manpages.debian.org/testing/python-jellyfish-doc/jellyfish.3.en.html '''


fuzzy_method = "jarowinkler"

# +
''' Required overall match score for all columns to count as a match '''

#score_cut_off = 0.9
score_cut_off = 0.987
#score_cut_off = 0.975

# -

ref_address_cols = ["SaoStartNumber", "SaoStartSuffix", "SaoEndNumber", "SaoEndSuffix",
       "SaoText", "PaoStartNumber", "PaoStartSuffix", "PaoEndNumber",
       "PaoEndSuffix", "PaoText", "Street", "PostTown", "Postcode"]

# +
# Create a list of matching variables. 

matching_variables = ref_address_cols
text_columns = ["PaoText", "Street", "PostTown", "Postcode"]
# -

# ### Modify relative importance of columns (weights)

# +
''' Modify weighting for scores - Town and AdministrativeArea are not very important as we have postcode. Street number and name are important '''

PaoStartNumber_weight = 2
SaoStartNumber_weight = 2
Street_weight = 3
PostTown_weight = 0
Postcode_weight = 1.25
AdministrativeArea_weight = 0
# -

weight_vals = [1] * len(ref_address_cols)
weight_keys = ref_address_cols
weights = {weight_keys[i]: weight_vals[i] for i in range(len(weight_keys))} 

# +
# Modify weighting for scores - Town and AdministrativeArea are not very important as we have postcode. Street number and name are important

weights["SaoStartNumber"] = SaoStartNumber_weight
weights["PaoStartNumber"] = PaoStartNumber_weight
weights["Street"] = Street_weight
weights["PostTown"] = PostTown_weight
weights["Postcode"] = Postcode_weight


# -

class MatcherClass:
    def __init__(self, 
                fuzzy_scorer_used, fuzzy_match_limit, fuzzy_search_addr_limit, filter_to_lambeth_pcodes, standardise, suffix_used,
                matching_variables, model_dir_name, file_step_suffix, exported_model, fuzzy_method, score_cut_off, text_columns, weights, file_name
                ):
        
        # Fuzzy/general attributes
        self.fuzzy_scorer_used = fuzzy_scorer_used
        self.fuzzy_match_limit = fuzzy_match_limit
        self.fuzzy_search_addr_limit = fuzzy_search_addr_limit
        self.filter_to_lambeth_pcodes = filter_to_lambeth_pcodes
        self.standardise = standardise
        self.suffix_used = suffix_used
        self.file_name = file_name
        
        # Neural net attributes
        self.matching_variables = matching_variables
        self.model_dir_name = model_dir_name
        self.file_step_suffix = file_step_suffix

        if exported_model:
            self.exported_model = exported_model
        else: self.exported_model = []

        self.fuzzy_method = fuzzy_method
        self.score_cut_off = score_cut_off
        self.text_columns = text_columns
        self.weights = weights     

        # These are variables that are added on later
        
        # Join data
        self.search_df = pd.DataFrame()
        self.search_address_cols = []
        self.search_postcode_col = []
        self.search_df_key_field = []
        self.ref = pd.DataFrame()
        self.ref_address_cols = []
        self.new_join_col = []
        self.in_joincol_list = []
        self.standard_llpg_format = []

        
        # Results attributes
        self.match_results = pd.DataFrame()
        self.predict_df_nnet = pd.DataFrame()
        
        # Other attributes generated during training
        self.compare_all_candidates = []
        self.diag_shortlist = []
        self.diag_best_match = []
        self.diag_best_match_matches = []
        self.diag_best_match_not_matched = []
        self.match_results_output = []
        self.results_on_orig_df = []
        self.summary = []
        self.search_address_cols = []
        self.output_summary = []
        self.search_df_not_matched = []
        
        self.match_outputs_name = []
        self.results_orig_df_name = []
    
InitMatch = MatcherClass(
                fuzzy_scorer_used, fuzzy_match_limit, fuzzy_search_addr_limit, filter_to_lambeth_pcodes, standardise, suffix_used,
                matching_variables, model_dir_name, file_step_suffix, exported_model, fuzzy_method, score_cut_off, text_columns, weights, file_name
                )