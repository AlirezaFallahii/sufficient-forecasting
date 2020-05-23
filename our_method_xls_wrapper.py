from ds_cleaning_funcs import *

#%%
# Loading dataset
X_input, y_input = prepare_dataset_from_xls(file_name='stock&watson.xls',
                                            sheet_name='Sheet1',
                                            numEmptyRows=9)

#%%
# Performing alg. on dataset
# Part 3: Running the learning algorithm (in a separate script)

numOfRep = 1 # can be changed to 1000
max_sir_features = 3
min_sir_features = 3  # from 3
pca_features_min = 20  # from 3
pca_features_max = 20

exec(open('our_method_xls_kernel.py').read())

#%%
