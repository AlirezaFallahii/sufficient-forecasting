from ds_cleaning_funcs import *

#%% Choosing the dataset
dataset_type_list = ['from_Stable_Dist.', 'from_stock&watson.xls']
dataset_type = dataset_type_list[0]

# Loading the dataset (only for dataset_type_list[1])
if dataset_type == dataset_type_list[1]:
    X_input, y_input = prepare_dataset_from_xls(file_name='stock&watson.xls',
                                                sheet_name='Sheet1',
                                                numEmptyRows=9)

#%% Setting hyper-parameters for training
numOfRep = 1000  # can be changed to 1000
if dataset_type == dataset_type_list[1]:
    alpha_values_list = [-1]  # should be [-1]
else:
    alpha_values_list = [1.9]  # can be changed to [1.5, 1.6, 1.7, 1.8, 1.9]

pca_features_min = 20
pca_features_max = 20

#%% Training on the chosen dataset
exec(open('KPCA_full_feature_regression_kernel.py').read())
