import pandas as pd


def assess_NA(data):
    """
    Returns a pandas dataframe denoting the total number of NA values
    and the percentage of NA values in each column.
    The column names are noted on the index.

    Parameters
    ----------
    data: dataframe
    """
    # pandas series denoting features and the sum of their null values
    null_sum = data.isnull().sum()  # instantiate columns for missing data
    total = null_sum.sort_values(ascending=False)
    percent = (((null_sum / len(data.index)) * 100).
               round(2)).sort_values(ascending=False)

    # concatenate along the columns to create the complete dataframe
    df_NA = pd.concat([total, percent], axis=1,
                      keys=['Number of NA', 'Percent NA'])

    # drop rows that don't have any missing data;
    # omit if you want to keep all rows
    df_NA = df_NA[(df_NA.T != 0).any()]

    return df_NA


def prepare_dataset_from_xls(file_name='stock&watson.xls',
                             sheet_name='Sheet1',
                             numEmptyRows=9):

    # numEmptyRows = 9  # Number of empty rows in the end of sheet one.
    # # This number corresponds to the dates from "4/1/2009" to "12/1/2009".

    # numEmptyRows = 3  # Number of empty rows in the end of sheet two.
    # # This number corresponds to the dates from "5/1/2009" to "11/1/2009".


    raw_df1 = pd.read_excel(file_name,
                            sheet_name=sheet_name,
                            index_col=0,
                            skipfooter=numEmptyRows)
    print(f'\nreading "{file_name}" file is finished. Note: '
          f'Only sheet "{sheet_name}" is read into the memory.')
    print(f'The "{sheet_name}" sheet has {raw_df1.shape[0]} '
          f'rows and {raw_df1.shape[1]} columns.')
    print('**************************************************************')
    # Performing alg. on dataset
    # Part 0: Visualizing a small portion of the dataset
    print('\nWe want to first Visualize a small portion of the dataset ...')
    print('First 10 rows of the dataset:')
    print(raw_df1.head(10))
    print('Last 10 rows of the dataset:')
    print(raw_df1.tail(10))
    print('summary of the dataset:')
    print(raw_df1.describe())
    print('the first 5*5 elements of the raw matrix of the dataset:')
    print(raw_df1.iloc[0:5, 0:5].values)
    print('the last 5*5 elements of the raw matrix of the dataset:')
    print(raw_df1.iloc[-5:, -5:].values)
    print('**************************************************************')
    print('\nWe want to clean dataset now (Removing metadata and'
          'Imputing missing values with "mean") ...\n')
    # Performing alg. on dataset
    # Part 1: Cleaning Dataset
    # Part 1.1: Removing metadata (rows in the beginning of the dataset)
    df1 = raw_df1.drop(raw_df1.index.values[:9])
    print('the first 5*5 elements of the new matrix of '
          'the dataset: (after removing metadata)')
    print(df1.iloc[0:5, 0:5].values)

    # Part 1.2: Imputing missing values with "mean"
    print('\nsummary statistics of present NA values'
          ' (before removing NA values):')
    assess_NA_df = assess_NA(df1)
    print(assess_NA_df)
    # Impute with mean on NA values
    for col_name in assess_NA_df.index.values:
        df1[col_name] = df1[col_name].fillna(df1[col_name].mean())
    print('\nsummary statistics of NA values: (after removing them)')
    print('[this code line is written for validating our imputation alg.]')
    assess_NA_df = assess_NA(df1)
    print(assess_NA_df)
    if len(assess_NA_df.index.values) == 0:
        print('As you can see, all of the NA values are removed.')

    print('**************************************************************')
    print('\nWe want to create "X_input" and "y_input" ...')
    # Performing alg. on dataset
    # Part 2: creating "X_input" and "y_input"
    print('Choose the column that has to become "dependent variable":')
    print('list of columns are:\n')
    print(df1.columns.values)
    column_str = input('\nEnter the column name acronym and end it with spaces'
                       ' until the string length happens to be 8 characters.')

    y_input = df1[[column_str]];
    X_input = df1.drop(columns=[column_str])

    X_input = X_input.values
    y_input = y_input.values

    print('**************************************************************')
    print('\nDataset is now thoroughly prepared for training and testing.\n')
    return X_input, y_input
