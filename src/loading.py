"""
Module for loading data from Hugging Face dataset.

"""

import pandas as pd
import numpy as np


def huggingface_dataset_to_dataframes(
    dataset_dict, keys: list, dict_cols: list, list_col: str = None
):
    """
    Converts a Hugging Face dataset in `DatasetDict` format to two Pandas DataFrames,
    one for the train dataset and one for the validation dataset. If any column in the
    dataset contains dictionaries, the function will convert them to new columns in the DataFrame.
    
    Parameters:
    dataset_dict : DatasetDict
        A Hugging Face 'DatasetDict' object containing the train and validation datasets.
    dict_cols : list 
        A list of the names of the columns that contain dictionaries.
    list_col : str
        Colunm name containing list of dicts.
    
    Returns:
    A tuple of two Pandas DataFrames, one for the train dataset and one for the validation dataset.
    
    """

    # Extract the train and validation datasets
    df0 = dataset_dict[keys[0]]
    df1 = dataset_dict[keys[1]]

    # Convert the train and validation dataset to a Pandas DataFrame
    df0 = pd.DataFrame(df0)
    df1 = pd.DataFrame(df1)

    # Convert columns containing dictionaries to new columns
    for col in dict_cols:

        # Train
        df0_cols = pd.DataFrame(df0[col].tolist(), index=df0.index)
        df0 = pd.concat([df0, df0_cols], axis=1)
        df0.drop(columns=[col], inplace=True)

        # Test
        df1_cols = pd.DataFrame(df1[col].tolist(), index=df1.index)
        df1 = pd.concat([df1, df1_cols], axis=1)
        df1.drop(columns=[col], inplace=True)

    if list_col:

        # Conver columns containing lists to to new colums
        df0 = pd.concat(
            [pd.DataFrame(np.stack(df0[list_col])), df0.drop(columns=[list_col])],
            axis=1,
        )
        df1 = pd.concat(
            [pd.DataFrame(np.stack(df1[list_col])), df1.drop(columns=[list_col])],
            axis=1,
        )

        # Transform columns from column list
        for n in [0, 1]:

            df0 = pd.concat(
                [df0.drop(columns=[n]), expand_dictionary_column(df0, n, f"_{n}"),],
                axis=1,
            )

            df1 = pd.concat(
                [df1.drop(columns=[n]), expand_dictionary_column(df1, n, f"_{n}"),],
                axis=1,
            )

    # Drop duplicates
    df0.drop_duplicates(inplace=True)
    df1.drop_duplicates(inplace=True)

    # Reset index
    df0.reset_index(drop=True, inplace=True)
    df1.reset_index(drop=True, inplace=True)

    # Replace column names
    df0.columns = df0.columns.str.replace("text", "summary")
    df1.columns = df1.columns.str.replace("text", "summary")

    # Simplify site, subreddit, article, and post columns
    df0 = _simplify_columns(df0)
    df1 = _simplify_columns(df1)

    # Return the two dataframes
    return df0, df1


def _extract_value(d, key):
    """
    Iterate over the rows and add the values to the new columns.
    
    """

    if key in d:
        return d[key]
    else:
        return None


def expand_dictionary_column(data, col_name, col_suffix):
    """
    Expand a column of dictionaries into separate columns for each key in the dictionary.
    
    Parameters:
    - df: The Pandas DataFrame to operate on.
    - col_name: The name of the column containing the dictionaries.
    - col_suffix: The suffix to use for the new column names.
    
    Returns:
    A new Pandas DataFrame with separate columns for each key in the dictionaries.
    
    """

    df = data.copy()

    # Create a set of all keys in the dictionaries
    keys = set()
    df[col_name].apply(lambda d: keys.update(d.keys()))

    # Create a dictionary to store the new columns
    new_cols = {}
    for key in keys:
        new_key = str(key) + col_suffix
        new_cols[new_key] = []

    for key in keys:
        new_key = str(key) + col_suffix
        df[new_key] = df[col_name].apply(lambda d: _extract_value(d, key))

    return df[list(new_cols.keys())]


def _simplify_columns(data):
    """
    Simplify site, subreddit, article, and post columns.
    
    """

    df = data.copy()

    # Create
    df[["source", "subsource"]] = df.apply(
        lambda row: ("cnn_dailymail", row["site"])
        if row["site"]
        else ("reddit", row["subreddit"])
        if row["subreddit"]
        else None,
        axis=1,
    ).apply(pd.Series)
    df["text"] = df.apply(
        lambda row: row["article"]
        if row["article"]
        else row["post"]
        if row["post"]
        else None,
        axis=1,
    )

    # Drop
    df.drop(["site", "subreddit", "article", "post"], axis=1, inplace=True)

    return df
