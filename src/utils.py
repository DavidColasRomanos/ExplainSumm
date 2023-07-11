"""
Utils.

"""

from typing import Tuple

import numpy as np
import pandas as pd

SEED = 108


def reduceMemory(df):
    """
    Reduces de memory used by the input DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame.

    Returns
    -------
    df : pd.DataFrame
        Reduced memory DataFrame.

    """

    print("Reducing dataset memory...")
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    print(
        "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
            end_mem, 100 * (start_mem - end_mem) / start_mem
        )
    )

    return df


def train_test_split_comparision(
    df: pd.DataFrame, train_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into a training set and a testing set based on the text 
    column. All records with the same text will be put into the same set.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split.
    train_size : float
        Proportion of the data to include in the train split 
        (0 < train_size < 1).

    Returns
    -------
    train_df : pd.DataFrame
        Training set DataFrame.
    test_df : pd.DataFrame
        Testing set DataFrame.
    """

    # Ensure valid train_size
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")

    # Create a list with unique texts
    texts = df["text"].unique()

    # Shuffle the list of unique texts
    np.random.seed(SEED)
    np.random.shuffle(texts)

    # Define the index to split
    split_index = int(len(texts) * train_size)

    # Split the texts into train and test
    train_texts = texts[:split_index]
    test_texts = texts[split_index:]

    # Use the split texts to split the dataframe
    train_df = df[df["text"].isin(train_texts)].reset_index(drop=True)
    test_df = df[df["text"].isin(test_texts)].reset_index(drop=True)

    return train_df, test_df
