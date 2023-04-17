"""
Module for pre-processing.

"""

import numpy as np
import pandas as pd


class Preprocessor:
    """
    Preprocessing class.
    
    """

    def __init__(self):
        """
        Initialize the Preprocessor class.
        
        """
        pass

    def preprocessing_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-processing pipeline.

        """
        df = data.copy()

        # Transform obj to bool columns
        df = self.object_to_bool(df)

        # Regex transformations
        df = self.regex_replacements(df)

        # Strip string columns
        df = self.strip_str_columns(df)

        # Replace empty cells and drop NaNs
        df = self.handle_missing_values(df)

        # Reset index
        df.reset_index(drop=True, inplace=True)

        return df

    def strip_str_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Strip string columns.

        """
        df = data.copy()

        # Strip object columns
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda x: x.str.strip())

        return df

    def object_to_bool(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform object columns to boolean columns.

        """
        df = data.copy()

        # Convert object columns with all boolean values to boolean type
        obj_cols = df.select_dtypes(include="object").columns

        for col in obj_cols:
            unique_vals = df[col].unique()

            if any(val in [True, False] for val in unique_vals):
                df[col] = df[col].astype(bool)

        return df

    def regex_replacements(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Replace [numbers] in string columns with "".

        """

        def remove_pattern(s):
            """
            Function to replace pattern with empty string.

            """
            return s.str.replace(r"\[\d+\]", "", regex=True)

        df = data.copy()

        # Replace [n] with "" in object columns
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(remove_pattern)

        return df

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill None values with NaN, replace empty values by np.nan, and drop rows 
        with no value in key columns.
        
        """
        df = data.copy()

        # Fill None and replace empty values
        df.fillna(value=np.nan, inplace=True)
        df.replace("", np.nan, inplace=True)

        # Select key columns
        cols_to_filter = df.columns[df.columns.str.contains("title|text|summary")]

        # Drop rows with NaN values in key columns
        df = df.dropna(subset=cols_to_filter)

        return df

