"""
processing various outputs from workflow executions
"""

import csv
import os
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import subprocess

def validate_file_not_empty(file_path: Union[str, Path]) -> None:
    """
    Validates that a given file exists and is not empty. 
    Useful for failing fast at the beginning of parsing functions.

    Args:
        file_path (Union[str, Path]): The path to the file to check.

    Raises:
        FileNotFoundError: If the file does not exist or is not a regular file.
        ValueError: If the file exists but has a size of 0 bytes.
    """
    path = Path(file_path)
    
    # Check if the file actually exists
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: '{file_path}'")
    
    # Check if the file is empty (size == 0 bytes)
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: '{file_path}'")

def parse_mapping_stats(mapping_stats_file:Union[str, Path],
                        verbose=False,) -> pd.DataFrame:
    """
    Parse mapping statistics from a mapping_stats output file.
    
    Parameters
    ----------
    mapping_stats_file : Union[str, Path]
        Path to input mapping_stats file.
    Returns
    -------
    mapping_stats_df : pd.DataFrame
        DataFrame containing parsed mapping statistics.
    Notes
    -----
    """ 
    validate_file_not_empty(mapping_stats_file)
    if verbose:
        print(f"Parsing mapping statistics from file: {mapping_stats_file}\n")
    tmp = pd.read_csv(mapping_stats_file, delimiter="\t", index_col=None, header=None)
    df_chunk_source, cur_col_name,value = [], "", None
    for elem in tmp[0].values:
        if elem.startswith(">"):
            # create a df from the previous chunk
            cur_col_name = elem[1:]
        else:
            if cur_col_name!="":
                value = elem
        if value is not None and cur_col_name!="":
            df_chunk_source.append([cur_col_name,value])
            value,cur_col_name = None,""
    if len(df_chunk_source) > 0:
        mapping_stats_df = pd.DataFrame(df_chunk_source,columns=['col','value'])
        mapping_stats_df.index = mapping_stats_df['col']
        mapping_stats_df = mapping_stats_df.drop(['col'],axis=1)
        mapping_stats_df = mapping_stats_df.transpose()
        mapping_stats_df = mapping_stats_df.reset_index(drop=True)
        mapping_stats_df.columns.name = ""
        if verbose:
            print(f"Successfully parsed {len(mapping_stats_df.columns)} mapping statistics for {mapping_stats_file}\n")
    else:
        if verbose:
            print(f"no content found in mapping stats file {mapping_stats_file}\n")
        mapping_stats_df = None
    return mapping_stats_df