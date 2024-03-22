# import os
# import sys
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
# import time
# import random

# import glob
# from sglm.features import sglm_pp
# import freely_moving_helpers as lpp
from tqdm import tqdm, trange
from pathlib import Path

def get_rename_columns_by_file(files_list, channel_definitions, verbose=0):
    """
    Rename channels in a file.
    Args:
        files_list : list(str)
            List of filenames with which to associate different names for channels
        channel_definitions : dict(dict)
            Dictionary of Keys: tuple of filename identifiers (all must be matched) to Vals: dictionary mapping initial channel names to final channel names
        verbose : int
            Verbosity level
    Returns:
        channel_assignments : dict
            Dictionary of Keys: filenames with required renamings to Vals: 

    Example:
        files_list = glob.glob(f'{dir_path}/../GLM_SIGNALS_WT61_*') + \
                    glob.glob(f'{dir_path}/../GLM_SIGNALS_WT63_*') + \
                    glob.glob(f'{dir_path}/../GLM_SIGNALS_WT64_*')
        channel_definitions = {
            ('WT61',): {'Ch1': 'gACH', 'Ch2': 'rDA'},
            ('WT64',): {'Ch1': 'gACH', 'Ch2': 'empty'},
            ('WT63',): {'Ch1': 'gDA', 'Ch2': 'empty'},
        }
        channel_assignments = rename_channels(files_list, channel_definitions)

        (channel_assignments will map each individual filename to a renaming dictionary of columns)
    """
    channel_assignments = {}
    for file_lookup in channel_definitions:
        print(file_lookup)
        channel_renamings = channel_definitions[file_lookup]
        relevant_files = [f for f in files_list if all(x in f for x in file_lookup)]
        for relevant_file in relevant_files:
            relevant_file = Path(relevant_file).parts[-1]
            print('>', relevant_file)
            channel_assignments[relevant_file] = channel_renamings
    
    return channel_assignments

def rename_consistent_columns(df, rename_columns={'Ch1':'Ch1',
                                                  'Ch2':'Ch2',
                                                  'Ch5':'Ch5',
                                                  'Ch6':'Ch6',

                                                  'centerOcc':'cpo',
                                                  'centerIn':'cpn',
                                                  'centerOut':'cpx',
                                                  'rightOcc':'rpo',
                                                  'rightIn':'rpn',
                                                  'rightOut':'rpx',
                                                  'rightLick':'rl',
                                                  'leftOcc':'lpo',
                                                  'leftIn':'lpn',
                                                  'leftOut':'lpx',
                                                  'leftLick':'ll',
                                                  'reward':'r',
                                                  'noreward':'nr',
                                                  'right':'Rt',
                                                  'left':'Lt',
                                                  'nTrial': 'TrialNumber',
                                                 }
                             ):
    '''
    Simplify variable names to match the GLM

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe in which to rename columns
    rename_columns : dict
        Dictionary of old column names to rename to new column names

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with renamed columns
    '''
    # Simplify variable names
    df = df.rename(rename_columns, axis=1)
    return df



def set_port_entry_exit_rewarded_unrewarded_indicators(df):
    '''
    Set port entry, exit, and intersecting reward / non-reward indicators
    Args:
        df: dataframe with right / left port entry / exit columns and reward/no_reward indicators
    Returns:
        dataframe with right / left, rewarded / unrewarded intersection indicators
    '''
    # Identify combined reward vs. non-rewarded / left vs. right / entries vs. exits
    df = df.assign(**{
        'rpxr':df['r']*df['rpx'],
        'rpxnr':df['nr']*df['rpx'],
        'lpxr':df['r']*df['lpx'],
        'lpxnr':df['nr']*df['lpx'],

        'rpnr':df['r']*df['rpn'],
        'rpnnr':df['nr']*df['rpn'],
        'lpnr':df['r']*df['lpn'],
        'lpnnr':df['nr']*df['lpn'],

    })
    return df


def define_side_agnostic_events(df, agnostic_definitions={'spn':['rpn','lpn'],
                                                          'spx':['rpx','lpx'],
                                                          'spnr':['rpnr','lpnr'],
                                                          'spnnr':['rpnnr','lpnnr'],
                                                          'spxr':['rpxr','lpxr'],
                                                          'spxnr':['rpxnr','lpxnr'],
                                                          'sl':['rl','ll'],
                                                         }):
    '''
    Define side agnostic events
    Args:
        df: dataframe with left / right entry / exit and rewarded / unrewarded indicators
    Returns:
        dataframe with added port entry/exit, and reward indicators
    '''
    
    dct = {}
    for key in agnostic_definitions:
#         print(agnostic_definitions[key])
        dct[key] = df[agnostic_definitions[key]].sum(axis=1)
    df = df.assign(**dct)

    return df



def add_timeshifts_to_col_list(all_cols, shifted_cols, neg_order=0, pos_order=1, shift_spacer='_'):
    """
    Add a number of timeshifts to the shifted_cols name list provided for every column used. 

    JZ 2021
    
    Args:
        all_cols : list(str)
            All column names prior to the addition of shifted column names
        shifted_cols : list(str)
            The list of columns that have been timeshifted
        neg_order : int
            Negative order i.e. number of shifts performed backwards (should be in range -inf to 0 (incl.))
        pos_order : int
            Positive order i.e. number of shifts performed forwards (should be in range 0 (incl.) to inf)
    
    Returns: List of all column names remaining after shifts in question
    """ 
    out_col_list = []
    for shift_amt in list(range(neg_order, 0))+list(range(1, pos_order + 1)):
        out_col_list.extend([_ + f'{shift_spacer}{shift_amt}' for _ in shifted_cols])
    return all_cols + out_col_list


def col_shift_bounds_dict_to_col_list(X_cols_basis, X_cols_sftd, shift_spacer='_'):
        X_cols_sftd_basis = []
        for X_col_single in X_cols_basis:
            col_bounds = X_cols_basis[X_col_single]
            if col_bounds == (0,0):
                cols = [_ for _ in X_cols_sftd if X_col_single+shift_spacer in _ or X_col_single == _]
                X_cols_sftd_basis += cols
            else:
                cols = add_timeshifts_to_col_list([X_col_single], [X_col_single], neg_order=col_bounds[0], pos_order=col_bounds[1])
                X_cols_sftd_basis += [_ for _ in X_cols_sftd if _ in cols]
        return X_cols_sftd_basis


def generate_toy_data():
    pass