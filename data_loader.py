"""
Data loading utility functions.
"""
import re

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

EPSILON = 1e-6


def construct_tcell_df(df: pd.DataFrame, feature_type: str = '', tcell_types: list = None, proteins: list = None,
                       samplings: list = None, scale: bool = True, norm_type: str = 'subtract') -> (np.ndarray,
                                                                                                    np.ndarray):
    """
    Converts given pandas dataframe with raw data into a numpy array with T-cell counts and/or percentages.

    :param df: pandas dataframe with raw data.
    :param feature_type: T-cell data type to be included ['count', 'perc', ''], where '' stands for both.
    :param tcell_types: list of strings specifying T-cell types to be included ['CD3', 'CD4', 'CD8'].
    :param proteins: list of integers specifying treatments to be included [2, 3, 4, 5, 6].
    :param samplings: list of strings specifying screening time points to be included ['S10', 'S11'].
    :param scale: flag identifying whether to standardise data.
    :param norm_type: string specifying normalisation type -- 'subtract' or 'ratio'.
    :return: returns (pre-processed) numpy array with specified data types, cell types, and treatments and an array
    with feature names.
    """
    cols = list(df.columns)

    if tcell_types is None:
        tcell_types = ['CD3', 'CD4', 'CD8']

    if proteins is None:
        proteins = [2, 3, 4, 5, 6]

    if samplings is None:
        samplings = ['S10', 'S11']

    # Background
    CD3_cols_bg = [s for s in cols if (re.search('CD3', s) and (re.search(feature_type + '_1_' + samplings[0], s) or
                                                                re.search(feature_type + '_1_' + samplings[1], s)))]
    CD4_cols_bg = [s for s in cols if (re.search('CD4', s) and (re.search(feature_type + '_1_' + samplings[0], s) or
                                                                re.search(feature_type + '_1_' + samplings[1], s)))]
    CD8_cols_bg = [s for s in cols if (re.search('CD8', s) and (re.search(feature_type + '_1_' + samplings[0], s) or
                                                                re.search(feature_type + '_1_' + samplings[1], s)))]

    # CoV-2 mix
    CD3_cols = [s for s in cols if (re.search('CD3', s) and (re.search(feature_type + '_2_' + samplings[0], s) or
                                                             re.search(feature_type + '_2_' + samplings[1], s)))]
    CD4_cols = [s for s in cols if (re.search('CD4', s) and (re.search(feature_type + '_2_' + samplings[0], s) or
                                                             re.search(feature_type + '_2_' + samplings[1], s)))]
    CD8_cols = [s for s in cols if (re.search('CD8', s) and (re.search(feature_type + '_2_' + samplings[0], s) or
                                                             re.search(feature_type + '_2_' + samplings[1], s)))]

    # Prot N
    CD3_cols_n = [s for s in cols if (re.search('CD3', s) and (re.search(feature_type + '_3_' + samplings[0], s) or
                                                               re.search(feature_type + '_3_' + samplings[1], s)))]
    CD4_cols_n = [s for s in cols if (re.search('CD4', s) and (re.search(feature_type + '_3_' + samplings[0], s) or
                                                               re.search(feature_type + '_3_' + samplings[1], s)))]
    CD8_cols_n = [s for s in cols if (re.search('CD8', s) and (re.search(feature_type + '_3_' + samplings[0], s) or
                                                               re.search(feature_type + '_3_' + samplings[1], s)))]

    # Prot S1
    CD3_cols_s1 = [s for s in cols if (re.search('CD3', s) and (re.search(feature_type + '_4_' + samplings[0], s) or
                                                                re.search(feature_type + '_4_' + samplings[1], s)))]
    CD4_cols_s1 = [s for s in cols if (re.search('CD4', s) and (re.search(feature_type + '_4_' + samplings[0], s) or
                                                                re.search(feature_type + '_4_' + samplings[1], s)))]
    CD8_cols_s1 = [s for s in cols if (re.search('CD8', s) and (re.search(feature_type + '_4_' + samplings[0], s) or
                                                                re.search(feature_type + '_4_' + samplings[1], s)))]

    # Prot S
    CD3_cols_s = [s for s in cols if (re.search('CD3', s) and (re.search(feature_type + '_5_' + samplings[0], s) or
                                                               re.search(feature_type + '_5_' + samplings[1], s)))]
    CD4_cols_s = [s for s in cols if (re.search('CD4', s) and (re.search(feature_type + '_5_' + samplings[0], s) or
                                                               re.search(feature_type + '_5_' + samplings[1], s)))]
    CD8_cols_s = [s for s in cols if (re.search('CD8', s) and (re.search(feature_type + '_5_' + samplings[0], s) or
                                                               re.search(feature_type + '_5_' + samplings[1], s)))]

    # Prot M
    CD3_cols_m = [s for s in cols if (re.search('CD3', s) and (re.search(feature_type + '_6_' + samplings[0], s) or
                                                               re.search(feature_type + '_6_' + samplings[1], s)))]
    CD4_cols_m = [s for s in cols if (re.search('CD4', s) and (re.search(feature_type + '_6_' + samplings[0], s) or
                                                               re.search(feature_type + '_6_' + samplings[1], s)))]
    CD8_cols_m = [s for s in cols if (re.search('CD8', s) and (re.search(feature_type + '_6_' + samplings[0], s) or
                                                               re.search(feature_type + '_6_' + samplings[1], s)))]

    if not ('CD3' in tcell_types):
        CD3_cols_bg = []
        CD3_cols = []
        CD3_cols_n = []
        CD3_cols_s1 = []
        CD3_cols_s = []
        CD3_cols_m = []
    if not ('CD4' in tcell_types):
        CD4_cols_bg = []
        CD4_cols = []
        CD4_cols_n = []
        CD4_cols_s1 = []
        CD4_cols_s = []
        CD4_cols_m = []
    if not ('CD8' in tcell_types):
        CD8_cols_bg = []
        CD8_cols = []
        CD8_cols_n = []
        CD8_cols_s1 = []
        CD8_cols_s = []
        CD8_cols_m = []

    if not (2 in proteins):
        CD3_cols = []
        CD4_cols = []
        CD8_cols = []
    if not (3 in proteins):
        CD3_cols_n = []
        CD4_cols_n = []
        CD8_cols_n = []
    if not (4 in proteins):
        CD3_cols_s1 = []
        CD4_cols_s1 = []
        CD8_cols_s1 = []
    if not (5 in proteins):
        CD3_cols_s = []
        CD4_cols_s = []
        CD8_cols_s = []
    if not (6 in proteins):
        CD3_cols_m = []
        CD4_cols_m = []
        CD8_cols_m = []

    bg_cols = None
    if len(proteins) == 1:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)
    if len(proteins) == 2:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)
    if len(proteins) == 3:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg,
                   CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)
    if len(proteins) == 4:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg,
                   CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)
    if len(proteins) == 5:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg,
                   CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg,
                   CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)

    df_tcell = df[np.concatenate((CD3_cols, CD4_cols, CD8_cols, CD3_cols_n, CD4_cols_n, CD8_cols_n, CD3_cols_s1,
                                  CD4_cols_s1, CD8_cols_s1, CD3_cols_s, CD4_cols_s, CD8_cols_s, CD3_cols_m, CD4_cols_m,
                                  CD8_cols_m))]
    df_tcell = df_tcell.values
    df_tcell[np.isnan(df_tcell)] = 0
    df_tcell_bg = df[np.concatenate(bg_cols)]
    df_tcell_bg = df_tcell_bg.values
    df_tcell_bg[np.isnan(df_tcell_bg)] = 0
    if norm_type == 'subtract':
        df_tcell = df_tcell - df_tcell_bg
    elif norm_type == 'ratio':
        df_tcell = df_tcell / (df_tcell_bg + EPSILON)

    if scale:
        df_tcell = StandardScaler().fit_transform(df_tcell)

    feature_names = np.concatenate((CD3_cols, CD4_cols, CD8_cols, CD3_cols_n, CD4_cols_n, CD8_cols_n, CD3_cols_s1,
                                    CD4_cols_s1, CD8_cols_s1, CD3_cols_s, CD4_cols_s, CD8_cols_s, CD3_cols_m,
                                    CD4_cols_m,
                                    CD8_cols_m))
    return df_tcell, feature_names


# Constructs a data frame with T cell MFI data
def construct_mfi_df(df: pd.DataFrame, tcell_types: list = None, scale: bool = True,
                     norm_type: str = 'subtract') -> (np.ndarray, np.ndarray):
    """
    Converts given pandas dataframe with raw data into a numpy array with T-cell MFIs.

    :param df: pandas dataframe with raw data.
    :param tcell_types:  list of strings specifying T-cell types to be included ['CD3', 'CD4', 'CD8']
    :param scale: flag identifying whether to standardise data.
    :param norm_type: string specifying normalisation type -- 'subtract' or 'ratio'.
    :return: returns (pre-processed) numpy array with T-cell MFIs for the specified cell types and an array with
    feature names.
    """
    cols = list(df.columns)

    if tcell_types is None:
        tcell_types = ['CD3', 'CD4', 'CD8']

    proteins = [2, 3, 4, 5, 6]

    # Background
    CD3_cols_bg = [s for s in cols if (re.search('CD3', s) and (re.search('_.1_S10', s) or re.search('_.1_S11', s)) and
                                       (not re.search('perc', s) and not re.search('count', s)))]
    CD4_cols_bg = [s for s in cols if (re.search('CD4', s) and (re.search('_.1_S10', s) or re.search('_.1_S11', s)) and
                                       (not re.search('perc', s) and not re.search('count', s)))]
    CD8_cols_bg = [s for s in cols if (re.search('CD8', s) and (re.search('_.1_S10', s) or re.search('_.1_S11', s)) and
                                       (not re.search('perc', s) and not re.search('count', s)))]

    # CoV-2 mix
    CD3_cols = [s for s in cols if (re.search('CD3', s) and (re.search('_.2_S10', s) or re.search('_.2_S11', s)) and
                                    (not re.search('perc', s) and not re.search('count', s)))]
    CD4_cols = [s for s in cols if (re.search('CD4', s) and (re.search('_.2_S10', s) or re.search('_.2_S11', s)) and
                                    (not re.search('perc', s) and not re.search('count', s)))]
    CD8_cols = [s for s in cols if (re.search('CD8', s) and (re.search('_.2_S10', s) or re.search('_.2_S11', s)) and
                                    (not re.search('perc', s) and not re.search('count', s)))]

    # Prot N
    CD3_cols_n = [s for s in cols if (re.search('CD3', s) and (re.search('_.3_S10', s) or re.search('_.3_S11', s)) and
                                      (not re.search('perc', s) and not re.search('count', s)))]
    CD4_cols_n = [s for s in cols if (re.search('CD4', s) and (re.search('_.3_S10', s) or re.search('_.3_S11', s)) and
                                      (not re.search('perc', s) and not re.search('count', s)))]
    CD8_cols_n = [s for s in cols if (re.search('CD8', s) and (re.search('_.3_S10', s) or re.search('_.3_S11', s)) and
                                      (not re.search('perc', s) and not re.search('count', s)))]

    # Prot S1
    CD3_cols_s1 = [s for s in cols if (re.search('CD3', s) and (re.search('_.4_S10', s) or re.search('_.4_S11', s)) and
                                       (not re.search('perc', s) and not re.search('count', s)))]
    CD4_cols_s1 = [s for s in cols if (re.search('CD4', s) and (re.search('_.4_S10', s) or re.search('_.4_S11', s)) and
                                       (not re.search('perc', s) and not re.search('count', s)))]
    CD8_cols_s1 = [s for s in cols if (re.search('CD8', s) and (re.search('_.4_S10', s) or re.search('_.4_S11', s)) and
                                       (not re.search('perc', s) and not re.search('count', s)))]

    # Prot S
    CD3_cols_s = [s for s in cols if (re.search('CD3', s) and (re.search('_.5_S10', s) or re.search('_.5_S11', s)) and
                                      (not re.search('perc', s) and not re.search('count', s)))]
    CD4_cols_s = [s for s in cols if (re.search('CD4', s) and (re.search('_.5_S10', s) or re.search('_.5_S11', s)) and
                                      (not re.search('perc', s) and not re.search('count', s)))]
    CD8_cols_s = [s for s in cols if (re.search('CD8', s) and (re.search('_.5_S10', s) or re.search('_.5_S11', s)) and
                                      (not re.search('perc', s) and not re.search('count', s)))]

    # Prot M
    CD3_cols_m = [s for s in cols if (re.search('CD3', s) and (re.search('_.6_S10', s) or re.search('_.6_S11', s)) and
                                      (not re.search('perc', s) and not re.search('count', s)))]
    CD4_cols_m = [s for s in cols if (re.search('CD4', s) and (re.search('_.6_S10', s) or re.search('_.6_S11', s)) and
                                      (not re.search('perc', s) and not re.search('count', s)))]
    CD8_cols_m = [s for s in cols if (re.search('CD8', s) and (re.search('_.6_S10', s) or re.search('_.6_S11', s)) and
                                      (not re.search('perc', s) and not re.search('count', s)))]

    if not ('CD3' in tcell_types):
        CD3_cols_bg = []
        CD3_cols = []
        CD3_cols_n = []
        CD3_cols_s1 = []
        CD3_cols_s = []
        CD3_cols_m = []
    if not ('CD4' in tcell_types):
        CD4_cols_bg = []
        CD4_cols = []
        CD4_cols_n = []
        CD4_cols_s1 = []
        CD4_cols_s = []
        CD4_cols_m = []
    if not ('CD8' in tcell_types):
        CD8_cols_bg = []
        CD8_cols = []
        CD8_cols_n = []
        CD8_cols_s1 = []
        CD8_cols_s = []
        CD8_cols_m = []

    if not (2 in proteins):
        CD3_cols = []
        CD4_cols = []
        CD8_cols = []
    if not (3 in proteins):
        CD3_cols_n = []
        CD4_cols_n = []
        CD8_cols_n = []
    if not (4 in proteins):
        CD3_cols_s1 = []
        CD4_cols_s1 = []
        CD8_cols_s1 = []
    if not (5 in proteins):
        CD3_cols_s = []
        CD4_cols_s = []
        CD8_cols_s = []
    if not (6 in proteins):
        CD3_cols_m = []
        CD4_cols_m = []
        CD8_cols_m = []

    bg_cols = None
    if len(proteins) == 1:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)
    if len(proteins) == 2:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)
    if len(proteins) == 3:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg,
                   CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)
    if len(proteins) == 4:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg,
                   CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)
    if len(proteins) == 5:
        bg_cols = (CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg,
                   CD3_cols_bg, CD4_cols_bg, CD8_cols_bg, CD3_cols_bg, CD4_cols_bg, CD8_cols_bg,
                   CD3_cols_bg, CD4_cols_bg, CD8_cols_bg)

    df_tcell = df[np.concatenate((CD3_cols, CD4_cols, CD8_cols, CD3_cols_n, CD4_cols_n, CD8_cols_n, CD3_cols_s1,
                                  CD4_cols_s1, CD8_cols_s1, CD3_cols_s, CD4_cols_s, CD8_cols_s, CD3_cols_m, CD4_cols_m,
                                  CD8_cols_m))]

    df_tcell = df_tcell.values
    df_tcell[np.isnan(df_tcell)] = 0
    df_tcell_bg = df[np.concatenate(bg_cols)]
    df_tcell_bg = df_tcell_bg.values
    df_tcell_bg[np.isnan(df_tcell_bg)] = 0
    if norm_type == 'subtract':
        df_tcell = df_tcell - df_tcell_bg
    elif norm_type == 'ratio':
        df_tcell = df_tcell / (df_tcell_bg + EPSILON)

    if scale:
        df_tcell = StandardScaler().fit_transform(df_tcell)

    feature_names = np.concatenate((CD3_cols, CD4_cols, CD8_cols, CD3_cols_n, CD4_cols_n, CD8_cols_n, CD3_cols_s1,
                                    CD4_cols_s1, CD8_cols_s1, CD3_cols_s, CD4_cols_s, CD8_cols_s, CD3_cols_m,
                                    CD4_cols_m,
                                    CD8_cols_m))
    return df_tcell, feature_names
