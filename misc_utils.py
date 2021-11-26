"""
Miscellaneous utility functions.
"""
import re

import numpy as np


# Prints average  performance and the empirical 1-alpha confidence interval
def print_perf_statistics(metrics: np.ndarray, alpha: float = 0.10, n_digits: int = 3) -> None:
    """
    Prints average performance metrics alongside an empirical confidence interval.

    :param metrics: a numpy array with performance metrics, columns correspond to different metrics, rows to bootstrap
    resamples / experiment replicates / MC samples.
    :param alpha: significance level.
    :param n_digits: number of decimals in the printout.
    :return: None.
    """
    for m in range(metrics.shape[1]):
        print(str(np.round(np.nanmean(metrics[:, m]), n_digits)) +
              "; [" + str(np.round(np.nanquantile(metrics[:, m], alpha / 2), n_digits)) +
              str(", ") + str(np.round(np.nanquantile(metrics[:, m], 1 - alpha / 2), n_digits)) + "]")


def count_underexpressed_cells_assay(data: np.ndarray, data_cols: list, assay: str,
                                     cutoff: int = 20) -> (int, int, int):
    """
    Counts the number of low T-cell measurements within the specified assay.

    :param data: numpy array with the T-cell data.
    :param data_cols: a list with feature names corresponding to the columns of the data array.
    :param assay: name of the T-cell assay to consider.
    :param cutoff: a cutoff for defining low measurements.
    :return: returns the number of assay measurements < 0, the number of measurements < cutoff, and the total number of
    assay measurements.
    """
    assay_cols = [i for i, s in enumerate(data_cols) if re.search('\.' + assay + '_pos_count', s) and
                    not (re.search('_1_', s)) and not (re.search('_7_', s))]
    assay_values = np.ravel(data[:, assay_cols]).astype(np.float64)
    assay_values[np.isnan(assay_values)] = 0.0
    return np.sum(assay_values < 0), np.sum(assay_values < cutoff), len(assay_values)


def measure_CV_assay(data: np.ndarray, data_cols: list, assay: str, ctype: str) -> np.ndarray:
    """
    Computes the coefficient of variation (CV) for the specified T-cell assay using control treatment measurements.

    :param data: numpy array with the T-cell data.
    :param data_cols: a list with feature names corresponding to the columns of the data array.
    :param assay: name of the T-cell assay to consider.
    :param ctype: type of control treatment to use for the assessment of CV, negative or positive.
    :return: returns an array with coefficients of variation for each measurement within the specified assay.
    """
    treatment = ''
    if ctype == 'positive':
        treatment = '_7_'
    elif ctype == 'negative':
        treatment = '_1_'
    assay_control_cols = np.array([i for i, s in enumerate(data_cols) if re.search('\.' + assay + '_pos_count', s) and
                                   re.search(treatment, s)])
    assay_control_names = np.array([s for i, s in enumerate(data_cols) if re.search('\.' + assay + '_pos_count', s) and
                                    re.search(treatment, s)])
    # Order controls correctly
    assay_control_cols = assay_control_cols[np.argsort(assay_control_names).astype(int)]
    assay_control_names = np.sort(assay_control_names)

    # Coefficients of variation for each cell
    cvs = np.zeros((int(len(assay_control_cols) / 2), ))

    cnt = 0
    for c in range(0, len(assay_control_cols), 2):

        values_c_s10 = data[:, int(assay_control_cols[c])].astype(np.float64)
        values_c_s11 = data[:, int(assay_control_cols[c + 1])].astype(np.float64)

        # Replace NaNs
        ignored_s10 = np.isnan(values_c_s10)
        values_c_s10[np.isnan(values_c_s10)] = 1.0
        ignored_s11 = np.isnan(values_c_s11)
        values_c_s11[np.isnan(values_c_s11)] = 1.0

        means_c = (values_c_s10 + values_c_s11) / 2
        stdevs_c = np.sqrt((values_c_s10 - means_c)**2 + (values_c_s11 - means_c)**2)
        cvs_c = stdevs_c / means_c
        ignored = np.logical_or(ignored_s10, ignored_s11)

        cvs[cnt] = np.mean(cvs_c[np.logical_not(ignored)])
        cnt = cnt + 1

    return cvs


def measure_CV_IAC(data_iac: np.ndarray, data_iac_cols: list, assay: str, treatment: str) -> (np.ndarray, list):
    # treatments: #1, #2, #7
    """
    Computes the coefficient of variation (CV) for the specified T-cell assay based on the repeated measurements from
    the control donor.

    :param data_iac: numpy array with repeated T-cell measurements.
    :param data_iac_cols: a list with feature names corresponding to the columns of the data array.
    :param assay: name of the T-cell assay to consider.
    :param treatment: type of control treatment to use for the assessment of CV, negative or positive: 1, 2 or 7.
    :return: returns coefficients of variation and the corresponding assay column names.
    """
    assay_cols = [i for i, s in enumerate(data_iac_cols) if re.search(r'\\' + assay + '\+_#_', s) and
                  re.search(treatment, s)]
    assay_cols_names = [s for i, s in enumerate(data_iac_cols) if re.search(r'\\' + assay + '\+_#_', s) and
                        re.search(treatment, s)]
    
    assay_values = data_iac[:, assay_cols].astype(np.float64)

    cvs = np.std(assay_values, 0) / np.mean(assay_values, 0)

    return cvs, assay_cols_names
