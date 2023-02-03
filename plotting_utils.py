"""
Utility functions for plotting.
"""
import pandas as pd

from heatmap import heatmap, corrplot

import re

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.collections import LineCollection

import seaborn as sns

from scipy.stats import pearsonr

import six

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


def parcoord_plot(x1: np.ndarray, x2: np.ndarray, tick1: str, tick2: str, colors: np.ndarray, labels: list,
                  font_size: int = 16, savedir: str = None) -> None:
    """
    Creates a parallel coordinate plot for paired observations.

    :param x1: values for group 1.
    :param x2: values for group 2.
    :param tick1: tick label for group 1.
    :param tick2: tick label for group 2.
    :param colors: colours for each observation pair.
    :param labels: labels for colours.
    :param font_size: font size.
    :param savedir: save file and directory names.
    :return: None.
    """
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=(10, 10))
    for c in np.unique(colors):
        label_c = labels[c]
        x1_c = x1[colors == c]
        x2_c = x2[colors == c]
        plt.scatter(np.ones_like(x1_c), x1_c, c=c, label=label_c, s=25, alpha=0.15)
        plt.scatter(np.ones_like(x2_c)*2, x2_c, c=c, s=25, alpha=0.15)
        for i in range(len(x1_c)):
            plt.plot([1, 2], [x1_c[i], x2_c[i]], c=c, alpha=0.15)
    plt.ylim([min(np.min(x1), np.min(x2)) - 1, max(np.max(x1), np.max(x2)) + 1])
    plt.xticks(ticks=[1, 2], labels=[tick1, tick2])
    plt.ylabel("Normalised Percentage")
    plt.legend()
    if savedir is not None:
        plt.tight_layout()
        plt.savefig(fname=savedir, dpi=300)


def parcoord_plot_sep(x1: np.ndarray, x2: np.ndarray, tick1: str, tick2: str, colors: np.ndarray, labels: list,
                      font_size: int = 16, savedir: str = None, figsize: tuple = (10, 10),
                      sample_names: list = None) -> None:
    """
    Creates a parallel coordinate plot for paired observations. Plots the pairs in two separate groups defined by
    colours.

    :param x1: values for group 1.
    :param x2: values for group 2.
    :param tick1: tick label for group 1.
    :param tick2: tick label for group 2.
    :param colors: colours for each observation pair.
    :param labels: labels for colours.
    :param font_size: font size.
    :param savedir: save file and directory names.
    :param figsize: figure size.
    :param sample_names: patient codes or indices corresponding to observation pairs.
    :return: None.
    """
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=figsize)
    cnt = 1
    ticks_x = []
    for c in np.unique(colors):
        label_c = labels[c]
        x1_c = x1[colors == c]
        x2_c = x2[colors == c]
        plt.scatter(np.ones_like(x1_c) * cnt, x1_c, c=c, label=label_c, s=25, alpha=0.15)
        plt.scatter(np.ones_like(x2_c) * (cnt + 1), x2_c, c=c, s=25, alpha=0.15)
        for i in range(len(x1_c)):
            plt.plot([cnt, cnt + 1], [x1_c[i], x2_c[i]], c=c, alpha=0.15)
            if sample_names is not None:
                plt.text(x=cnt - 0.15, y=x1_c[i], s=sample_names[colors == c][i], fontsize=5, color=c)
        ticks_x.append(cnt)
        ticks_x.append(cnt + 1)
        cnt = cnt + 1.5
    plt.ylim([min(np.min(x1), np.min(x2)) - 1, max(np.max(x1), np.max(x2)) + 1])
    plt.xticks(ticks=ticks_x, labels=[tick1, tick2] * len(np.unique(colors)))
    plt.ylabel("Normalised Percentage")
    plt.legend()
    if savedir is not None:
        plt.tight_layout()
        plt.savefig(fname=savedir, dpi=300)


def plot_errorbars(ks: np.ndarray, avg: np.ndarray, lower: np.ndarray, upper: np.ndarray, xlab: str, ylab: str,
                   baseline: float = None, font_size: int = 16, savedir: str = None) -> None:
    """
    Plots a sequence of average values with error bars.

    :param ks: x axis values.
    :param avg: averages.
    :param lower: lower confidence bounds.
    :param upper: upper confidence bounds
    :param xlab: x axis label.
    :param ylab: y axis label.
    :param baseline: baseline value (constant w.r.t. y axis).
    :param font_size: font size.
    :param savedir: save file and directory names.
    :return: None.
    """
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(figsize=(10, 8))
    plt.errorbar(ks, avg, yerr=np.stack((avg - lower, upper - avg), axis=0), color='tab:blue', ecolor='tab:blue',
                 barsabove=True, marker='D')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if baseline is not None:
        plt.hlines(y=baseline, xmin=np.min(ks) - 5, xmax=np.max(ks) + 5, linestyle='dashed', color='red')
    if savedir is not None:
        plt.tight_layout()
        plt.savefig(fname=savedir, dpi=300)
    plt.show()


def plot_hist_assay_by_tretament(rdata: np.ndarray, rdata_cols: list, ndata: np.ndarray, ndata_cols: list, assay: str,
                                 dtype: str, bins: int = 150, rrange: list = [-100, 5000], nrange: list =[-1000, 1000],
                                 dtype_name: str = '', alpha: float = 0.5, savedir: str = None) -> None:
    """
    Plots treatment-specific histograms of T-cell assay measurements for raw and pre-processed data.

    :param rdata: numpy array with raw data.
    :param rdata_cols: column names for raw data array.
    :param ndata: numpy array with pre-processed data.
    :param ndata_cols: column names for pre-processed data array.
    :param assay: assay name -- TNF, IFN, IL2, CD154.
    :param dtype: measurement data type to consider -- count or perc.
    :param bins: number of bins in histograms.
    :param rrange: x axis range for the raw data histograms.
    :param nrange: x axis range for the pre-processed data histograms.
    :param dtype_name: data type display name.
    :param alpha: histogram transparency level.
    :param savedir: save file and directory names.
    :return: None.
    """
    assay_cols_r = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + dtype, s) and
                    not(re.search('_1_', s)) and not(re.search('_7_', s))]
    assay_cols_r_cov2mix = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_2', s)]
    assay_cols_r_protn = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_3', s)]
    assay_cols_r_prots1 = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_4', s)]
    assay_cols_r_prots = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_5', s)]
    assay_cols_r_protm = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_6', s)]

    assay_cols_n = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + dtype, s)]
    assay_cols_n_cov2mix = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_2', s)]
    assay_cols_n_protn = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_3', s)]
    assay_cols_n_prots1 = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_4', s)]
    assay_cols_n_prots = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_5', s)]
    assay_cols_n_protm = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + dtype + '_6', s)]

    # Retrieve values
    values_r = np.ravel(rdata[:, assay_cols_r]).astype(np.float64)
    values_r_cov2mix = np.ravel(rdata[:, assay_cols_r_cov2mix]).astype(np.float64)
    values_r_protn = np.ravel(rdata[:, assay_cols_r_protn]).astype(np.float64)
    values_r_prots1 = np.ravel(rdata[:, assay_cols_r_prots1]).astype(np.float64)
    values_r_prots = np.ravel(rdata[:, assay_cols_r_prots]).astype(np.float64)
    values_r_protm = np.ravel(rdata[:, assay_cols_r_protm]).astype(np.float64)

    values_n = np.ravel(ndata[:, assay_cols_n]).astype(np.float64)
    values_n_cov2mix = np.ravel(ndata[:, assay_cols_n_cov2mix]).astype(np.float64)
    values_n_protn = np.ravel(ndata[:, assay_cols_n_protn]).astype(np.float64)
    values_n_prots1 = np.ravel(ndata[:, assay_cols_n_prots1]).astype(np.float64)
    values_n_prots = np.ravel(ndata[:, assay_cols_n_prots]).astype(np.float64)
    values_n_protm = np.ravel(ndata[:, assay_cols_n_protm]).astype(np.float64)

    # Remove NaNs
    values_r[np.isnan(values_r)] = 0.0
    values_r_cov2mix[np.isnan(values_r_cov2mix)] = 0.0
    values_r_protn[np.isnan(values_r_protn)] = 0.0
    values_r_prots1[np.isnan(values_r_prots1)] = 0.0
    values_r_prots[np.isnan(values_r_prots)] = 0.0
    values_r_protm[np.isnan(values_r_protm)] = 0.0

    values_n[np.isnan(values_n)] = 0.0
    values_n_cov2mix[np.isnan(values_n_cov2mix)] = 0.0
    values_n_protn[np.isnan(values_n_protn)] = 0.0
    values_n_prots1[np.isnan(values_n_prots1)] = 0.0
    values_n_prots[np.isnan(values_n_prots)] = 0.0
    values_n_protm[np.isnan(values_n_protm)] = 0.0

    fig, ax = plt.subplots(2, 6, figsize=(30, 10))

    # Raw data
    # All treatments
    bin_edges_r = np.linspace(start=np.min(values_r), stop=np.max(values_r), num=bins)
    ax[0, 0].hist(values_r, bins=bin_edges_r, color=CB_COLOR_CYCLE[0], alpha=alpha)
    ax[0, 0].set_xlim(rrange)
    ax[0, 0].set_xlabel('Raw ' + str(dtype_name))
    ax[0, 0].set_ylabel('Frequency')
    ax[0, 0].set_title('All treatments')
    # Cov-2 mix
    ax[0, 1].hist(values_r_cov2mix, bins=bin_edges_r, color=CB_COLOR_CYCLE[1], alpha=alpha)
    ax[0, 1].set_xlim(rrange)
    ax[0, 1].set_title('Cov-2 mix')
    # Protein N
    ax[0, 2].hist(values_r_protn, bins=bin_edges_r, color=CB_COLOR_CYCLE[2], alpha=alpha)
    ax[0, 2].set_xlim(rrange)
    ax[0, 2].set_title('Protein N')
    # Protein S1
    ax[0, 3].hist(values_r_prots1, bins=bin_edges_r, color=CB_COLOR_CYCLE[3], alpha=alpha)
    ax[0, 3].set_xlim(rrange)
    ax[0, 3].set_title('Protein S1')
    # Protein S
    ax[0, 4].hist(values_r_prots, bins=bin_edges_r, color=CB_COLOR_CYCLE[4], alpha=alpha)
    ax[0, 4].set_xlim(rrange)
    ax[0, 4].set_title('Protein S')
    # Protein M
    ax[0, 5].hist(values_r_protm, bins=bin_edges_r, color=CB_COLOR_CYCLE[5], alpha=alpha)
    ax[0, 5].set_xlim(rrange)
    ax[0, 5].set_title('Protein M')

    # Normalised data
    # All treatments
    bin_edges_n = np.linspace(start=np.min(values_n), stop=np.max(values_n), num=bins)
    ax[1, 0].hist(values_n, bins=bin_edges_n, color=CB_COLOR_CYCLE[0], alpha=alpha)
    ax[1, 0].set_xlim(nrange)
    ax[1, 0].set_xlabel('Normalised ' + str(dtype_name))
    ax[1, 0].set_ylabel('Frequency')
    # Cov-2 mix
    ax[1, 1].hist(values_n_cov2mix, bins=bin_edges_n, color=CB_COLOR_CYCLE[1], alpha=alpha)
    ax[1, 1].set_xlim(nrange)
    # Protein N
    ax[1, 2].hist(values_n_protn, bins=bin_edges_n, color=CB_COLOR_CYCLE[2], alpha=alpha)
    ax[1, 2].set_xlim(nrange)
    # Protein S1
    ax[1, 3].hist(values_n_prots1, bins=bin_edges_n, color=CB_COLOR_CYCLE[3], alpha=alpha)
    ax[1, 3].set_xlim(nrange)
    # Protein S
    ax[1, 4].hist(values_n_prots, bins=bin_edges_n, color=CB_COLOR_CYCLE[4], alpha=alpha)
    ax[1, 4].set_xlim(nrange)
    # Protein M
    ax[1, 5].hist(values_n_protm, bins=bin_edges_n, color=CB_COLOR_CYCLE[5], alpha=alpha)
    ax[1, 5].set_xlim(nrange)

    plt.tight_layout()

    if savedir is not None:
        plt.savefig(fname=savedir, dpi=300)


def plot_hist_assay_by_tretament_mfi(rdata: np.ndarray, rdata_cols: list, ndata: np.ndarray, ndata_cols: list,
                                     assay: str, bins: int = 150, rrange: list = [-100, 5000],
                                     nrange: list = [-1000, 1000], alpha: float = 0.5, savedir: str = None) -> None:
    """
    Plots treatment-specific histograms of T-cell assay measurements for raw and pre-processed MFI data.

    :param rdata: numpy array with raw data.
    :param rdata_cols: column names for raw data array.
    :param ndata: numpy array with pre-processed data.
    :param ndata_cols: column names for pre-processed data array.
    :param assay: assay name -- TNF, IFN, IL2, CD154.
    :param bins: number of bins in histograms.
    :param rrange: x axis range for the raw data histograms.
    :param nrange: x axis range for the pre-processed data histograms.
    :param alpha: histogram transparency level.
    :param savedir: save file and directory names.
    :return: None.
    """
    assay_cols_r = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + assay +'_\.', s) and
                    not (re.search('_\.1_', s)) and not (re.search('_.7_', s))]
    assay_cols_r_cov2mix = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.2', s)]
    assay_cols_r_protn = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.3', s)]
    assay_cols_r_prots1 = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.4', s)]
    assay_cols_r_prots = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.5', s)]
    assay_cols_r_protm = [i for i, s in enumerate(rdata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.6', s)]

    assay_cols_n = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.', s)]
    assay_cols_n_cov2mix = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.2', s)]
    assay_cols_n_protn = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.3', s)]
    assay_cols_n_prots1 = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.4', s)]
    assay_cols_n_prots = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.5', s)]
    assay_cols_n_protm = [i for i, s in enumerate(ndata_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.6', s)]

    # Retrieve values
    values_r = np.ravel(rdata[:, assay_cols_r]).astype(np.float64)
    values_r_cov2mix = np.ravel(rdata[:, assay_cols_r_cov2mix]).astype(np.float64)
    values_r_protn = np.ravel(rdata[:, assay_cols_r_protn]).astype(np.float64)
    values_r_prots1 = np.ravel(rdata[:, assay_cols_r_prots1]).astype(np.float64)
    values_r_prots = np.ravel(rdata[:, assay_cols_r_prots]).astype(np.float64)
    values_r_protm = np.ravel(rdata[:, assay_cols_r_protm]).astype(np.float64)

    values_n = np.ravel(ndata[:, assay_cols_n]).astype(np.float64)
    values_n_cov2mix = np.ravel(ndata[:, assay_cols_n_cov2mix]).astype(np.float64)
    values_n_protn = np.ravel(ndata[:, assay_cols_n_protn]).astype(np.float64)
    values_n_prots1 = np.ravel(ndata[:, assay_cols_n_prots1]).astype(np.float64)
    values_n_prots = np.ravel(ndata[:, assay_cols_n_prots]).astype(np.float64)
    values_n_protm = np.ravel(ndata[:, assay_cols_n_protm]).astype(np.float64)

    # Remove NaNs
    values_r[np.isnan(values_r)] = 0.0
    values_r_cov2mix[np.isnan(values_r_cov2mix)] = 0.0
    values_r_protn[np.isnan(values_r_protn)] = 0.0
    values_r_prots1[np.isnan(values_r_prots1)] = 0.0
    values_r_prots[np.isnan(values_r_prots)] = 0.0
    values_r_protm[np.isnan(values_r_protm)] = 0.0

    values_n[np.isnan(values_n)] = 0.0
    values_n_cov2mix[np.isnan(values_n_cov2mix)] = 0.0
    values_n_protn[np.isnan(values_n_protn)] = 0.0
    values_n_prots1[np.isnan(values_n_prots1)] = 0.0
    values_n_prots[np.isnan(values_n_prots)] = 0.0
    values_n_protm[np.isnan(values_n_protm)] = 0.0

    fig, ax = plt.subplots(2, 6, figsize=(30, 10))

    # Raw data
    # All treatments
    bin_edges_r = np.linspace(start=np.min(values_r), stop=np.max(values_r), num=bins)
    ax[0, 0].hist(values_r, bins=bin_edges_r, color=CB_COLOR_CYCLE[0], alpha=alpha)
    ax[0, 0].set_xlim(rrange)
    ax[0, 0].set_xlabel('Raw MFI')
    ax[0, 0].set_ylabel('Frequency')
    ax[0, 0].set_title('All treatments')
    # Cov-2 mix
    ax[0, 1].hist(values_r_cov2mix, bins=bin_edges_r, color=CB_COLOR_CYCLE[1], alpha=alpha)
    ax[0, 1].set_xlim(rrange)
    ax[0, 1].set_title('Cov-2 mix')
    # Protein N
    ax[0, 2].hist(values_r_protn, bins=bin_edges_r, color=CB_COLOR_CYCLE[2], alpha=alpha)
    ax[0, 2].set_xlim(rrange)
    ax[0, 2].set_title('Protein N')
    # Protein S1
    ax[0, 3].hist(values_r_prots1, bins=bin_edges_r, color=CB_COLOR_CYCLE[3], alpha=alpha)
    ax[0, 3].set_xlim(rrange)
    ax[0, 3].set_title('Protein S1')
    # Protein S
    ax[0, 4].hist(values_r_prots, bins=bin_edges_r, color=CB_COLOR_CYCLE[4], alpha=alpha)
    ax[0, 4].set_xlim(rrange)
    ax[0, 4].set_title('Protein S')
    # Protein M
    ax[0, 5].hist(values_r_protm, bins=bin_edges_r, color=CB_COLOR_CYCLE[5], alpha=alpha)
    ax[0, 5].set_xlim(rrange)
    ax[0, 5].set_title('Protein M')

    # Normalised data
    # All treatments
    bin_edges_n = np.linspace(start=np.min(values_n), stop=np.max(values_n), num=bins)
    ax[1, 0].hist(values_n, bins=bin_edges_n, color=CB_COLOR_CYCLE[0], alpha=alpha)
    ax[1, 0].set_xlim(nrange)
    ax[1, 0].set_xlabel('Normalised MFI')
    ax[1, 0].set_ylabel('Frequency')
    # Cov-2 mix
    ax[1, 1].hist(values_n_cov2mix, bins=bin_edges_n, color=CB_COLOR_CYCLE[1], alpha=alpha)
    ax[1, 1].set_xlim(nrange)
    # Protein N
    ax[1, 2].hist(values_n_protn, bins=bin_edges_n, color=CB_COLOR_CYCLE[2], alpha=alpha)
    ax[1, 2].set_xlim(nrange)
    # Protein S1
    ax[1, 3].hist(values_n_prots1, bins=bin_edges_n, color=CB_COLOR_CYCLE[3], alpha=alpha)
    ax[1, 3].set_xlim(nrange)
    # Protein S
    ax[1, 4].hist(values_n_prots, bins=bin_edges_n, color=CB_COLOR_CYCLE[4], alpha=alpha)
    ax[1, 4].set_xlim(nrange)
    # Protein M
    ax[1, 5].hist(values_n_protm, bins=bin_edges_n, color=CB_COLOR_CYCLE[5], alpha=alpha)
    ax[1, 5].set_xlim(nrange)

    plt.tight_layout()

    if savedir is not None:
        plt.savefig(fname=savedir, dpi=300)


def plot_assay_ecdfs(rdata: np.ndarray, rdata_cols: list, ndata: np.ndarray, ndata_cols: list, ndata_mfi: np.ndarray,
                     ndata_cols_mfi: list, bins: int = 150, linewidth: int = 5, savedir: str = None) -> None:
    """
    Plots assay-specific empirical CDFs before and after pre-processing.

    :param rdata: numpy array with raw data.
    :param rdata_cols: column names for raw data array.
    :param ndata: numpy array with pre-processed data.
    :param ndata_cols: column names for pre-processed data array.
    :param ndata_mfi: numpy array with pre-processed MFI data.
    :param ndata_cols_mfi: column names for pre-processed MFI data array.
    :param bins: number of bins in histograms.
    :param savedir: save file and directory names.
    :return: None.
    """
    xranges = [[-750, 750], [-0.005, 0.005], [-100, 100]]
    dtypes = ['count', 'perc', 'mfi']
    dtype_names = ['Count', 'Percentage', 'MFI']

    plotting_setup(font_size=30)
    fig, ax = plt.subplots(1, len(dtypes), figsize=(7 + 10 * len(dtypes), 9))

    for j in range(len(dtypes)):
        xrange = xranges[j]
        dtype = dtypes[j]
        dtype_name = dtype_names[j]

        if dtype == 'mfi':
            assay_cols_r_TNF = [i for i, s in enumerate(rdata_cols) if re.search('\.' + 'TNF' + '_pos_' +
                                                                                 'TNF' + '_\.', s) and
                                not (re.search('_\.1_', s)) and not (re.search('_.7_', s))]
            assay_cols_r_IFN = [i for i, s in enumerate(rdata_cols) if re.search('\.' + 'IFN' + '_pos_' +
                                                                                 'IFN' + '_\.', s) and
                                not (re.search('_\.1_', s)) and not (re.search('_.7_', s))]
            assay_cols_r_IL2 = [i for i, s in enumerate(rdata_cols) if re.search('\.' + 'IL2' + '_pos_' +
                                                                                 'IL2' + '_\.', s) and
                                not (re.search('_\.1_', s)) and not (re.search('_.7_', s))]
            assay_cols_r_CD154 = [i for i, s in enumerate(rdata_cols) if re.search('\.' + 'CD154' + '_pos_' +
                                                                                   'CD154' + '_\.', s) and
                                  not (re.search('_\.1_', s)) and not (re.search('_.7_', s))]

            assay_cols_n_TNF = [i for i, s in enumerate(ndata_cols_mfi) if re.search('\.' + 'TNF' + '_pos_' +
                                                                                     'TNF' + '_\.', s)]
            assay_cols_n_IFN = [i for i, s in enumerate(ndata_cols_mfi) if re.search('\.' + 'IFN' + '_pos_' +
                                                                                     'IFN' + '_\.', s)]
            assay_cols_n_IL2 = [i for i, s in enumerate(ndata_cols_mfi) if re.search('\.' + 'IL2' + '_pos_' +
                                                                                     'IL2' + '_\.', s)]
            assay_cols_n_CD154 = [i for i, s in enumerate(ndata_cols_mfi) if re.search('\.' + 'CD154' + '_pos_' +
                                                                                       'CD154' + '_\.', s)]

        else:
            assay_cols_r_TNF = [i for i, s in enumerate(rdata_cols) if re.search('\.' + 'TNF' + '_pos_' +
                                                                                 dtype, s) and
                                not (re.search('_1_', s)) and not (re.search('_7_', s))]
            assay_cols_r_IFN = [i for i, s in enumerate(rdata_cols) if re.search('\.' + 'IFN' + '_pos_' +
                                                                                 dtype, s) and
                                not (re.search('_1_', s)) and not (re.search('_7_', s))]
            assay_cols_r_IL2 = [i for i, s in enumerate(rdata_cols) if re.search('\.' + 'IL2' + '_pos_' +
                                                                                 dtype, s) and
                                not (re.search('_1_', s)) and not (re.search('_7_', s))]
            assay_cols_r_CD154 = [i for i, s in enumerate(rdata_cols) if re.search('\.' + 'CD154' + '_pos_' +
                                                                                   dtype, s) and
                                  not (re.search('_1_', s)) and not (re.search('_7_', s))]

            assay_cols_n_TNF = [i for i, s in enumerate(ndata_cols) if re.search('\.' + 'TNF' + '_pos_' +
                                                                                 dtype, s)]
            assay_cols_n_IFN = [i for i, s in enumerate(ndata_cols) if re.search('\.' + 'IFN' + '_pos_' +
                                                                                 dtype, s)]
            assay_cols_n_IL2 = [i for i, s in enumerate(ndata_cols) if re.search('\.' + 'IL2' + '_pos_' +
                                                                                 dtype, s)]
            assay_cols_n_CD154 = [i for i, s in enumerate(ndata_cols) if re.search('\.' + 'CD154' + '_pos_' +
                                                                                   dtype, s)]

        # Retrieve values
        values_r_TNF = np.ravel(rdata[:, assay_cols_r_TNF]).astype(np.float64)
        values_r_IFN = np.ravel(rdata[:, assay_cols_r_IFN]).astype(np.float64)
        values_r_IL2 = np.ravel(rdata[:, assay_cols_r_IL2]).astype(np.float64)
        values_r_CD154 = np.ravel(rdata[:, assay_cols_r_CD154]).astype(np.float64)
        if dtype == 'mfi':
            values_n_TNF = np.ravel(ndata_mfi[:, assay_cols_n_TNF]).astype(np.float64)
            values_n_IFN = np.ravel(ndata_mfi[:, assay_cols_n_IFN]).astype(np.float64)
            values_n_IL2 = np.ravel(ndata_mfi[:, assay_cols_n_IL2]).astype(np.float64)
            values_n_CD154 = np.ravel(ndata_mfi[:, assay_cols_n_CD154]).astype(np.float64)
        else:
            values_n_TNF = np.ravel(ndata[:, assay_cols_n_TNF]).astype(np.float64)
            values_n_IFN = np.ravel(ndata[:, assay_cols_n_IFN]).astype(np.float64)
            values_n_IL2 = np.ravel(ndata[:, assay_cols_n_IL2]).astype(np.float64)
            values_n_CD154 = np.ravel(ndata[:, assay_cols_n_CD154]).astype(np.float64)

        # Remove NaNs
        values_r_TNF[np.isnan(values_r_TNF)] = 0.0
        values_r_IFN[np.isnan(values_r_IFN)] = 0.0
        values_r_IL2[np.isnan(values_r_IL2)] = 0.0
        values_r_CD154[np.isnan(values_r_CD154)] = 0.0
        values_n_TNF[np.isnan(values_n_TNF)] = 0.0
        values_n_IFN[np.isnan(values_n_IFN)] = 0.0
        values_n_IL2[np.isnan(values_n_IL2)] = 0.0
        values_n_CD154[np.isnan(values_n_CD154)] = 0.0

        values_r_TNF = np.append(values_r_TNF, xrange[1])
        values_r_IFN = np.append(values_r_IFN, xrange[1])
        values_r_IL2 = np.append(values_r_IL2, xrange[1])
        values_r_CD154 = np.append(values_r_CD154, xrange[1])

        values_n_TNF = np.append(values_n_TNF, xrange[1])
        values_n_IFN = np.append(values_n_IFN, xrange[1])
        values_n_IL2 = np.append(values_n_IL2, xrange[1])
        values_n_CD154 = np.append(values_n_CD154, xrange[1])

        # All treatments
        if dtype == 'mfi':
            ax[j].hist(values_r_IL2, bins=bins, ls='-', edgecolor=CB_COLOR_CYCLE[2], facecolor="None", cumulative=True,
                       density=True, label='IL2, raw', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_r_IFN, bins=bins, ls='-', edgecolor=CB_COLOR_CYCLE[1], facecolor="None", cumulative=True,
                       density=True, label='IFN, raw', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_r_TNF, bins=bins, ls='-', edgecolor=CB_COLOR_CYCLE[0], facecolor="None", cumulative=True,
                       density=True, label='TNF, raw', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_r_CD154, bins=bins, ls='-', edgecolor=CB_COLOR_CYCLE[3], facecolor="None",
                       cumulative=True,
                       density=True, label='CD154, raw', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_n_IL2, bins=bins, ls=':', edgecolor=CB_COLOR_CYCLE[2], facecolor="None", cumulative=True,
                       density=True, label='IL2, normalized', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_n_IFN, bins=bins, ls=':', edgecolor=CB_COLOR_CYCLE[1], facecolor="None", cumulative=True,
                       density=True, label='IFN, normalized', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_n_TNF, bins=bins, ls=':', edgecolor=CB_COLOR_CYCLE[0], facecolor="None", cumulative=True,
                       density=True, label='TNF, normalized', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_n_CD154, bins=bins, ls=':', edgecolor=CB_COLOR_CYCLE[3], facecolor="None",
                       cumulative=True,
                       density=True, label='CD154, normalized', histtype=u'step', linewidth=linewidth)

        else:
            ax[j].hist(values_r_CD154, bins=bins, ls='-', edgecolor=CB_COLOR_CYCLE[3], facecolor="None",
                       cumulative=True,
                       density=True, label='CD154, raw', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_r_IFN, bins=bins, ls='-', edgecolor=CB_COLOR_CYCLE[1], facecolor="None", cumulative=True,
                       density=True, label='IFN, raw', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_r_TNF, bins=bins, ls='-', edgecolor=CB_COLOR_CYCLE[0], facecolor="None", cumulative=True,
                       density=True, label='TNF, raw', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_r_IL2, bins=bins, ls='-', edgecolor=CB_COLOR_CYCLE[2], facecolor="None", cumulative=True,
                       density=True, label='IL2, raw', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_n_CD154, bins=bins, ls=':', edgecolor=CB_COLOR_CYCLE[3], facecolor="None",
                       cumulative=True,
                       density=True, label='CD154, normalized', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_n_IFN, bins=bins, ls=':', edgecolor=CB_COLOR_CYCLE[1], facecolor="None", cumulative=True,
                       density=True, label='IFN, normalized', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_n_TNF, bins=bins, ls=':', edgecolor=CB_COLOR_CYCLE[0], facecolor="None", cumulative=True,
                       density=True, label='TNF, normalized', histtype=u'step', linewidth=linewidth)
            ax[j].hist(values_n_IL2, bins=bins, ls=':', edgecolor=CB_COLOR_CYCLE[2], facecolor="None", cumulative=True,
                       density=True, label='IL2, normalized', histtype=u'step', linewidth=linewidth)

        if j == len(dtypes) - 1:
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05))

        ax[j].set_xlim(xrange)
        ax[j].set_xlabel(str(dtype_name))
        if j == 0:
            ax[j].set_ylabel('% Measurements')

        plt.tight_layout(pad=2.5)

        if savedir is not None:
            plt.savefig(fname=savedir, dpi=300)


def plot_corr_assay(data: np.ndarray, data_cols: list, data_mfi: np.ndarray, data_mfi_cols: list, assay: str,
                    savedir: str = None) -> None:
    """
    Visualises correlations among T-cell assay data types (counts, percentages, MFIs) as a heatmap.

    :param data: numpy array with the data.
    :param data_cols: column names for data array.
    :param data_mfi: numpy array with the MFI data.
    :param data_mfi_cols: column names for MFI data array.
    :param assay: assay name -- TNF, IFN, IL2, CD154.
    :param savedir: save file and directory names.
    :return: None.
    """
    assay_cols_count = [i for i, s in enumerate(data_cols) if re.search('\.' + assay + '_pos_count', s) and
                        not (re.search('_1_', s)) and not (re.search('_7_', s))]
    assay_names_count = [s for i, s in enumerate(data_cols) if re.search('\.' + assay + '_pos_count', s) and
                        not (re.search('_1_', s)) and not (re.search('_7_', s))]
    assay_cols_perc = [i for i, s in enumerate(data_cols) if re.search('\.' + assay + '_pos_perc', s) and
                        not (re.search('_1_', s)) and not (re.search('_7_', s))]
    assay_names_perc = [s for i, s in enumerate(data_cols) if re.search('\.' + assay + '_pos_perc', s) and
                       not (re.search('_1_', s)) and not (re.search('_7_', s))]
    assay_cols_mfi = [i for i, s in enumerate(data_mfi_cols) if re.search('\.' + assay + '_pos_' + assay +'_\.', s) and
                      not (re.search('_\.1', s)) and not (re.search('_\.7', s))]
    assay_names_mfi = [s for i, s in enumerate(data_mfi_cols) if re.search('\.' + assay + '_pos_' + assay + '_\.', s) and
                      not (re.search('_\.1', s)) and not (re.search('_\.7', s))]

    # Retrieve values
    values_count = np.ravel(data[:, assay_cols_count]).astype(np.float64)
    values_perc = np.ravel(data[:, assay_cols_perc]).astype(np.float64)
    values_mfi = np.ravel(data_mfi[:, assay_cols_mfi]).astype(np.float64)

    # Remove NaNs
    values_count[np.isnan(values_count)] = 0.0
    values_perc[np.isnan(values_perc)] = 0.0
    values_mfi[np.isnan(values_mfi)] = 0.0

    # [count, perc, MFI]
    corr_mat = np.zeros((3, 3))
    corr_mat[0, 0] = 1.0; corr_mat[1, 1] = 1.0; corr_mat[2, 2] = 1.0
    corr_mat[0, 1] = np.corrcoef(values_count, values_perc)[0, 1]
    corr_mat[0, 2] = np.corrcoef(values_count, values_mfi)[0, 1]
    corr_mat[1, 0] = np.corrcoef(values_perc, values_count)[0, 1]
    corr_mat[1, 2] = np.corrcoef(values_perc, values_mfi)[0, 1]
    corr_mat[2, 0] = np.corrcoef(values_mfi, values_count)[0, 1]
    corr_mat[2, 1] = np.corrcoef(values_mfi, values_perc)[0, 1]

    sns_plot = sns.heatmap(data=corr_mat, xticklabels=['Count', 'Perc.', 'MFI'], yticklabels=['Count', 'Perc.', 'MFI'],
                           vmin=0.0, vmax=1.0)

    if savedir is not None:
        figure = sns_plot.get_figure()
        figure.savefig(savedir, dpi=300)


def plot_large_corr_mat(data: np.ndarray, data_cols: list, data_mfi: np.ndarray, data_mfi_cols: list,
                        savedir: str = None) -> None:
    """
    Visualises correlations across different data types and assays as a heatmap.

    :param data: numpy array with the data.
    :param data_cols: column names for the data array.
    :param data_mfi: numpy array with the MFI data.
    :param data_mfi_cols: column names for the MFI data array.
    :param savedir: save file and directory names.
    :return: None.
    """
    assays = ['TNF', 'IFN', 'IL2', 'CD154']
    xs = []
    ys = []
    for ass in assays:
        assay_cols_count = [i for i, s in enumerate(data_cols) if re.search('\.' + ass + '_pos_count', s) and
                            not (re.search('_1_', s)) and not (re.search('_7_', s))]
        assay_names_count = [s for i, s in enumerate(data_cols) if re.search('\.' + ass + '_pos_count', s) and
                            not (re.search('_1_', s)) and not (re.search('_7_', s))]
        assay_cols_perc = [i for i, s in enumerate(data_cols) if re.search('\.' + ass + '_pos_perc', s) and
                            not (re.search('_1_', s)) and not (re.search('_7_', s))]
        assay_names_perc = [s for i, s in enumerate(data_cols) if re.search('\.' + ass + '_pos_perc', s) and
                           not (re.search('_1_', s)) and not (re.search('_7_', s))]
        assay_cols_mfi = [i for i, s in enumerate(data_mfi_cols) if re.search('\.' + ass + '_pos_' + 
                                                                              ass +'_\.', s) and
                          not (re.search('_\.1', s)) and not (re.search('_\.7', s))]
        assay_names_mfi = [s for i, s in enumerate(data_mfi_cols) if re.search('\.' + ass + '_pos_' + 
                                                                               ass + '_\.', s) and
                          not (re.search('_\.1', s)) and not (re.search('_\.7', s))]

        # Retrieve values
        values_count = np.ravel(data[:, assay_cols_count]).astype(np.float64)
        values_perc = np.ravel(data[:, assay_cols_perc]).astype(np.float64)
        values_mfi = np.ravel(data_mfi[:, assay_cols_mfi]).astype(np.float64)

        # Remove NaNs
        values_count[np.isnan(values_count)] = 0.0
        values_perc[np.isnan(values_perc)] = 0.0
        values_mfi[np.isnan(values_mfi)] = 0.0
        
        xs.append(values_count)
        xs.append(values_perc)
        xs.append(values_mfi)
        ys.append(values_count)
        ys.append(values_perc)
        ys.append(values_mfi)
    xs = np.stack(xs)
    ys = np.stack(ys)
    xs = np.transpose(xs)
    ys = np.transpose(ys)
    
    df = pd.DataFrame(xs, columns=['TNF   #  ', 'TNF   %  ', 'TNF   MFI', 'IFN   #  ', 'IFN   %  ', 
                                   'IFN   MFI', 'IL2   #  ', 'IL2   %  ', 'IL2   MFI', 'CD154 #  ', 
                                   'CD154 %  ', 'CD154 MFI'])
    pvals = calculate_pvalues(df)

    plotting_setup(20)
    fig = plt.figure(figsize=(10, 10))
    corrplot(df.corr(), size_scale=1200)
    plt.text(0, 1.2, 'Pearson\'s $r$')
    
    pvals = pvals.to_numpy()
    
    y_coords = np.linspace(1.05, -0.97, 12)
    x_coords = np.linspace(-14.4, -0.4, 12)
    
    for i in range(pvals.shape[0]):
        for j in range(pvals.shape[1]):
            if pvals[i, j] <= 0.05:
                plt.text(x_coords[i], y_coords[j], '*', c='red')
    plt.text(-14.4, -1.6,'*', c='red')
    plt.text(-13.9, -1.6,'significant')
    
    if savedir is not None:
        plt.savefig(fname=savedir, dpi=300, bbox_inches='tight')
        
        
def plot_by_symptoms(values: np.ndarray, symptoms_score: np.ndarray, figsize: tuple = (10, 10), xlab: str = '',
                     ylab: str = '', savedir: str = None) -> None:
    """
    Generates box plots of T-cell measurements against the symptoms score.

    :param values: an array with T-cell measurements.
    :param symptoms_score: an array with symptoms scores.
    :param figsize: figure size.
    :param xlab: x axis label.
    :param ylab: y axis label.
    :param savedir: save file and directory names.
    :return: None.
    """
    scores = np.unique(symptoms_score)

    fig = plt.figure(figsize=figsize)

    cnt = 0
    for s in scores:
        bx_s = plt.boxplot(values[symptoms_score == s], positions=[cnt], showfliers=False,
                              labels=['' + str(int(s))], notch=True, patch_artist=True,
                              vert=True)
        colors = ['n']
        for patch, color in zip(bx_s['boxes'], colors):
            patch.set_facecolor((0.1, 0.2, 0.5, 0.0))

        xs = np.ones_like(values[symptoms_score == s]) * cnt
        xs = xs + np.random.normal(loc=0, scale=0.05, size=(values[symptoms_score == s].shape[0], ))

        plt.scatter(xs, values[symptoms_score == s], color='black', alpha=0.25, s=10)

        cnt = cnt + 1

    plt.xlabel(xlab)
    plt.ylabel(ylab)

    plt.tight_layout()

    if savedir is not None:
        plt.savefig(fname=savedir, dpi=300)


# Further minor utility functions for plotting

def calculate_pvalues(df: pd.DataFrame):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = pearsonr(df[r], df[c])[1]
    # Bonferroni adjustment
    pvalues = pvalues * (len(df.columns) * (len(df.columns) - 1) / 2)
    return pvalues


def plot_roc(tpr, fpr, thresholds, axes, subplots_kwargs=None,
             label_every=None, label_kwargs=None,
             fpr_label='False Positive Rate',
             tpr_label='True Positive Rate',
             luck_label='Random guess',
             title='Receiver operating characteristic',
             **kwargs):

    plotting_setup(24)

    if subplots_kwargs is None:
        subplots_kwargs = {}

    if 'lw' not in kwargs:
        kwargs['lw'] = 1

    axes.plot(fpr, tpr, linewidth=5, color='black', label='ROC curve')

    if label_every is not None:
        if label_kwargs is None:
            label_kwargs = {}

        if 'bbox' not in label_kwargs:
            label_kwargs['bbox'] = dict(
                boxstyle='round,pad=0.25', fc='yellow', alpha=0.5,
            )

        for k in six.moves.range(len(tpr)):
            if k % label_every != 0:
                continue

            threshold = str(np.round(thresholds[k], 2))
            x = fpr[k]
            y = tpr[k]
            axes.annotate(threshold, (x, y), fontsize=16, **label_kwargs)

    if luck_label is not None:
        axes.plot((0, 1), (0, 1), '--', color='Gray', label='Random guess')

    axes.set_xlim([-0.05, 1.05])
    axes.set_ylim([-0.05, 1.05])

    axes.set_xlabel(fpr_label)
    axes.set_ylabel(tpr_label)

    axes.set_title(title)

    return axes


def plotting_setup(font_size: int = 12):
    plt.style.use("seaborn-colorblind")
    plt.rcParams['font.size'] = font_size
    rc('text', usetex=False)
    plt.rcParams["font.family"] = "Times New Roman"
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, axes=None,
              cmap=plt.get_cmap('coolwarm'),
              norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0,
              **kwargs):
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(
        segments, array=z, cmap=cmap, norm=norm,
        linewidth=linewidth, alpha=alpha, **kwargs
    )

    if axes is None:
        axes = plt.gca()

    axes.add_collection(lc)
    axes.autoscale()

    return lc
