"""
Utility functions for bootstrapping, model comparison and evaluation.
"""
import os

import numpy as np

import pandas as pd
from pandas import read_csv

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, r2_score, confusion_matrix, recall_score, 
                             roc_auc_score, average_precision_score)

import xgboost as xgb

from misc_utils import print_perf_statistics

from data_loader import construct_tcell_df, construct_mfi_df

# Some performance metrics
sensitivity = recall_score


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Evaluates specificity.

    :param y_true: ground truth labels.
    :param y_pred: predicted labels.
    :return: returns specificity.
    """
    return balanced_accuracy_score(y_true, y_pred) * 2 - recall_score(y_true, y_pred)


def run_repeated_CV_XGB(X: np.ndarray, y: np.ndarray, metrics: list, borderline_cases: np.ndarray = None,
                        drop_borderline: bool = False, B: int = 200, K: int = 5, random_state: int = 42,
                        verbose: bool = True) -> (np.ndarray, np.ndarray):
    """
    Runs repeated cross-validation of the XGBoost classifier.

    :param X: numpy array with features.
    :param y: numpy array with labels.
    :param metrics: list of functions used as performance metrics.
    :param borderline_cases: numpy array of booleans identifying borderline cases.
    :param drop_borderline: flag identifying whether to drop borderline cases from the analysis.
    :param B: number of repetitions.
    :param K: number of folds.
    :param random_state: random generator seed.
    :param verbose: flag identifying whether to enable printouts.
    :return: returns an array with performance metrics, averaged across folds, and an array with feature importance.
    """
    if drop_borderline:
        assert borderline_cases is not None
        X = np.copy(X)[np.logical_not(borderline_cases), :]
        y = np.copy(y)[np.logical_not(borderline_cases)]
    np.random.seed(random_state)
    avg_metrics = np.zeros((B, len(metrics)))
    feature_importances = np.zeros((B * K, X.shape[1]))
    cnt_ = 0
    for i in range(B):
        kf = StratifiedKFold(n_splits=K, shuffle=True)
        metrics_b = np.zeros((K, len(metrics)))
        cnt = 0
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            if len(np.unique(y_train)) == 2:
                scale_weight = (X_train.shape[0] - np.sum(y_train)) / np.sum(y_train)
                gbm = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False,
                                    scale_pos_weight=scale_weight)
            else:
                gbm = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
            gbm.fit(X_train, y_train)
            feature_importances[cnt_, :] = gbm.feature_importances_
            y_pred = gbm.predict(X_test)
            for j in range(len(metrics)):
                metrics_b[cnt, j] = metrics[j](y_true=y_test, y_pred=y_pred)
            cnt += 1
            cnt_ += 1
        if verbose:
            print(str(i) + ". " + str(np.mean(metrics_b, axis=0)))
        avg_metrics[i, :] = np.mean(metrics_b, axis=0)
    if verbose:
        print()
        print()
    return avg_metrics, feature_importances


def boot_train_test_XGB(X: np.ndarray, y: np.ndarray, metrics_clf: list, metrics_proba: list,
                        borderline_cases: np.ndarray = None, drop_borderline: bool = False, B: int = 1000,
                        random_state: int = 42, verbose: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Runs bootstrapped train-test split for the XGBoost classifier.

    :param X: numpy array with features.
    :param y: numpy array with labels.
    :param metrics_clf: list of functions used as performance metrics.
    :param metrics_proba: list of functions used as performance metrics for probabilistic predictions.
    :param borderline_cases: numpy array of booleans identifying borderline cases.
    :param drop_borderline: flag identifying whether to drop borderline cases from the analysis.
    :param B: number of bootstrap resamples.
    :param random_state: random generator seed.
    :param verbose: flag identifying whether to enable printouts.
    :return: returns arrays with classification and probabilistic performance metrics and feature importance.
    """
    if drop_borderline:
        assert borderline_cases is not None
        X = np.copy(X)[np.logical_not(borderline_cases), :]
        y = np.copy(y)[np.logical_not(borderline_cases)]
    np.random.seed(random_state)
    metrics_clf_boot = np.zeros((B, len(metrics_clf)))
    metrics_proba_boot = np.zeros((B, len(metrics_proba)))
    feature_importances = np.zeros((B, X.shape[1]))
    for b in range(B):
        metrics_clf_b = np.zeros((len(metrics_clf),))
        metrics_proba_b = np.zeros((len(metrics_proba),))
        inds = np.arange(0, X.shape[0])
        np.random.shuffle(inds)
        inds_b = np.random.choice(a=inds, size=(inds.shape[0],), replace=True)
        inds_train, inds_test = train_test_split(inds, test_size=0.20, stratify=y)
        inds_b_train = np.array([i for i in inds_b if (i in inds_train)])
        inds_b_test = np.array([i for i in inds_b if (i in inds_test)])
        X_train_b = X[inds_b_train, :]
        X_test_b = X[inds_b_test, :]
        y_train_b = y[inds_b_train]
        y_test_b = y[inds_b_test]
        if len(np.unique(y_train_b)) == 2:
            scale_weight = (X_train_b.shape[0] - np.sum(y_train_b)) / np.sum(y_train_b)
            gbm = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False,
                                    scale_pos_weight=scale_weight)
        else:
            gbm = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        gbm.fit(X_train_b, y_train_b)
        feature_importances[b, :] = gbm.feature_importances_
        y_pred = gbm.predict(X_test_b)
        if len(np.unique(y_train_b)) == 2:
            y_pred_proba = gbm.predict_proba(X_test_b)[:, 1]
        else:
            y_pred_proba = gbm.predict_proba(X_test_b)

        for j in range(len(metrics_clf)):
            metrics_clf_b[j] = metrics_clf[j](y_true=y_test_b, y_pred=y_pred)

        for j in range(len(metrics_proba)):
            try:
                if len(np.unique(y_train_b)) == 2:
                    metrics_proba_b[j] = metrics_proba[j](y_true=y_test_b, y_score=y_pred_proba)
                else:
                    metrics_proba_b[j] = metrics_proba[j](y_true=y_test_b, y_score=y_pred_proba, average='weighted',
                                                          multi_class='ovo')
            except ValueError:
                metrics_proba_b[j] = np.NaN

        if verbose:
            print(str(b) + ". " + str(metrics_clf_b) + "  " + str(metrics_proba_b))
        metrics_clf_boot[b, :] = metrics_clf_b
        metrics_proba_boot[b, :] = metrics_proba_b
    if verbose:
        print()
        print()
    return metrics_clf_boot, metrics_proba_boot, feature_importances


# Runs bootstrapped train-test split procedure for logistic regression
def boot_train_test_LR(X: np.ndarray, y: np.ndarray, metrics_clf: list, metrics_proba: list, B: int = 1000,
                       random_state: int = 42, verbose: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Runs bootstrapped train-test split for the logistic regression.

    :param X: numpy array with features.
    :param y: numpy array with labels.
    :param metrics_clf: list of functions used as performance metrics.
    :param metrics_proba: list of functions used as performance metrics for probabilistic predictions.
    :param B: number of bootstrap resamples.
    :param random_state: random generator seed.
    :param verbose: flag identifying whether to enable printouts.
    :return: returns arrays with classification and probabilistic performance metrics and feature importance.
    """
    np.random.seed(random_state)
    metrics_clf_boot = np.zeros((B, len(metrics_clf)))
    metrics_proba_boot = np.zeros((B, len(metrics_proba)))
    feature_importances = np.zeros((B, X.shape[1]))
    for b in range(B):
        metrics_clf_b = np.zeros((len(metrics_clf),))
        metrics_proba_b = np.zeros((len(metrics_proba),))
        inds = np.arange(0, X.shape[0])
        np.random.shuffle(inds)
        inds_b = np.random.choice(a=inds, size=(inds.shape[0],), replace=True)
        inds_train, inds_test = train_test_split(inds, test_size=0.20, stratify=y)
        inds_b_train = np.array([i for i in inds_b if (i in inds_train)])
        inds_b_test = np.array([i for i in inds_b if (i in inds_test)])
        X_train_b = X[inds_b_train, :]
        X_test_b = X[inds_b_test, :]
        y_train_b = y[inds_b_train]
        y_test_b = y[inds_b_test]
        lr = LogisticRegression(class_weight='balanced')
        lr.fit(X_train_b, y_train_b)
        feature_importances[b, :] = np.abs(lr.coef_)
        y_pred = lr.predict(X_test_b)
        y_pred_proba = lr.predict_proba(X_test_b)[:, 1]

        for j in range(len(metrics_clf)):
            metrics_clf_b[j] = metrics_clf[j](y_true=y_test_b, y_pred=y_pred)
        for j in range(len(metrics_proba)):
            try:
                metrics_proba_b[j] = metrics_proba[j](y_true=y_test_b, y_score=y_pred_proba)
            except ValueError:
                metrics_proba_b[j] = np.NaN

        if verbose:
            print(str(b) + ". " + str(metrics_clf_b) + "  " + str(metrics_proba_b))
        metrics_clf_boot[b, :] = metrics_clf_b
        metrics_proba_boot[b, :] = metrics_proba_b
    if verbose:
        print()
        print()
    return metrics_clf_boot, metrics_proba_boot, feature_importances


def boot_train_test_random(X: np.ndarray, y: np.ndarray, metrics_clf: list, metrics_proba: list, B: int = 1000,
                           random_state: int = 42, verbose: bool = True) -> (np.ndarray, np.ndarray):
    """
    Runs bootstrapped train-test split for the random classifier.
    NOTE: in practice, just use it for the sanity check.

    :param X: numpy array with features.
    :param y: numpy array with labels.
    :param metrics_clf: list of functions used as performance metrics.
    :param metrics_proba: list of functions used as performance metrics for probabilistic predictions.
    :param B: number of bootstrap resamples.
    :param random_state: random generator seed.
    :param verbose: flag identifying whether to enable printouts.
    :return: returns arrays with classification and probabilistic performance metrics.
    """
    np.random.seed(random_state)
    metrics_clf_boot = np.zeros((B, len(metrics_clf)))
    metrics_proba_boot = np.zeros((B, len(metrics_proba)))
    for b in range(B):
        metrics_clf_b = np.zeros((len(metrics_clf),))
        metrics_proba_b = np.zeros((len(metrics_proba),))
        inds = np.arange(0, X.shape[0])
        np.random.shuffle(inds)
        inds_b = np.random.choice(a=inds, size=(inds.shape[0],), replace=True)
        inds_train, inds_test = train_test_split(inds, test_size=0.20, stratify=y)
        inds_b_test = np.array([i for i in inds_b if (i in inds_test)])
        y_test_b = y[inds_b_test]
        y_pred = np.copy(y_test_b)
        np.random.shuffle(y_pred)
        y_pred_proba = np.copy(y_pred)

        for j in range(len(metrics_clf)):
            metrics_clf_b[j] = metrics_clf[j](y_true=y_test_b, y_pred=y_pred)
        for j in range(len(metrics_proba)):
            try:
                metrics_proba_b[j] = metrics_proba[j](y_true=y_test_b, y_score=y_pred_proba)
            except ValueError:
                metrics_proba_b[j] = 0

        if verbose:
            print(str(b) + ". " + str(metrics_clf_b) + "  " + str(metrics_proba_b))
        metrics_clf_boot[b, :] = metrics_clf_b
        metrics_proba_boot[b, :] = metrics_proba_b
    if verbose:
        print()
        print()
    return metrics_clf_boot, metrics_proba_boot


def boot_train_test_feature(X: np.ndarray, y: np.ndarray, indx: int, metrics_proba, metrics_clf: list = [],
                            B: int = 1000, random_state: int = 42, verbose: bool = True) -> (np.ndarray, np.ndarray):
    """
    Runs bootstrapped train-test split for evaluating the predictive power of a single feature.

    :param X: numpy array with features.
    :param y: numpy array with labels.
    :param indx: index of the considered feature.
    :param metrics_proba: list of functions used as performance metrics for probabilistic predictions.
    :param metrics_clf: list of functions used as performance metrics.
    :param B: number of bootstrap resamples.
    :param random_state: random generator seed.
    :param verbose: flag identifying whether to enable printouts.
    :return: returns arrays with classification and probabilistic performance metrics.
    """
    np.random.seed(random_state)
    metrics_clf_boot = np.zeros((B, len(metrics_clf)))
    metrics_proba_boot = np.zeros((B, len(metrics_proba)))
    for b in range(B):
        metrics_clf_b = np.zeros((len(metrics_clf),))
        metrics_proba_b = np.zeros((len(metrics_proba),))
        inds = np.arange(0, X.shape[0])
        np.random.shuffle(inds)
        inds_b = np.random.choice(a=inds, size=(inds.shape[0],), replace=True)
        inds_train, inds_test = train_test_split(inds, test_size=0.20, stratify=y)
        inds_b_test = np.array([i for i in inds_b if (i in inds_test)])
        X_test_b = X[inds_b_test, :]
        y_test_b = y[inds_b_test]
        y_pred = None
        y_pred_proba = X_test_b[:, indx]

        for j in range(len(metrics_clf)):
            metrics_clf_b[j] = metrics_clf[j](y_true=y_test_b, y_pred=y_pred)
        for j in range(len(metrics_proba)):
            metrics_proba_b[j] = metrics_proba[j](y_true=y_test_b, y_score=y_pred_proba)

        if verbose:
            print(str(b) + ". " + str(metrics_clf_b) + "  " + str(metrics_proba_b))
        metrics_clf_boot[b, :] = metrics_clf_b
        metrics_proba_boot[b, :] = metrics_proba_b
    if verbose:
        print()
        print()
    return metrics_clf_boot, metrics_proba_boot


def boot_feature_selection(X: np.ndarray, y: np.ndarray, metric, p_max: int, B: int = 500, alpha: float = 0.10,
                           random_state: int = 42,
                           verbose: bool = True) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Runs bootstrapped feature selection using XGBoost variable importance measure.

    :param X: numpy array with features.
    :param y: numpy array with labels.
    :param metric: performance metric for evaluating probabilistic predictions.
    :param p_max: maximum number of predictor variables to be selected.
    :param B: number of bootstrap resamples.
    :param alpha: significance level for constructing empirical confidence intervals.
    :param random_state: random generator seed.
    :param verbose: flag identifying whether to enable printouts.
    :return: returns arrays with average bootstrapped performance metrics, lower and upper confidence bounds, and
    frequencies of selection for features.
    """
    np.random.seed(random_state)
    perf_mean = np.zeros((p_max, ))
    perf_lower = np.zeros((p_max, ))
    perf_upper = np.zeros((p_max, ))
    selec_freqs = np.zeros((p_max, X.shape[1]))
    for p in range(1, p_max + 1):
        perf_boot = np.zeros((B, ))
        for b in range(B):
            inds = np.arange(0, X.shape[0])
            np.random.shuffle(inds)
            inds_b = np.random.choice(a=inds, size=(inds.shape[0],), replace=True)
            inds_train, inds_test = train_test_split(inds, test_size=0.20, stratify=y)
            inds_b_train = np.array([i for i in inds_b if (i in inds_train)])
            inds_b_test = np.array([i for i in inds_b if (i in inds_test)])
            X_train_b = X[inds_b_train, :]
            X_test_b = X[inds_b_test, :]
            y_train_b = y[inds_b_train]
            y_test_b = y[inds_b_test]
            importances_b = np.array([metric(y_true=y_train_b, y_score=X_train_b[:, i])
                                      for i in range(X_train_b.shape[1])])
            chosen_features = np.argsort(-importances_b)[:p]

            selec_freqs[p - 1, chosen_features] = selec_freqs[p - 1, chosen_features] + 1

            X_train_b = X_train_b[:, chosen_features]
            X_test_b = X_test_b[:, chosen_features]

            if len(np.unique(y_train_b)) == 2:
                scale_weight = (X_train_b.shape[0] - np.sum(y_train_b)) / np.sum(y_train_b)
                gbm = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False,
                                        scale_pos_weight=scale_weight)
            else:
                gbm = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
            gbm.fit(X_train_b, y_train_b)

            y_pred = gbm.predict(X_test_b)
            y_pred_proba = gbm.predict_proba(X_test_b)[:, 1]

            perf_boot[b] = metric(y_true=y_test_b, y_score=y_pred_proba)

        perf_mean[p - 1] = np.mean(perf_boot)
        perf_lower[p - 1] = np.quantile(perf_boot, q=alpha / 2)
        perf_upper[p - 1] = np.quantile(perf_boot, q=1 - alpha / 2)

        if verbose:
            print("p = " + str(p) + ": ")
            print_perf_statistics(np.expand_dims(perf_boot, 1), alpha=alpha)

    selec_freqs = selec_freqs / B

    if verbose:
        print()
        print()

    return perf_mean, perf_lower, perf_upper, selec_freqs


def boot_feature_selection_LR(X: np.ndarray, y: np.ndarray, metric, p_max: int, B: int = 500, alpha: float = 0.10,
                              random_state: int = 42, verbose: bool = True):
    """
    Runs bootstrapped feature selection for the logistic regression model.

    :param X: numpy array with features.
    :param y: numpy array with labels.
    :param metric: performance metric for evaluating probabilistic predictions.
    :param p_max: maximum number of predictor variables to be selected.
    :param B: number of bootstrap resamples.
    :param alpha: significance level for constructing empirical confidence intervals.
    :param random_state: random generator seed.
    :param verbose: flag identifying whether to enable printouts.
    :return: returns arrays with average bootstrapped performance metrics, lower and upper confidence bounds, and
    frequencies of selection for features.
    """
    np.random.seed(random_state)
    perf_mean = np.zeros((p_max, ))
    perf_lower = np.zeros((p_max, ))
    perf_upper = np.zeros((p_max, ))
    for p in range(1, p_max + 1):
        perf_boot = np.zeros((B, ))
        for b in range(B):
            inds = np.arange(0, X.shape[0])
            np.random.shuffle(inds)
            inds_b = np.random.choice(a=inds, size=(inds.shape[0],), replace=True)
            inds_train, inds_test = train_test_split(inds, test_size=0.20, stratify=y)
            inds_b_train = np.array([i for i in inds_b if (i in inds_train)])
            inds_b_test = np.array([i for i in inds_b if (i in inds_test)])
            X_train_b = X[inds_b_train, :]
            X_test_b = X[inds_b_test, :]
            y_train_b = y[inds_b_train]
            y_test_b = y[inds_b_test]
            importances_b = np.array([metric(y_true=y_train_b, y_score=X_train_b[:, i])
                                      for i in range(X_train_b.shape[1])])
            chosen_features = np.argsort(-importances_b)[:p]

            X_train_b = X_train_b[:, chosen_features]
            X_test_b = X_test_b[:, chosen_features]

            lr = LogisticRegression(class_weight='balanced')
            lr.fit(X_train_b, y_train_b)

            y_pred = lr.predict(X_test_b)
            y_pred_proba = lr.predict_proba(X_test_b)[:, 1]

            perf_boot[b] = metric(y_true=y_test_b, y_score=y_pred_proba)

        perf_mean[p - 1] = np.mean(perf_boot)
        perf_lower[p - 1] = np.quantile(perf_boot, q=alpha / 2)
        perf_upper[p - 1] = np.quantile(perf_boot, q=1 - alpha / 2)

        if verbose:
            print("p = " + str(p) + ": ")
            print_perf_statistics(np.expand_dims(perf_boot, 1), alpha=alpha)

    if verbose:
        print()
        print()

    return perf_mean, perf_lower, perf_upper


def boot_train_test_XGB_trt_vs_ab(trt: int, ab: str) -> None:
    """
    Runs the bootstrapped train-test split procedure for XGBoost to predict the specified antibody response from
    the specified treatment.

    :param trt: treatment.
    :param ab: antibody type.
    :return: None.
    """
    
    np.random.seed(42)
    
    # Load raw data
    # NB: low_memory=False is needed to deal with mixed type variables
    df = read_csv(filepath_or_buffer="CoV-ETH_190721_to_JV_190721.csv", low_memory=False)
    # Inclusion criteria: 
    included = np.logical_or(df['Proband_in_miltenyi_positive_negative_consolidated'].values == 'positive', 
                             df['Proband_in_miltenyi_positive_negative_consolidated'].values == 'negative')
    df_included = df.loc[np.squeeze(np.argwhere(included)),]
    
    # Define consolidated antibody response
    consolidated_response_S10 = np.logical_and(
        df_included['RB50_IgG_S10'].values >= 50, np.logical_or(
            df_included['NP50_IgG_S10'].values >= 5.0, np.logical_or(df_included['S150_IgG_S10'].values >= 20, 
                                                                     df_included['S250_IgG_S10'].values >= 5.0)))
    consolidated_response_S11 = np.logical_and(
        df_included['RB50_IgG_S11'].values >= 50, np.logical_or(
            df_included['NP50_IgG_S11'].values >= 5.0, np.logical_or(df_included['S150_IgG_S11'].values >= 20, 
                                                                     df_included['S250_IgG_S11'].values >= 5.0)))
    consolidated_response_ = np.logical_or(consolidated_response_S10, consolidated_response_S11)

    RBn_only_S10 = np.logical_and(
        df_included['RB50_IgG_S10'].values >= 50, np.logical_not(np.logical_or(
            df_included['NP50_IgG_S10'].values >= 5.0, np.logical_or(df_included['S150_IgG_S10'].values >=20, 
                                                                     df_included['S250_IgG_S10'].values >= 5.0))))
    RBn_only_S11 = np.logical_and(
        df_included['RB50_IgG_S11'].values >= 50, np.logical_not(np.logical_or(
            df_included['NP50_IgG_S11'].values >= 5.0, np.logical_or(df_included['S150_IgG_S11'].values >= 20, 
                                                                     df_included['S250_IgG_S11'].values >= 5.0))))
    RBn_only = np.logical_and(RBn_only_S10, RBn_only_S11)
    wo_blocking = df_included['BoB_all_IC50_consolidated_S10'].values >= 10

    consolidated_response = np.copy(consolidated_response_).astype('U32')
    consolidated_response[consolidated_response_] = 'positive'
    consolidated_response[np.logical_not(consolidated_response_)] = 'negative'
    consolidated_response[RBn_only] = 'borderline'
    
    # Define treatment
    trt_suffix = ''
    if trt == 2:
        trt_suffix = '_cov2mix'
    elif trt == 3:
        trt_suffix = '_protn'
    elif trt == 4:
        trt_suffix = '_prots1'
    elif trt == 5:
        trt_suffix = '_prots'
    elif trt == 6:
        trt_suffix = '_protm'
    elif trt == -1:
        trt_suffix = '_rand'
    else:
        NotImplementedError('ERROR: wrong treatment!')
    
    # Define antibody response
    ab_suffix = ''
    if ab == 'cons':
        ab_suffix = '_cons'
        target_response = consolidated_response
    elif ab == 'rb50':
        ab_suffix = '_rb50'
        # RB50 response 
        rb50_response_s10 = df_included['Proband_in_miltenyi_RB50_S10_consolidated'].values
        rb50_response_s11 = df_included['Proband_in_miltenyi_RB50_S11_consolidated'].values
        rb50_response = np.copy(rb50_response_s10)
        rb50_response[:] = "negative"
        rb50_response[np.logical_or(rb50_response_s10 == 'positive', 
                                    rb50_response_s11 == 'positive')] = 'positive'
        rb50_borderline_cases = np.logical_and(rb50_response_s10 == 'borderline', 
                                               rb50_response_s11 == 'borderline')
        target_response = rb50_response
    elif ab == 'np50':
        ab_suffix = '_np50'
        # NP50 response 
        np50_response_s10 = df_included['Proband_in_miltenyi_NP50_S10_consolidated'].values
        np50_response_s11 = df_included['Proband_in_miltenyi_NP50_S11_consolidated'].values
        np50_response = np.copy(np50_response_s10)
        np50_response[:] = "negative"
        np50_response[np.logical_or(np50_response_s10 == 'positive', 
                                    np50_response_s11 == 'positive')] = 'positive'
        np50_borderline_cases = np.logical_and(np50_response_s10 == 'borderline', 
                                               np50_response_s11 == 'borderline')
        target_response = np50_response
    elif ab == 's150':
        ab_suffix = '_s150'
        # S150 response 
        s150_response_s10 = df_included['Proband_in_miltenyi_S150_S10_consolidated'].values
        s150_response_s11 = df_included['Proband_in_miltenyi_S150_S11_consolidated'].values
        s150_response = np.copy(s150_response_s10)
        s150_response[:] = "negative"
        s150_response[np.logical_or(s150_response_s10 == 'positive', 
                                    s150_response_s11 == 'positive')] = 'positive'
        s150_borderline_cases = np.logical_and(s150_response_s10 == 'borderline', 
                                               s150_response_s11 == 'borderline')
        
        target_response = s150_response
    elif ab == 's250':
        ab_suffix = '_s250'
        # S250 response 
        s250_response_s10 = df_included['Proband_in_miltenyi_S250_S10_consolidated'].values
        s250_response_s11 = df_included['Proband_in_miltenyi_S250_S11_consolidated'].values
        s250_response = np.copy(s250_response_s10)
        s250_response[:] = "negative"
        s250_response[np.logical_or(s250_response_s10 == 'positive', 
                                    s250_response_s11 == 'positive')] = 'positive'
        s250_borderline_cases = np.logical_and(s250_response_s10 == 'borderline', 
                                               s250_response_s11 == 'borderline')
        target_response = s250_response
    elif ab == 'nab':
        ab_suffix = '_nab'
        # nAB response 
        nab_response_s10 = df_included['Proband_in_miltenyi_nAB_positive_negative_consolidated'].values
        nab_response_s11 = df_included['Proband_in_miltenyi_nAB_positive_negative_consolidated'].values
        nab_response = np.copy(nab_response_s10)
        nab_response[:] = "negative"
        nab_response[np.logical_or(nab_response_s10 == 'positive', 
                                   nab_response_s11 == 'positive')] = 'positive'
        nab_borderline_cases = np.logical_and(nab_response_s10 == 'borderline', 
                                              nab_response_s11 == 'borderline')
        target_response = nab_response
    else:
        NotImplementedError('ERROR: wrong antibody type!')
    
    if trt != -1:
        # Run bootstrapping for XGBoost
        if os.path.isfile('results/metrics_clf' + ab_suffix + trt_suffix + '.csv') and \
           os.path.isfile('results/metrics_proba' + ab_suffix + trt_suffix + '.csv') and \
           os.path.isfile('results/feat_imps' + ab_suffix + trt_suffix + '.csv'):
            metrics_clf = np.loadtxt(fname='results/metrics_clf' + ab_suffix + trt_suffix + '.csv', delimiter=",")
            metrics_proba = np.loadtxt(fname='results/metrics_proba' + ab_suffix + trt_suffix + '.csv', delimiter=",")
            feature_importances = np.loadtxt(fname='results/feat_imps' + ab_suffix + trt_suffix + '.csv', 
                                             delimiter=",")
        else:
            df_tcell, feature_names = construct_tcell_df(df_included, feature_type='perc', proteins=[trt])
            disease_status = ((target_response == "positive") * 1.0).astype(int)
            metrics_clf, metrics_proba, feature_importances = boot_train_test_XGB(
                X=df_tcell, y=disease_status, metrics_clf=[balanced_accuracy_score, sensitivity, specificity], 
                metrics_proba=[roc_auc_score, average_precision_score], B=1000, borderline_cases=RBn_only, 
                drop_borderline=True)
            np.savetxt(fname='results/metrics_clf' + ab_suffix + trt_suffix + '.csv', X=metrics_clf, delimiter=",")
            np.savetxt(fname='results/metrics_proba' + ab_suffix + trt_suffix + '.csv', X=metrics_proba, delimiter=",")
            np.savetxt(fname='results/feat_imps' + ab_suffix + trt_suffix + '.csv', X=feature_importances, 
                       delimiter=",")
    else:
        # Run bootstrapping for random guess
        if os.path.isfile('results/metrics_clf' + ab_suffix + trt_suffix + '.csv') and \
           os.path.isfile('results/metrics_proba' + ab_suffix + trt_suffix + '.csv'):
            metrics_clf = np.loadtxt(fname='results/metrics_clf' + ab_suffix + trt_suffix + '.csv', delimiter=",")
            metrics_proba = np.loadtxt(fname='results/metrics_proba' + ab_suffix + trt_suffix + '.csv', delimiter=",")
        else:
            df_tcell, feature_names = construct_tcell_df(df_included, feature_type='perc')
            disease_status = ((target_response == "positive") * 1.0).astype(int)
            metrics_clf, metrics_proba = boot_train_test_random(
                X=df_tcell, y=disease_status, metrics_clf=[balanced_accuracy_score, sensitivity, specificity], 
                metrics_proba=[roc_auc_score, average_precision_score], B=10000)
            np.savetxt(fname='results/metrics_clf' + ab_suffix + trt_suffix + '.csv', X=metrics_clf, delimiter=",")
            np.savetxt(fname='results/metrics_proba' + ab_suffix + trt_suffix + '.csv', X=metrics_proba, delimiter=",")
