import pandas as pd
import numpy as np
import random
from typing import Union, Dict, Any, Type, List, Tuple
import warnings
import torch.nn as nn
import torch
import copy
import torch.optim as optim
import torch.nn.functional as F
import math
import re
import matplotlib.pyplot as plt
import os
import datetime
from scipy.stats import f
from statsmodels.stats.multitest import multipletests
mouse_acrophases = [
    0, 0.0790637050481884, 
    0.151440116812406, 
    2.29555301890004, 
    2.90900605826091, 
    2.98706493493206, 
    2.99149022777511, 
    3.00769248308471, 
    3.1219769314524, 
    3.3058682224604, 
    3.31357155959037, 
    3.42557704861225, 
    3.50078722833753, 
    3.88658015146741, 
    4.99480367551318, 
    5.04951134876313, 
    6.00770260397838
]
mouse_gene_symbol = [
    "Arntl", 
    "Clock", 
    "Npas2", 
    "Nr1d1", 
    "Bhlhe41", 
    "Nr1d2", 
    "Dbp", 
    "Ciart", 
    "Per1", 
    "Per3", 
    "Tef", 
    "Hlf", 
    "Cry2", 
    "Per2", 
    "Cry1", 
    "Rorc", 
    "Nfil3"
]
human_homologue_gene_symbol = [gene.upper() for gene in mouse_gene_symbol]
subfolders = ["Plots", "Fits", "Models", "Parameters"]

def my_info(message: str):
    print(f"INFO: {message}")
def my_warn(message: str):
    warnings.warn(message)
def my_error(message: str):
    raise ValueError(message)
def findXinY(search_x: List[Any], in_y: List[Any]) -> List[int]:
    indices = []
    for z in search_x:
        for i, val_y in enumerate(in_y):
            if val_y == z:
                indices.append(i)
    return indices
def my_mse_Loss(predictions: torch.Tensor, targets: torch.Tensor, dim: int):
    loss = F.mse_loss(predictions, targets, reduction='none')
    loss = loss.mean(dim=dim) if dim is not None else loss.mean()
    return loss
def DefaultDict():
    theDefaultDictionary = {
        "regex_cont": r".*_C",
        "regex_disc": r".*_D",

        "blunt_percent": 0.975,

        "seed_min_CV": 0.14,
        "seed_max_CV": 0.7,
        "seed_mth_Gene": 10000,

        "norm_gene_level": True,
        "norm_disc": False,
        "norm_disc_cov": 1,

        "eigen_reg": True,
        "eigen_reg_disc_cov": 1,
        "eigen_reg_exclude": False,
        "eigen_reg_r_squared_cutoff": 0.6,
        "eigen_reg_remove_correct": False,

        "eigen_first_var": False,
        "eigen_first_var_cutoff": 0.85,

        "eigen_total_var": 0.85,
        "eigen_contr_var": 0.06,
        "eigen_var_override": False,
        "eigen_max": 5,

        "out_covariates": True,
        "out_use_disc_cov": True,
        "out_all_disc_cov": True,
        "out_disc_cov": 1,
        "out_use_cont_cov": False,
        "out_all_cont_cov": True,
        "out_use_norm_cont_cov": False,
        "out_all_norm_cont_cov": True,
        "out_cont_cov": 1,
        "out_norm_cont_cov": 1,

        "init_scale_change": True,
        "init_scale_1": False,

        "train_n_models": 80,
        "train_μA": 0.001,
        "train_β": (0.9, 0.999),
        "train_min_steps": 1500,
        "train_max_steps": 2050,
        "train_μA_scale_lim": 1000,
        "train_circular": False,
        "train_collection_times": True,
        "train_collection_time_balance": 0.1,

        "cosine_shift_iterations": 192,
        "cosine_covariate_offset": True,

        "align_p_cutoff": 0.05,
        "align_base": "radians",
        "align_disc": False,
        "align_disc_cov": 1,
        "align_other_covariates": False,
        "align_batch_only": False,
        "align_samples": [],
        "align_phases": [],

        "X_Val_k": 10,
        "X_Val_omit_size": 0.1,

        "plot_use_o_cov": True,
        "plot_correct_batches": False,
        "plot_disc": False,
        "plot_disc_cov": 1,
        "plot_separate": False,
        "plot_color": ["b", "orange", "g", "r", "m", "y", "k"],
        "plot_only_color": True,
        "plot_p_cutoff": 0.05
    }
    return theDefaultDictionary

def update_DefaultDict(alternateDict: Dict[str, Any], display_changes: bool = False) -> Dict[str, Any]:
    theDefaultDictionary = DefaultDict()
    for change_key, change_value in alternateDict.items():
        if change_key in theDefaultDictionary:
            if theDefaultDictionary[change_key] != change_value:
                show_change_made = f" = {theDefaultDictionary[change_key]} for {change_value}" if display_changes else ""
                print("")
                my_warn(f"Replacing existing key value: {change_key}{show_change_made}.")
                print("")
        else:
            additional_keys = ["align_genes", "align_acrophases", "align_samples", "align_phases", "train_sample_id", "train_sample_phase"]
            if change_key not in additional_keys:
                print("")
                my_error(f"{change_key} IS NOT A KEY KNOWN TO CYCLOPS. PLEASE REVISE OR REMOVE KEYS GIVEN IN alternateOps BEFORE PROCEEDING.")
                print("")
            show_change_made = f" => {change_value}" if display_changes else ""
            print("")
            my_warn(f"Adding :{change_key}{show_change_made}. Delete this key if you do not intend to use it for alignment.")
            print("")
        theDefaultDictionary[change_key] = change_value
    return theDefaultDictionary

def FindRegexinArray(search_x: str, in_y: List[str]) -> List[int]:
    indices = []
    for i, x in enumerate(in_y):
        if re.search(search_x, x):
            indices.append(i)
    return indices

def MakeFloat(ar: Union[pd.DataFrame, np.ndarray], OUT_TYPE: Type = float) -> np.ndarray:
    if isinstance(ar, pd.DataFrame):
        ar_np = ar.values
    else:
        ar_np = ar

    vec_parse = np.vectorize(lambda x: OUT_TYPE(x) if isinstance(x, (str, np.str_)) else x)
    result_array = vec_parse(ar_np).astype(OUT_TYPE)
    
    return result_array

def onehotbatch(labels, unique_labels):
    num_classes = len(unique_labels)
    label_to_idx = {v: i for i, v in enumerate(unique_labels)}
    indices = np.array([label_to_idx[label] for label in labels], dtype=int)
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), indices] = 1
    return one_hot.T

def CovariateProcessing(dataFile: pd.DataFrame, options: Dict[str, Any]) -> Dict[str, Any]:
    my_info("COLLECTING COVARIATE INFORMATION.")

    options["o_column_ids"] = dataFile.columns[1:].tolist()
    geneColumn = dataFile.iloc[:, 0].tolist()

    contCovIndx = FindRegexinArray(options["regex_cont"], geneColumn)
    
    cont_cov_row: Union[pd.DataFrame, Any] = None
    cont_min_max: Union[List[np.ndarray], Any] = None
    norm_cov_row: Union[pd.DataFrame, Any] = None

    if len(contCovIndx) > 0:
        print_statement = f"{len(contCovIndx)} CONTINUOUS COVARIATES." if len(contCovIndx) > 1 else "1 CONTINUOUS COVARIATE."
        my_info(print_statement)

        cont_cov_row = MakeFloat(dataFile.iloc[contCovIndx, 1:], np.float64)
        rowMinima = cont_cov_row.min(axis=1).values.reshape(-1, 1)
        rowMaxima = cont_cov_row.max(axis=1).values.reshape(-1, 1)
        cont_min_max = [rowMinima, rowMaxima]
        norm_cov_row = (cont_cov_row - rowMinima) / (rowMaxima - rowMinima) + 1
    
    discCovIndx = FindRegexinArray(options["regex_disc"], geneColumn)
    
    discCov: Union[np.ndarray, Any] = None
    disc_cov_labels: Union[List[List[Any]], Any] = None
    disc_full_onehot: Union[List[np.ndarray], Any] = None
    onehot_redundance_removed: Union[List[np.ndarray], Any] = None
    onehot_redundance_removed_transpose: Union[List[np.ndarray], Any] = None

    if len(discCovIndx) > 0:
        print_statement = f"{len(discCovIndx)} DISCONTINUOUS COVARIATES." if len(discCovIndx) > 1 else "1 DISCONTINUOUS COVARIATE."
        my_info(print_statement)

        discCov = dataFile.iloc[discCovIndx, 1:].values
        disc_cov_labels = [np.unique(discCov[i, :]).tolist() for i in range(discCov.shape[0])]
        disc_full_onehot = [
            onehotbatch(discCov[i, :], disc_cov_labels[i])
            for i in range(discCov.shape[0])
        ]
        onehot_redundance_removed = [x[1:, :] for x in disc_full_onehot]
        onehot_redundance_removed_transpose = [x.T for x in onehot_redundance_removed]

    n_cov = len(contCovIndx) + len(discCovIndx)
    my_info(f"{n_cov} TOTAL COVARIATES.")

    all_cov_indices = contCovIndx + discCovIndx
    first_expression_row = 0
    if all_cov_indices:
        first_expression_row = max(all_cov_indices) + 1
    
    dictionary_keys_short = ("o_fxr", "o_dcl", "o_dc", "o_dco", "o_dcorr", "o_dcorrt", "o_cc", "o_ncc", "o_ccmm")
    dictionary_values = (
        first_expression_row,
        disc_cov_labels,
        discCov,
        disc_full_onehot,
        onehot_redundance_removed,
        onehot_redundance_removed_transpose,
        cont_cov_row,
        norm_cov_row,
        cont_min_max
    )
    for new_key, new_value in zip(dictionary_keys_short, dictionary_values):
        options[new_key] = new_value
    
    return options

def BluntXthPercentile(
    dataFile: Union[pd.DataFrame, np.ndarray],
    options: Dict[str, Any],
    subset_logical: np.ndarray,
    OUT_TYPE: Type = pd.DataFrame
) -> Union[pd.DataFrame, np.ndarray]:
    
    my_info("BLUNTING OUTLIERS IN DATASET.")

    data = dataFile

    if isinstance(dataFile, pd.DataFrame):
        data_numeric_part = MakeFloat(dataFile.iloc[options["o_fxr"]:, 1:]).copy()
        data = data_numeric_part[:, subset_logical]

    ngene, nsamples = data.shape

    nfloor = int(1 + np.floor((1 - options["blunt_percent"]) * nsamples))
    nceiling = int(np.ceil(options["blunt_percent"] * nsamples))

    sorted_data = np.sort(data, axis=1)

    row_min = sorted_data[:, nfloor - 1]
    row_max = sorted_data[:, nceiling - 1]

    too_small = data < row_min[:, None]
    too_large = data > row_max[:, None]

    for ii in range(ngene):
        data[ii, too_small[ii, :]] = row_min[ii]
        data[ii, too_large[ii, :]] = row_max[ii]

    if OUT_TYPE == pd.DataFrame:
        first_col_for_blunted_data = dataFile.iloc[options["o_fxr"]:, 0].values.reshape(-1, 1)
        header_df = dataFile.iloc[:options["o_fxr"], :].copy()
        header_cols_to_keep_indices = [0] + (np.where(subset_logical)[0] + 1).tolist()
        header_df_subset = header_df.iloc[:, header_cols_to_keep_indices]
        blunted_df = pd.DataFrame(data, columns=dataFile.columns[1:][subset_logical])
        blunted_df.insert(0, dataFile.columns[0], dataFile.iloc[options["o_fxr"]:, 0].values)
        bluntedDataFile = pd.concat([header_df_subset, blunted_df], axis=0, ignore_index=True)
        bluntedDataFile.columns = dataFile.columns[[True] + subset_logical.tolist()].tolist()
        return bluntedDataFile

    output_data = data.astype(OUT_TYPE)
    return output_data

def GetBluntXthPercentile(
    dataFile: Union[pd.DataFrame, np.ndarray], 
    ops: Dict[str, Any], 
    OUT_TYPE: Type = pd.DataFrame
) -> Union[pd.DataFrame, np.ndarray]:
    output_data: Union[pd.DataFrame, np.ndarray]

    if ops["norm_gene_level"]:
        if ops["norm_disc"] and ("o_dcl" in ops and ops["o_dcl"] is not None):
            n_cov = len(ops["norm_disc_cov"])
            if n_cov > 1:
                my_error(f"Only a single covariate can be used. Please specify one, you have specified {n_cov}: {ops['norm_disc_cov']}.")

            norm_disc_cov_idx = ops["norm_disc_cov"] - 1 if isinstance(ops["norm_disc_cov"], int) else [x - 1 for x in ops["norm_disc_cov"]]

            normCovOnehot = ops["o_dco"][norm_disc_cov_idx]
            
            subset_logical_list = [normCovOnehot[ll, :].astype(bool) for ll in range(normCovOnehot.shape[0])]
            
            output_data_parts = [
                BluntXthPercentile(dataFile, ops, subset_logical, OUT_TYPE=OUT_TYPE)
                for subset_logical in subset_logical_list
            ]
            
            if OUT_TYPE == pd.DataFrame:
                gene_symbols_col = output_data_parts[0].iloc[:, 0]
                
                numeric_data_parts = [part.iloc[:, 1:] for part in output_data_parts]
                output_data = pd.concat(numeric_data_parts, axis=1)

                cols = pd.Series(output_data.columns)
                for dup in cols[cols.duplicated()].unique():
                    cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(len(cols[cols == dup]))]
                output_data.columns = cols
                
                output_data.insert(0, dataFile.columns[0], gene_symbols_col)

                columns_to_keep_mask = ~output_data.columns.str.contains(r"Gene_Symbols_[0-9]{1,3}", regex=True)
                output_data = output_data.loc[:, columns_to_keep_mask]

            else:
                output_data = np.hstack(output_data_parts)
                pass
        else:
            subset_logical = np.full(dataFile.shape[1] - 1, True, dtype=bool)
            output_data = BluntXthPercentile(dataFile, ops, subset_logical, OUT_TYPE=OUT_TYPE)
    else:
        subset_logical = np.full(dataFile.shape[1] - 1, True, dtype=bool)
        output_data = BluntXthPercentile(dataFile, ops, subset_logical, OUT_TYPE=OUT_TYPE)

    return output_data

def GeneLevelCutoffs(
    dataFile: pd.DataFrame, 
    genesOfInterest: List[str], 
    options: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[int]]:
    
    geneColumnNoCov = dataFile.iloc[options["o_fxr"]:, 0].tolist()
    expressionData = MakeFloat(dataFile.iloc[options["o_fxr"]:, 1:], np.float64)
    
    mthGeneMeanCutoff = np.sort(expressionData.mean(axis=1))[-(options["seed_mth_Gene"] + 1)]
    genesOfInterestIndices = findXinY(genesOfInterest, geneColumnNoCov)

    genesOfInterestExpressionData = expressionData[genesOfInterestIndices, :]
    
    genesOfInterestMeans = genesOfInterestExpressionData.mean(axis=1).reshape(-1, 1)
    genesOfInterestStd = genesOfInterestExpressionData.std(axis=1).reshape(-1, 1)
    genesOfInterestCVs = genesOfInterestStd / genesOfInterestMeans
    
    cvBelowCutoff = np.where(genesOfInterestCVs[:, 0] < options["seed_max_CV"])[0]
    cvAboveCutoff = np.where(genesOfInterestCVs[:, 0] > options["seed_min_CV"])[0]
    meanBelowCutoff = np.where(genesOfInterestMeans[:, 0] > mthGeneMeanCutoff)[0]
    
    common_indices_sets = set(cvAboveCutoff).intersection(set(cvBelowCutoff), set(meanBelowCutoff))
    
    genesOfInterestIndicesToKeep_relative = sorted(list(common_indices_sets))
    
    genesOfInterestIndicesToKeep = [genesOfInterestIndices[i] for i in genesOfInterestIndicesToKeep_relative]
    
    output_genes_of_interest_expression_data = expressionData[genesOfInterestIndicesToKeep, :]
    
    options["o_seed_genes"] = [geneColumnNoCov[i] for i in genesOfInterestIndicesToKeep_relative]
    
    return output_genes_of_interest_expression_data, genesOfInterestIndicesToKeep

def MeanNormalize(data: np.ndarray, OUT_TYPE: Type = float) -> np.ndarray:
    gene_means = np.mean(data, axis=1, keepdims=True)
    norm_data = (data - gene_means) / gene_means
    output_data = norm_data.astype(OUT_TYPE)
    return output_data

def GetMeanNormalizedData(
    keepGenesOfInterestExpressionData: Union[pd.DataFrame, np.ndarray], 
    ops: Dict[str, Any]
) -> Union[pd.DataFrame, np.ndarray]:
    
    my_info("PERFORMING DISPERSION NORMALIZATION.")
    
    meanNormalizedData = keepGenesOfInterestExpressionData
    
    if ops["norm_gene_level"]:
        meanNormalizedData = MeanNormalize(keepGenesOfInterestExpressionData)
        
        if ops["norm_disc"] and ("o_dcl" in ops and ops["o_dcl"] is not None):
            n_cov = len(ops["norm_disc_cov"])
            if n_cov > 1:
                my_error(f"Only a single covariate can be used. Please specify one, you have specified {n_cov}: {ops['norm_disc_cov']}.")
            
            norm_disc_cov_idx = ops["norm_disc_cov"] - 1 
            normCovOnehot = ops["o_dco"][norm_disc_cov_idx]

            dataByCovariates = []
            for ll in range(normCovOnehot.shape[0]):
                mask = normCovOnehot[ll, :].astype(bool)
                if isinstance(keepGenesOfInterestExpressionData, pd.DataFrame):
                    dataByCovariates.append(keepGenesOfInterestExpressionData.loc[:, mask])
                else:
                    dataByCovariates.append(keepGenesOfInterestExpressionData[:, mask])
            
            normalized_parts = [MeanNormalize(data_part) for data_part in dataByCovariates]
            
            if isinstance(keepGenesOfInterestExpressionData, pd.DataFrame):
                meanNormalizedData = pd.concat(normalized_parts, axis=1)
            else:
                meanNormalizedData = np.hstack(normalized_parts)

    return meanNormalizedData

def SVDTransform(
    expression_data: Union[pd.DataFrame, np.ndarray], 
    ops: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    
    my_info("TRANSFORMING SEED GENES INTO EIGEN SPACE.")

    if isinstance(expression_data, pd.DataFrame):
        data_matrix = expression_data.values
    else:
        data_matrix = expression_data

    U_svd, s_svd, Vh_svd = np.linalg.svd(data_matrix, full_matrices=False)

    V = Vh_svd.T.astype(np.float64)
    Vt = Vh_svd.astype(np.float64)
    S = s_svd.astype(np.float32)

    ops["o_svd_S"] = S
    ops["o_svd_U"] = U_svd.astype(np.float32)
    ops["o_svd_V"] = V.astype(np.float32)

    s_squared = S**2
    total_variance = np.sum(s_squared)
    cumvar = np.cumsum(s_squared) / total_variance

    dimvar = np.concatenate(([cumvar[0]], np.diff(cumvar)))

    ops["o_svd_cumvar"] = cumvar
    ops["o_svd_dimvar"] = dimvar

    svd_obj_l = (U_svd, s_svd, Vh_svd)

    return V, Vt, S, cumvar, svd_obj_l

def SVDBatchRegression(V: np.ndarray, Vt: np.ndarray, ops: Dict[str, Any]):
    
    if ops["eigen_reg"] and ("o_dcl" in ops and ops["o_dcl"] is not None):
        my_info("PERFORMING REGRESSION AGAINST DISCONTINUOUS COVARIATES.")
        
        n_cov = (ops["eigen_reg_disc_cov"])
        if n_cov > 1:
            my_error(f"Only a single covariate can be used. Please specify one, you have provided an array with length {n_cov}.")
        
        reg_cov_idx = ops["eigen_reg_disc_cov"] - 1
        regCovT = ops["o_dcorrt"][reg_cov_idx]
        regCov = ops["o_dcorr"][reg_cov_idx]

        intercept_column = np.ones((regCovT.shape[0], 1), dtype=np.float64)
        regCovT_tmp = np.hstack((regCovT, intercept_column))
        Linear_regression_models = np.linalg.lstsq(regCovT_tmp.astype(np.float64), V, rcond=None)
        Linear_regression_models = Linear_regression_models[0]
        print("LINEAR REGRESSION MODELS SHAPE:", Linear_regression_models.shape)
        SSE_total = np.sum((Vt - np.mean(Vt, axis=1, keepdims=True))**2, axis=1)

        predicted_values =  Linear_regression_models[:regCovT.shape[1], :].T @ regCov + Linear_regression_models[-1, :]

        SSE_residual = np.sum((Vt - predicted_values)**2, axis=1)
        
        R_squared_values = np.where(SSE_total != 0, 1 - (SSE_residual / SSE_total), 0)
        
        ops["o_svd_Rsquared"] = R_squared_values

def SVDReduceDimensions(S: np.ndarray, S_logical: np.ndarray, ops: Dict[str, Any]) -> np.ndarray:
    if ops["eigen_max"] < 2:
        my_error(f"FATAL INPUT ERROR. :eigen_max CANNOT BE LESS THAN 2 BUT HAS BEEN DEFINED AS {ops['eigen_max']}. PLEASE INCREASE :eigen_max.")

    cumvar = np.cumsum(S**2) / np.sum(S**2)
    ops["o_svd_cumvar_corrected"] = cumvar

    vardiff = np.concatenate(([cumvar[0]], np.diff(cumvar)))
    ops["o_svd_dimvar_corrected"] = vardiff
    reduction_dim1_candidates = np.where(cumvar > ops["eigen_total_var"])[0]
    if len(reduction_dim1_candidates) > 0:
        ReductionDim1 = reduction_dim1_candidates[0] + 1
    else:
        ReductionDim1 = len(S)

    my_info(f"{ReductionDim1} DIMENSIONS REQUIRED TO CAPTURE {np.trunc(100 * ops['eigen_total_var'])}% OF THE REMAINING VARIANCE.")
    if ReductionDim1 < 2:
        my_warn(f"THE TOTAL VARIANCE LIMIT REDUCES THE # OF EIGEN GENE DIMENSIONS TO {ReductionDim1}. SETTING DIMENSION TO 2. PLEASE INCREASE :eigen_total_var TO CAPTURE MORE DIMENSIONS.")
        ReductionDim1 = 2

    reduction_dim2_candidates = np.where(vardiff > ops["eigen_contr_var"])[0]
    if len(reduction_dim2_candidates) > 0:
        ReductionDim2 = reduction_dim2_candidates[-1] + 1
    else:
        ReductionDim2 = 0

    less_than_10_or_greater_than_20 = (ReductionDim2 < 10) or (ReductionDim2 > 20)
    if less_than_10_or_greater_than_20:
        number_suffix = "th"
        if ReductionDim2 % 10 == 1 and ReductionDim2 % 100 != 11:
            number_suffix = "st"
        elif ReductionDim2 % 10 == 2 and ReductionDim2 % 100 != 12:
            number_suffix = "nd"
        elif ReductionDim2 % 10 == 3 and ReductionDim2 % 100 != 13:
            number_suffix = "rd"
    else:
        number_suffix = "th"

    my_info(f"THE {ReductionDim2}{number_suffix} DIMENSION IS THE LAST DIMENSION TO CONTRIBUTE AT LEAST {np.trunc(100 * ops['eigen_contr_var'])}% VARIANCE.")
    if ReductionDim2 < 2:
        my_warn(f"THE INDIVIDUAL VARIANCE LIMIT REDUCES THE # OF EIGEN GENE DIMENSIONS TO {ReductionDim2}. SETTING DIMENSION TO 2. PLEASE DECREASE :eigen_contr_var TO CAPTURE MORE DIMENSIONS.")
        ReductionDim2 = 2

    if ReductionDim2 < ReductionDim1:
        my_warn("CONTRIBUTED VARIANCE OF INDIVIDUAL EIGENGENES IS LIMITING THE TOTAL VARIANCE CAPTURED")
        if ops.get("eigen_var_override", False):
            my_warn(f"OVERRIDING MINIMUM CONTRIBUTED VARIANCE REQUIREMENT ({ReductionDim2} EIGENGENES) FOR MINIMUM CAPTURED VARIANCE REQUIREMENT ({ReductionDim1} EIGENGENES).\nSET :eigen_var_override TO FALSE IF YOU DO NOT WISH TO DO SO")
            ReductionDim2 = ReductionDim1

    ReductionDim = int(min(ReductionDim1, ReductionDim2, ops["eigen_max"]))

    all_S_indeces = np.where(S_logical)[0] + 1
    keep_eigen_gene_row_index = all_S_indeces[:ReductionDim]

    ops["o_svd_n_dims"] = ReductionDim
    ops["o_svd_dim_indices"] = keep_eigen_gene_row_index.tolist()

    my_info(f"KEEPING {ReductionDim} DIMENSIONS, NAMELY DIMENSIONS {', '.join(map(str, keep_eigen_gene_row_index[:-1]))} AND {keep_eigen_gene_row_index[-1]}.")
    return keep_eigen_gene_row_index

def SVDRegressionExclusion(S: np.ndarray, ops: Dict[str, Any]) -> List[int]:

    if "o_svd_Rsquared" in ops:
        R_squared = ops["o_svd_Rsquared"]
        S_logical = R_squared < ops["eigen_reg_r_squared_cutoff"]
        
        num_excluded = np.sum(~S_logical)
        singular_or_plural = num_excluded == 1
        gene_or_genes = "GENE" if singular_or_plural else "GENES"
        
        excluded_indices_1_based = np.where(~S_logical)[0] + 1
        excluded_index_list = ", ".join(map(str, excluded_indices_1_based))
        
        excluded_r_squared_values = R_squared[~S_logical]
        excluded_r_squared_list = ", ".join(map(lambda x: f"{x:.3f}", excluded_r_squared_values))
        
        has_or_have = "HAS" if singular_or_plural else "HAVE"
        respectively = "" if singular_or_plural else ", RESPECTIVELY"
        
        if ops["eigen_reg_exclude"]:
            my_info(f"REMOVING EIGEN GENES WITH R SQUARED GREATER THAN {ops['eigen_reg_r_squared_cutoff']:.3f}.")
            if num_excluded < 1:
                my_info("NO EIGEN GENES EXCLUDED BY R SQUARED.")
            else:
                my_warn(f"EXCLUDING EIGEN {gene_or_genes} {excluded_index_list}, WHICH {has_or_have} AN R SQUARED VALUE OF {excluded_r_squared_list}{respectively}.")
            Reduce_S = S[S_logical]
            S_logical_for_reduce_dims = S_logical
        elif ops["eigen_reg_remove_correct"]:
            my_info(f"EXCLUDING VARIANCE CONTRIBUTED BY EIGEN GENES WITH R SQUARED GREATER THAN {ops['eigen_reg_r_squared_cutoff']:.3f}.")
            if num_excluded < 1:
                my_info("NO VARIANCE OF EIGEN GENES EXCLUDED BY R SQUARED.")
            else:
                my_warn(f"EXCLUDING VARIANCE OF EIGEN {gene_or_genes} {excluded_index_list}, WHICH {has_or_have} AN R SQUARED VALUE OF {excluded_r_squared_list}{respectively}.")
            
            temp_S = S.copy()
            temp_S[~S_logical] = 0.0
            Reduce_S = temp_S
            S_logical_for_reduce_dims = np.full(len(S), True)
        else:
            my_info(f"REDUCING VARIANCE CONTRIBUTED BY EIGEN GENES WITH R SQUARED GREATER THAN {ops['eigen_reg_r_squared_cutoff']:.3f}.")
            if num_excluded < 1:
                my_info("NO VARIANCE OF EIGEN GENES REDUCED BY R SQUARED.")
            else:
                my_warn(f"REDUCING VARIANCE OF EIGEN {gene_or_genes} {excluded_index_list}, WHICH {has_or_have} AN R SQUARED VALUE OF {excluded_r_squared_list}{respectively}.")
            
            temp_S = S.copy()
            temp_S[~S_logical] *= (1 - R_squared[~S_logical])
            Reduce_S = temp_S
            S_logical_for_reduce_dims = np.full(len(S), True)
    else:
        Reduce_S = S
        S_logical_for_reduce_dims = np.full(len(S), True)

    dimension_indices = SVDReduceDimensions(Reduce_S, S_logical_for_reduce_dims, ops)

    return dimension_indices

def CovariateOnehotEncoder(Transform: np.ndarray, ops: Dict[str, Any]) -> np.ndarray:
    
    Transform_copy = np.copy(Transform)
    
    are_there_discontinuous_covariates = "o_dco" in ops and ops["o_dco"] is not None and len(ops["o_dco"]) > 0
    are_there_continuous_covariates = "o_cc" in ops and ops["o_cc"] is not None and ops["o_cc"].shape[0] > 0

    if ops.get("out_covariates", False) and (are_there_discontinuous_covariates or are_there_continuous_covariates):
        my_info("ADDING COVARIATES TO EIGEN GENES.")
        scale_array: List[np.ndarray] = []

        if are_there_discontinuous_covariates and ops.get("out_use_disc_cov", False):
            my_info("ADDING DISCONTINUOUS COVARIATES TO EIGEN GENES.")
            
            if ops.get("out_all_disc_cov", False) or (len(ops["o_dcl"]) == 1):
                Transform_copy = np.vstack([Transform_copy] + ops["o_dcorr"])
                
                for ii_dco in ops["o_dco"]:
                    batch_stds = []
                    for row_idx in range(ii_dco.shape[0]):
                        mask = ii_dco[row_idx, :].astype(bool)
                        if np.any(mask):
                            batch_stds.append(np.std(Transform[:, mask], axis=1, keepdims=True))
                        else:
                            batch_stds.append(np.zeros((Transform.shape[0], 1)))
                    if len(batch_stds) > 1:
                        scales = [(batch_stds[0] / x) - 1 for x in batch_stds[1:]]
                        scale_array.extend(scales)
            else:
                more_than_one_covariate = len(ops["out_disc_cov"]) > 1
                disc_cov_indices_0_based = [idx - 1 for idx in ops["out_disc_cov"]]

                if more_than_one_covariate:
                    selected_dcorr_parts = [ops["o_dcorr"][idx] for idx in disc_cov_indices_0_based]
                    Transform_copy = np.vstack([Transform_copy] + selected_dcorr_parts)
                else:
                    Transform_copy = np.vstack([Transform_copy, ops["o_dcorr"][disc_cov_indices_0_based[0]]])
                
                selected_dco_parts = [ops["o_dco"][idx] for idx in disc_cov_indices_0_based]
                
                for ii_dco in selected_dco_parts:
                    batch_stds = []
                    for row_idx in range(ii_dco.shape[0]):
                        mask = ii_dco[row_idx, :].astype(bool)
                        if np.any(mask):
                            batch_stds.append(np.std(Transform[:, mask], axis=1, keepdims=True))
                        else:
                            batch_stds.append(np.zeros((Transform.shape[0], 1)))
                    
                    if len(batch_stds) > 1:
                        scales = [(batch_stds[0] / x) - 1 for x in batch_stds[1:]]
                        scale_array.extend(scales)

        if ops.get("out_use_cont_cov", False) and are_there_continuous_covariates:
            my_info("ADDING CONTINUOUS COVARIATES TO EIGEN GENES.")
            
            if ops.get("out_all_cont_cov", False):
                if ops.get("out_use_norm_cont_cov", False):
                    my_info("ADDING NORMALIZED CONTINUOUS COVARIATES TO EIGEN GENES.")
                    if ops.get("out_all_norm_cont_cov", False):
                        Transform_copy = np.vstack((Transform_copy, ops["o_ncc"]))
                    else:
                        for cci_0_based in range(ops["o_cc"].shape[0]):
                            if (cci_0_based + 1) in ops.get("out_norm_cont_cov", []):
                                Transform_copy = np.vstack((Transform_copy, ops["o_ncc"][cci_0_based, :]))
                            else:
                                Transform_copy = np.vstack((Transform_copy, ops["o_cc"][cci_0_based, :]))
                else:
                    Transform_copy = np.vstack((Transform_copy, ops["o_cc"]))

                scales = [np.full((ops["o_svd_n_dims"], ops["o_cc"].shape[0]), -1.0)]
                scale_array.extend(scales)
            else:
                cont_cov_indices_0_based = [idx - 1 for idx in ops["out_cont_cov"]]
                
                if ops.get("out_use_norm_cont_cov", False):
                    my_info("ADDING NORMALIZED CONTINUOUS COVARIATES TO EIGEN GENES.")
                    if ops.get("out_all_norm_cont_cov", False):
                        selected_ncc = ops["o_ncc"][cont_cov_indices_0_based, :]
                        Transform_copy = np.vstack((Transform_copy, selected_ncc))
                    else:
                        for cci_0_based in cont_cov_indices_0_based:
                            if (cci_0_based + 1) in ops.get("out_norm_cont_cov", []):
                                Transform_copy = np.vstack((Transform_copy, ops["o_ncc"][cci_0_based, :]))
                            else:
                                Transform_copy = np.vstack((Transform_copy, ops["o_cc"][cci_0_based, :]))
                else:
                    selected_cc = ops["o_cc"][cont_cov_indices_0_based, :]
                    Transform_copy = np.vstack((Transform_copy, selected_cc))
                
                scales = [np.full((ops["o_svd_n_dims"], len(ops["out_cont_cov"])), -1.0)]
                scale_array.extend(scales)

        if scale_array:
            ops["S_OH"] = np.hstack(scale_array)
        else:
            ops["S_OH"] = np.empty((ops["o_svd_n_dims"], 0))

        if ops.get("init_scale_change", False):
            my_warn("BATCH SCALE FACTORS WILL BE ALTERED. SET :init_scale_change TO false IF THIS IS NOT THE DESIRED BEHAVIOR.")
            if ops.get("init_scale_1", False):
                my_warn("BATCH SCALE FACTORS HAVE BEEN ALTERED TO 1.")
                ops["S_OH"] = np.zeros_like(ops["S_OH"])
            else:
                my_warn("BATCH SCALE FACTORS HAVE BEEN ALTERED TO HALFWAY BETWEEN THEIR INITIAL GUESS AND 1.")
                ops["S_OH"] = ops["S_OH"] / 2

        if not (are_there_discontinuous_covariates and ops.get("out_use_disc_cov", False)) and \
           not (ops.get("out_use_cont_cov", False) and are_there_continuous_covariates):
            my_error("It appears there are conflicting inputs. Please check the \":out_\" keys relating to covariates.")
        else:
            if "o_svd_n_dims" not in ops:
                ops["o_svd_n_dims"] = Transform.shape[0]
            
            ops["o_covariates"] = Transform_copy[ops["o_svd_n_dims"]:, :].T.astype(np.float32)

    return Transform_copy.astype(np.float32)



def Eigengenes(dataFile: pd.DataFrame, genesOfInterest: List[str], ops: Dict[str, Any]) -> np.ndarray:
    my_info("BEGIN DATA PREPROCESSING")

    ops = CovariateProcessing(dataFile, ops)

    bluntedDataFile = GetBluntXthPercentile(dataFile, ops, pd.DataFrame)
    
    keepGenesOfInterestExpressionData, _ = GeneLevelCutoffs(bluntedDataFile, genesOfInterest, ops)

    meanNormalizedData = GetMeanNormalizedData(keepGenesOfInterestExpressionData, ops)

    V, Vt, S, cumvar, _ = SVDTransform(meanNormalizedData, ops)

    SVDBatchRegression(V, Vt, ops)

    dimension_indices_0_indexed = SVDRegressionExclusion(S, ops)
    
    Transform = 10 * Vt[dimension_indices_0_indexed, :]

    OHTransform = CovariateOnehotEncoder(Transform, ops)

    return OHTransform

class CircularNode(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 2:
            my_error(f"Invalid input shape for CircularNode. Last dimension must be 2, but got {x.shape}.")

        norm = torch.sqrt(torch.sum(x.pow(2), dim=-1, keepdim=True))

        if torch.any(norm == 0.0):
            my_error("One or more inputs to the circular node have a magnitude of 0.")
        if torch.any(torch.isinf(norm)):
            my_error("One or more inputs to the circular node have an infinite magnitude.")
        if torch.any(torch.isnan(norm)):
            my_error("One or more inputs to the circular node have a NaN magnitude.")

        return x / norm

class Order(nn.Module):
    def __init__(self, L1_layer: nn.Module, c_layer: nn.Module, L2_layer: nn.Module, output_dim: int):
        super(Order, self).__init__()
        self.L1 = L1_layer
        self.c = c_layer
        self.L2 = L2_layer
        self.o = output_dim

    def forward(self, x):
        x = self.L1(x)
        x = self.c(x)
        x = self.L2(x)
        return x

class Covariates(nn.Module):
    def __init__(
        self,
        S_OH_initial_value: torch.Tensor,
        B_initial_value: torch.Tensor,
        B_OH_initial_value: torch.Tensor,
        L1_layer: nn.Module,
        L2_layer: nn.Module,
        output_dimension: int
    ):
        super(Covariates, self).__init__()

        self.S_OH = nn.Parameter(S_OH_initial_value)
        self.B = nn.Parameter(B_initial_value)
        self.B_OH = nn.Parameter(B_OH_initial_value)

        self.L1 = L1_layer
        self.L2 = L2_layer

        self.C = CircularNode()

        self.register_buffer('o', torch.tensor(output_dimension, dtype=torch.int))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            return [self.forward(xi) for xi in x]
        if x.ndim == 2:
            return torch.stack([self.forward(x[:, i]) for i in range(x.shape[1])], dim=1)
        encoding_onehot = x[:self.o] * (1 + (self.S_OH @ x[self.o:])) + (self.B_OH @ x[self.o:]) + self.B
        fully_connected_encoding = self.L1(encoding_onehot)
        circular_layer = self.C(fully_connected_encoding)
        fully_connected_decoding = self.L2(circular_layer)
        denominator = (1 + (self.S_OH @ x[self.o:]))
        decoding_onehot = (fully_connected_decoding - (self.B_OH @ x[self.o:]) - self.B) / denominator
        return decoding_onehot

def InitializeModel(eigen_data: np.ndarray, options: Dict[str, Any]) -> List[Any]:
    if eigen_data.shape[0] == options["o_svd_n_dims"]:
        output_models = [
            Order(
                nn.Linear(options["o_svd_n_dims"], 2),
                CircularNode, 
                nn.Linear(2, options["o_svd_n_dims"]), 
                options["o_svd_n_dims"])
            for _ in range(options["train_n_models"])
        ]
        return output_models
    else:
        x_terms_original = eigen_data[options["o_svd_n_dims"]:, :]
        y_terms_original = eigen_data[:options["o_svd_n_dims"], :]
        x_terms_for_lstsq = x_terms_original.T
        num_samples = x_terms_for_lstsq.shape[0]
        intercept_column = np.ones((num_samples, 1), dtype=np.float64)
        X_design = np.hstack((x_terms_for_lstsq, intercept_column))
        Y_design = y_terms_original.T
        llsq_coeffs_raw_tuple = np.linalg.lstsq(X_design, Y_design, rcond=None)
        llsq_coeffs = llsq_coeffs_raw_tuple[0]
        llsq_coeffs = -llsq_coeffs.T

        B = llsq_coeffs[:, -1]
        B_OH = llsq_coeffs[:, :-1]
        
        if "S_OH" not in options:
            my_warn("'S_OH' not found in options. Using a dummy value for scaled_B_OH calculation.")
            options["S_OH"] = np.ones(B_OH.shape)
        
        scaled_B_OH = (1 + options["S_OH"]) * (B_OH + B[:, np.newaxis]) - B[:, np.newaxis]
        
        options["B"] = B
        options["B_OH"] = scaled_B_OH
        
        output_models = []
        for _ in range(options["train_n_models"]):
            use_S_OH = np.random.rand(*options["S_OH"].shape) * options["S_OH"]
            use_B = np.random.rand(*B.shape) * B
            use_B_OH = np.random.rand(*B_OH.shape) * B_OH
            use_S_OH_tensor = torch.from_numpy(use_S_OH).float()
            use_B_tensor = torch.from_numpy(use_B).float()
            use_B_OH_tensor = torch.from_numpy(use_B_OH).float()
            
            output_models.append(
                Covariates(
                    use_S_OH_tensor,
                    use_B_tensor,
                    use_B_OH_tensor,
                    nn.Linear(options["o_svd_n_dims"], 2),
                    nn.Linear(2, options["o_svd_n_dims"]),
                    options["o_svd_n_dims"]
                )
            )
        return output_models

def TrainOrder(
    m: Order,
    gea_vectorized: List[torch.Tensor],
    MinSteps: int = 250,
    MaxSteps: int = 1000,
    μA: float = 0.0001,
    β: Tuple[float, float] = (0.9, 0.999),
    cutoff: int = 1000
) -> Order:
    μA_original = copy.deepcopy(μA)
    
    optimizer = optim.Adam(list(m.L1.parameters()) + list(m.L2.parameters()), lr=μA, betas=β)

    c1 = 0
    while c1 < MinSteps:
        c1 += 1
        for x in gea_vectorized:
            optimizer.zero_grad()
            output = m(x)
            target = x[:m.o]
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

    c2 = 0
    c3 = 0
    after = 0.0
    while (c2 < MaxSteps) and (μA > μA_original / cutoff):
        c2 += 1
        μA = μA * 1.05
        for param_group in optimizer.param_groups:
            param_group['lr'] = μA

        before_losses = [F.mse_loss(m(x), x[:m.o]) for x in gea_vectorized]
        before = torch.mean(torch.stack(before_losses)).item() # Convert to scalar
        
        before_m_state_dict = copy.deepcopy(m.state_dict())

        for x in gea_vectorized:
            optimizer.zero_grad()
            output = m(x)
            target = x[:m.o]
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        after_losses = [F.mse_loss(m(x), x[:m.o]) for x in gea_vectorized]
        after = torch.mean(torch.stack(after_losses)).item() # Convert to scalar
        
        change = before - after
        c4 = 0
        while (change <= 0) and (μA > μA_original / cutoff):
            c3 += 1
            c4 += 1
            μA = μA * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = μA
            
            m.load_state_dict(before_m_state_dict)

            for x in gea_vectorized:
                optimizer.zero_grad()
                output = m(x)
                target = x[:m.o]
                loss = F.mse_loss(output, target)
                loss.backward()
                optimizer.step()
            
            after_losses = [F.mse_loss(m(x), x[:m.o]) for x in gea_vectorized]
            after = torch.mean(torch.stack(after_losses)).item() # Convert to scalar
            
            change = before - after
    return m

def MultiTrainOrder(m_array: List[Any], gea: np.ndarray, options: Dict[str, Any]) -> List[Any]:
    gea_vectorized = [gea[:, i] for i in range(gea.shape[1])]
    
    trained_models = []
    for model in m_array:
        trained_models.append(
            TrainOrder(model, gea_vectorized, μA=options["train_muA"], β=options["train_beta"],
                       MinSteps=options["train_min_steps"], MaxSteps=options["train_max_steps"],
                       cutoff=options["train_muA_scale_lim"])
        )
    
    return trained_models


def sqrtsumofsquares(
    x: Union[torch.Tensor, List[torch.Tensor]],
    m: Union[Covariates, Order] = None
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if m is not None:
        if isinstance(m, (Covariates, Order)):
            x = m(x)
        else:
            raise TypeError("'m' must be an instance of Covariates or Order.")

    if isinstance(x, torch.Tensor):
        if x.ndim == 1:
            return torch.sqrt(torch.sum(x ** 2))
        elif x.ndim == 2:
            return torch.sqrt(torch.sum(x ** 2, dim=0))
        else:
            raise ValueError("Unsupported torch.Tensor dimensions. Expected 1D or 2D.")
    elif isinstance(x, list):
        if all(isinstance(item, torch.Tensor) and item.ndim == 1 for item in x):
            return [torch.sqrt(torch.sum(item ** 2)) for item in x]
        else:
            raise ValueError("Unsupported list structure. Expected a list of 1D torch.Tensor.")
    else:
        raise TypeError(f"Unsupported input type for sqrtsumofsquares: {type(x)}. "
                        "Expected torch.Tensor or List[torch.Tensor].")


def OrderMagnitude(
    model: Order,
    ohEigenData: Union[np.ndarray, torch.Tensor],
    OUT_TYPE: Type = np.float64
) -> np.ndarray:
    
    m = model
    if isinstance(ohEigenData, np.ndarray):
        ohEigenData_tensor = torch.from_numpy(ohEigenData).float()
    elif isinstance(ohEigenData, torch.Tensor):
        ohEigenData_tensor = ohEigenData.float()
    else:
        my_error("ohEigenData must be a numpy.ndarray or torch.Tensor.")
    num_samples = ohEigenData_tensor.shape[1]
    
    magnitudes = torch.zeros(num_samples, dtype=torch.float32)
    for ii in range(num_samples):
        sample_input = ohEigenData_tensor[:m.o, ii]
        l1_output = m.L1(sample_input)
        magnitudes[ii] = torch.norm(l1_output, p=2)
        
    output_magnitudes = magnitudes.numpy().astype(OUT_TYPE)
    return output_magnitudes

def OrderSkipCircularNode(
    x_data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Order
) -> Union[torch.Tensor, List[torch.Tensor]]:
    processed_x_data = None
    if isinstance(x_data, np.ndarray):
        processed_x_data = torch.from_numpy(x_data).float()
    elif isinstance(x_data, torch.Tensor):
        processed_x_data = x_data.float()
    elif isinstance(x_data, list):
        temp_list = []
        for item in x_data:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray or torch.Tensor, but got {type(item)}")
        processed_x_data = temp_list
    else:
        my_error(f"Unsupported input type for OrderSkipCircularNode: {type(x_data)}. "
                        "Expected torch.Tensor, numpy.ndarray, or List thereof.")

    if isinstance(processed_x_data, torch.Tensor):
        if processed_x_data.ndim == 1:
            return m.L2(m.L1(processed_x_data))
        elif processed_x_data.ndim == 2:
            transposed_input = processed_x_data.T
            
            l1_out = m.L1(transposed_input)
            final_out = m.L2(l1_out)

            return final_out.T
            
        else:
            my_error(f"Unsupported torch.Tensor dimensions: {processed_x_data.ndim}. Expected 1 or 2.")
            
    elif isinstance(processed_x_data, list):
        results = []
        for y_tensor in processed_x_data:
            if not (isinstance(y_tensor, torch.Tensor) and y_tensor.ndim == 1):
                my_error("List elements must be 1D torch.Tensor.")
            
            output = m.L2(m.L1(y_tensor))
            results.append(output)
        return results
    else:
        my_error("Unexpected internal type after input processing in OrderSkipCircularNode.")


def OrderSkipCircularNodeMagnitude(
    x_data: Union[torch.Tensor, List[torch.Tensor]],
    m: Order
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(x_data, torch.Tensor):
        x_data = x_data.float() 
        
        if x_data.ndim == 1:
            return sqrtsumofsquares(OrderSkipCircularNode(x_data, m))
        elif x_data.ndim == 2:
            magnitudes = []
            for i in range(x_data.shape[1]):
                column_sample = x_data[:, i]
                output_skipped_node = OrderSkipCircularNode(column_sample, m)
                magnitude = sqrtsumofsquares(output_skipped_node)
                magnitudes.append(magnitude)
            
            return torch.stack(magnitudes)
            
        else:
            my_error(f"Unsupported torch.Tensor dimensions: {x_data.ndim}. Expected 1 or 2.")
            
    elif isinstance(x_data, list) and all(isinstance(elem, torch.Tensor) for elem in x_data):
        results = []
        for y in x_data:
            if not (isinstance(y, torch.Tensor) and y.ndim == 1):
                my_error("List elements must be 1D torch.Tensor.")
            y = y.float()
            
            output_skipped_node = OrderSkipCircularNode(y, m)
            magnitude = sqrtsumofsquares(output_skipped_node)
            results.append(magnitude)
        return results
    else:
        my_error(f"Unsupported input type or structure for OrderSkipCircularNodeMagnitude: {type(x_data)}. "
                        "Expected torch.Tensor (1D/2D) or List[torch.Tensor (1D)].")

def OrderThroughEncodingDense(
    x_data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Order
) -> Union[torch.Tensor, List[torch.Tensor]]:
    processed_x_data = None
    if isinstance(x_data, np.ndarray):
        processed_x_data = torch.from_numpy(x_data).float()
    elif isinstance(x_data, torch.Tensor):
        processed_x_data = x_data.float()
    elif isinstance(x_data, list):
        temp_list = []
        for item in x_data:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"列表元素必须是 1D numpy.ndarray 或 torch.Tensor，但得到的是 {type(item)}")
        processed_x_data = temp_list
    else:
        my_error(f"OrderThroughEncodingDense 不支持的输入类型: {type(x_data)}。 "
                        "期望是 torch.Tensor, numpy.ndarray 或它们的列表。")

    if isinstance(processed_x_data, torch.Tensor):
        if processed_x_data.ndim == 1:
            return m.L1(processed_x_data)
        elif processed_x_data.ndim == 2:
            transposed_input = processed_x_data.T
            
            l1_out = m.L1(transposed_input)
            
            return l1_out.T
            
        else:
            my_error(f"不支持的 torch.Tensor 维度: {processed_x_data.ndim}。期望 1 或 2。")
            
    elif isinstance(processed_x_data, list):
        results = []
        for y_tensor in processed_x_data:
            if not (isinstance(y_tensor, torch.Tensor) and y_tensor.ndim == 1):
                my_error("列表元素必须是 1D torch.Tensor。")
            
            output = m.L1(y_tensor)
            results.append(output)
        return results
    else:
        my_error("OrderThroughEncodingDense 中输入处理后出现意外的内部类型。")

def OrderThroughCircularNode(
    x_data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Order
) -> Union[torch.Tensor, List[torch.Tensor]]:
    encoding_dense_output = OrderThroughEncodingDense(x_data, m)
    if isinstance(encoding_dense_output, torch.Tensor):
        if encoding_dense_output.ndim == 1:
            return m.c(encoding_dense_output)
        elif encoding_dense_output.ndim == 2:
            return m.c(encoding_dense_output.T).T
        else:
            my_error(f"OrderThroughEncodingDense 返回了不支持的张量维度: {encoding_dense_output.ndim}。")
            

def OrderPhase(
    x_data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Order
) -> Union[torch.Tensor, List[torch.Tensor]]:
    processed_x_data = None
    if isinstance(x_data, np.ndarray):
        processed_x_data = torch.from_numpy(x_data).float()
    elif isinstance(x_data, torch.Tensor):
        processed_x_data = x_data.float() 
    elif isinstance(x_data, list):
        temp_list = []
        for item in x_data:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"列表元素必须是 1D numpy.ndarray 或 torch.Tensor，但得到的是 {type(item)}")
        processed_x_data = temp_list
    else:
        my_error(f"OrderPhase 不支持的输入类型: {type(x_data)}。 "
                        "期望是 torch.Tensor, numpy.ndarray 或它们的列表。")

    if isinstance(processed_x_data, torch.Tensor):
        if processed_x_data.ndim == 1:
            circular_output = OrderThroughCircularNode(processed_x_data, m)

            phase = torch.atan2(circular_output[1], circular_output[0])
            return torch.fmod(phase, 2 * torch.pi)
        elif processed_x_data.ndim == 2:
            
            transposed_input = processed_x_data.T 
            
            circular_outputs_batch = OrderThroughCircularNode(transposed_input, m)
            
            phases_batch = torch.atan2(circular_outputs_batch[:, 1], circular_outputs_batch[:, 0])
            return torch.fmod(phases_batch, 2 * torch.pi)
            
        else:
            my_error(f"不支持的 torch.Tensor 维度: {processed_x_data.ndim}。期望 1 或 2。")
            
    elif isinstance(processed_x_data, list):
        results = []
        for y_tensor in processed_x_data:
            if not (isinstance(y_tensor, torch.Tensor) and y_tensor.ndim == 1):
                my_error("列表元素必须是 1D torch.Tensor。")
            
            circular_output = OrderThroughCircularNode(y_tensor, m)
            phase = torch.atan2(circular_output[1], circular_output[0])
            results.append(torch.fmod(phase, 2 * torch.pi))
        return results
    else:
        my_error("OrderPhase 中输入处理后出现意外的内部类型。")

def MSELoss(
    x_l: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m_l: Order
) -> torch.Tensor:
    processed_x_l = None
    if isinstance(x_l, np.ndarray):
        processed_x_l = torch.from_numpy(x_l).float()
    elif isinstance(x_l, torch.Tensor):
        processed_x_l = x_l.float()
    elif isinstance(x_l, list):
        temp_list = []
        for item in x_l:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"列表元素必须是 1D numpy.ndarray 或 torch.Tensor，但得到的是 {type(item)}")
        processed_x_l = temp_list
    else:
        my_error(f"MSELoss 不支持的输入类型: {type(x_l)}。")

    if isinstance(processed_x_l, torch.Tensor):
        if processed_x_l.ndim == 1:
            predictions = m_l(processed_x_l)
            targets = processed_x_l[:m_l.o]
            return F.mse_loss(predictions, targets)
        elif processed_x_l.ndim == 2:
            input_for_model = processed_x_l
            predictions = m_l(input_for_model)
            targets_julia_style = processed_x_l[:m_l.o, :]
            targets_for_loss = targets_julia_style
            return my_mse_Loss(predictions, targets_for_loss, 0)
        else:
            my_error(f"不支持的 torch.Tensor 维度: {processed_x_l.ndim}。期望 1 或 2。")

    elif isinstance(processed_x_l, list):
        input_batch = torch.stack(processed_x_l)
        predictions_batch = m_l(input_batch)
        targets_batch = torch.stack([y_item[:m_l.o] for y_item in processed_x_l])
        
        return my_mse_Loss(predictions_batch, targets_batch, 0)
    else:
        my_error("MSELoss 中输入处理后出现意外的内部类型。")

def CovariatesEncodingDense(
    x_data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:
    processed_x_data = None
    if isinstance(x_data, np.ndarray):
        processed_x_data = torch.from_numpy(x_data).float()
    elif isinstance(x_data, torch.Tensor):
        processed_x_data = x_data.float()
    elif isinstance(x_data, list):
        temp_list = []
        for item in x_data:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray or torch.Tensor, but got {type(item)}")
        processed_x_data = temp_list
    else:
        my_error(f"CovariatesEncodingDense does not support input type: {type(x_data)}.")

    if isinstance(processed_x_data, torch.Tensor):
        if processed_x_data.ndim == 1:
            return m.L1(processed_x_data)
        elif processed_x_data.ndim == 2:
            results = []
            for i in range(processed_x_data.shape[1]):
                sample_column = processed_x_data[:, i]
                output = CovariatesEncodingDense(sample_column, m)
                results.append(output)
            return torch.stack(results, dim=1)
        else:
            my_error(f"Unsupported torch.Tensor dimensions: {processed_x_data.ndim}. Expected 1 or 2.")

    elif isinstance(processed_x_data, list):
        results = []
        for y_tensor in processed_x_data:
            if not (isinstance(y_tensor, torch.Tensor) and y_tensor.ndim == 1):
                my_error("List elements must be 1D torch.Tensor.")
            
            output = CovariatesEncodingDense(y_tensor, m)
            results.append(output)
        return results
    else:
        my_error("Unexpected internal type after input processing in CovariatesEncodingDense.")

def CovariatesEncodingOH(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:
    processed_x = None
    if isinstance(x, np.ndarray):
        processed_x = torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        processed_x = x.float()
    elif isinstance(x, list):
        temp_list = []
        for item in x:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray or torch.Tensor, but got {type(item)}")
        processed_x = temp_list
    else:
        my_error(f"Unsupported input type for CovariatesEncodingOH: {type(x)}.")

    if isinstance(processed_x, torch.Tensor):
        if processed_x.ndim == 1:
            x_slice_1 = processed_x[:m.o]
            x_slice_2 = processed_x[m.o:]
            
            term1 = x_slice_1 * (1 + (m.S_OH @ x_slice_2))
            term2 = m.B_OH @ x_slice_2
            term3 = m.B
            
            return term1 + term2 + term3
        elif processed_x.ndim == 2:
            results = []
            for i in range(processed_x.shape[1]):
                sample_column = processed_x[:, i]
                output = CovariatesEncodingOH(sample_column, m) # Recursive call for 1D sample
                results.append(output)
            return torch.stack(results, dim=1)
        else:
            my_error(f"Unsupported torch.Tensor dimensions: {processed_x.ndim}. Expected 1 or 2.")

    elif isinstance(processed_x, list):
        results = []
        for y_tensor in processed_x:
            if not (isinstance(y_tensor, torch.Tensor) and y_tensor.ndim == 1):
                my_error("List elements must be 1D torch.Tensor.")
            output = CovariatesEncodingOH(y_tensor, m) # Recursive call for 1D sample
            results.append(output)
        return results
    else:
        my_error("Unexpected internal type after input processing in CovariatesEncodingOH.")


def CovariatesThroughEncodingDense(
    x_data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:
    processed_x_data = None
    if isinstance(x_data, np.ndarray):
        processed_x_data = torch.from_numpy(x_data).float()
    elif isinstance(x_data, torch.Tensor):
        processed_x_data = x_data.float() # 确保 float32
    elif isinstance(x_data, list):
        temp_list = []
        for item in x_data:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"列表元素必须是 1D numpy.ndarray 或 torch.Tensor，但得到的是 {type(item)}")
        processed_x_data = temp_list
    else:
        my_error(f"CovariatesThroughEncodingDense 不支持的输入类型: {type(x_data)}。")

    if isinstance(processed_x_data, torch.Tensor):
        if processed_x_data.ndim == 1:
            oh_output = CovariatesEncodingOH(processed_x_data, m)
            return CovariatesEncodingDense(oh_output, m)
        elif processed_x_data.ndim == 2:
            results = []
            for i in range(processed_x_data.shape[1]): # 迭代列 (样本)
                sample_column = processed_x_data[:, i] # 获取单个列 (1D 张量)
                
                output = CovariatesThroughEncodingDense(sample_column, m)
                results.append(output)
            
            return torch.stack(results, dim=1)
        else:
            my_error(f"不支持的 torch.Tensor 维度: {processed_x_data.ndim}。期望 1 或 2。")

    elif isinstance(processed_x_data, list):
        results = []
        for y_tensor in processed_x_data:
            if not (isinstance(y_tensor, torch.Tensor) and y_tensor.ndim == 1):
                my_error("列表元素必须是 1D torch.Tensor。")
            
            output = CovariatesThroughEncodingDense(y_tensor, m)
            results.append(output)
        return results
    else:
        my_error("在 CovariatesThroughEncodingDense 中处理输入后出现意外的内部类型。")

def CovariatesThroughCircularNode(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:
    processed_x = None
    if isinstance(x, np.ndarray):
        processed_x = torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        processed_x = x.float()
    elif isinstance(x, list):
        temp_list = []
        for item in x:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray or torch.Tensor, but got {type(item)}")
        processed_x = temp_list
    else:
        my_error(f"Unsupported input type for CovariatesThroughCircularNode: {type(x)}.")

    if isinstance(processed_x, torch.Tensor):
        if processed_x.ndim == 1:
            return m.C(CovariatesThroughEncodingDense(processed_x, m))
        elif processed_x.ndim == 2:
            results = []
            for i in range(processed_x.shape[1]):
                sample_column = processed_x[:, i]
                output = CovariatesThroughCircularNode(sample_column, m)
                results.append(output)
            return torch.stack(results, dim=1)
        else:
            my_error(f"Unsupported torch.Tensor dimensions: {processed_x.ndim}. Expected 1 or 2.")
    elif isinstance(processed_x, list):
        results = []
        for y_tensor in processed_x:
            if not (isinstance(y_tensor, torch.Tensor) and y_tensor.ndim == 1):
                my_error("List elements must be 1D torch.Tensor.")
            output = CovariatesThroughCircularNode(y_tensor, m)
            results.append(output)
        return results
    else:
        my_error("Unexpected internal type after input processing in CovariatesThroughCircularNode.")


def get_circ_mse(
    the_eigen_data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    the_model: Union[Covariates, Order]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    
    processed_data = None
    if isinstance(the_eigen_data, np.ndarray):
        processed_data = torch.from_numpy(the_eigen_data).float()
    elif isinstance(the_eigen_data, torch.Tensor):
        processed_data = the_eigen_data.float()
    elif isinstance(the_eigen_data, list):
        temp_list = []
        for item in the_eigen_data:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"列表元素必须是 1D numpy.ndarray 或 torch.Tensor，但得到的是 {type(item)}")
        processed_data = temp_list
    else:
        my_error(f"get_circ_mse 不支持的输入类型: {type(the_eigen_data)}。")

    def _calculate_single_sample_circ_mse(sample_data: torch.Tensor, model: Union[Covariates, Order]) -> torch.Tensor:
        if isinstance(model, Covariates):
            prediction_output = CovariatesThroughEncodingDense(sample_data, model)
            target_output = CovariatesThroughCircularNode(sample_data, model)
            return F.mse_loss(prediction_output, target_output)
        elif isinstance(model, Order):
            prediction_output = OrderThroughEncodingDense(sample_data, model)
            target_output = OrderThroughCircularNode(sample_data, model)
            return F.mse_loss(prediction_output, target_output)
        else:
            my_error(f"不支持的模型类型: {type(model)}。期望 Covariates 或 Order。")

    if isinstance(processed_data, torch.Tensor):
        if processed_data.ndim == 1:
            return _calculate_single_sample_circ_mse(processed_data, the_model)
        elif processed_data.ndim == 2:
            results = []
            for i in range(processed_data.shape[1]):
                sample_column = processed_data[:, i]
                mse_value = _calculate_single_sample_circ_mse(sample_column, the_model)
                results.append(mse_value)
            return torch.stack(results)
        else:
            my_error(f"不支持的 torch.Tensor 维度: {processed_data.ndim}。期望 1 或 2。")

    elif isinstance(processed_data, list):
        results = []
        for sample_item in processed_data:
            if not (isinstance(sample_item, torch.Tensor) and sample_item.ndim == 1):
                my_error("列表元素必须是 1D torch.Tensor。")
            mse_value = _calculate_single_sample_circ_mse(sample_item, the_model)
            results.append(mse_value)
        return results
    else:
        my_error("在 get_circ_mse 中处理输入后出现意外的内部类型。")

def CovariatesDecodingDense(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:
    processed_x = None
    if isinstance(x, np.ndarray):
        processed_x = torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        processed_x = x.float()
    elif isinstance(x, list):
        temp_list = []
        for item in x:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray or torch.Tensor, but got {type(item)}")
        processed_x = temp_list
    else:
        my_error(f"Unsupported input type for CovariatesDecodingDense: {type(x)}.")

    if isinstance(processed_x, torch.Tensor):
        if processed_x.ndim == 1:
            return m.L2(processed_x)
        elif processed_x.ndim == 2:
            results = []
            for i in range(processed_x.shape[1]):
                sample_column = processed_x[:, i]
                output = CovariatesDecodingDense(sample_column, m)
                results.append(output)
            return torch.stack(results, dim=1)
        else:
            my_error(f"Unsupported torch.Tensor dimensions: {processed_x.ndim}. Expected 1 or 2.")
    elif isinstance(processed_x, list):
        results = []
        for y_tensor in processed_x:
            if not (isinstance(y_tensor, torch.Tensor) and y_tensor.ndim == 1):
                my_error("List elements must be 1D torch.Tensor.")
            output = CovariatesDecodingDense(y_tensor, m)
            results.append(output)
        return results
    else:
        my_error("Unexpected internal type after input processing in CovariatesDecodingDense.")



def CovariatesSkipCircularNodeDecodingDense(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    encoded_dense_output = CovariatesThroughEncodingDense(x, m)
    
    return CovariatesDecodingDense(encoded_dense_output, m)

def CovariatesDecodingOH(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray, List[float]]]],
    y: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray, List[float]]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 1 and \
       isinstance(y, (torch.Tensor, np.ndarray)) and y.ndim == 1:
        
        x_tensor = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()
        y_tensor = torch.from_numpy(y).float() if isinstance(y, np.ndarray) else y.float()
        
        x_slice_from_m_o = x_tensor[m.o:]
        
        term_numerator = y_tensor - (m.B_OH @ x_slice_from_m_o) - m.B
        term_denominator = 1 + (m.S_OH @ x_slice_from_m_o)
        
        return term_numerator / term_denominator

    x_normalized_list = []
    if isinstance(x, np.ndarray) and x.ndim == 2:
        for i in range(x.shape[1]):
            x_normalized_list.append(torch.from_numpy(x[:, i]).float())
    elif isinstance(x, torch.Tensor) and x.ndim == 2:
        for i in range(x.shape[1]):
            x_normalized_list.append(x[:, i].float())
    elif isinstance(x, list):
        for item in x:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                x_normalized_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                x_normalized_list.append(item.float())
            elif isinstance(item, List):
                x_normalized_list.append(torch.tensor(item).float())
            else:
                my_error(f"Unsupported list element type for x: {type(item)}")
    else:
        my_error(f"Unsupported input type for x: {type(x)}")

    y_normalized_list = []
    if isinstance(y, np.ndarray) and y.ndim == 2:
        for i in range(y.shape[1]):
            y_normalized_list.append(torch.from_numpy(y[:, i]).float())
    elif isinstance(y, torch.Tensor) and y.ndim == 2:
        for i in range(y.shape[1]):
            y_normalized_list.append(y[:, i].float())
    elif isinstance(y, list):
        for item in y:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                y_normalized_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                y_normalized_list.append(item.float())
            elif isinstance(item, List):
                y_normalized_list.append(torch.tensor(item).float())
            else:
                my_error(f"Unsupported list element type for y: {type(item)}")
    else:
        my_error(f"Unsupported input type for y: {type(y)}")

    if len(x_normalized_list) != len(y_normalized_list):
        my_error("Normalized input lists for x and y must have the same number of samples.")

    results = []
    for x_i, y_i in zip(x_normalized_list, y_normalized_list):
        result_i = CovariatesDecodingOH(x_i, y_i, m)
        results.append(result_i)
    
    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 2:
        return torch.stack(results, dim=1) # Stack columns for 2D output
    else:
        return results

def CovariatesSkipCircularNodeDecodingOH(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:
    
    decoded_dense_output = CovariatesSkipCircularNodeDecodingDense(x, m)
    
    return CovariatesDecodingOH(x, decoded_dense_output, m)


def get_skip_circle_mse(
    sample_eigen_expression: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    the_model: Union[Covariates, Order]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    processed_data = None
    if isinstance(sample_eigen_expression, np.ndarray):
        processed_data = torch.from_numpy(sample_eigen_expression).float()
    elif isinstance(sample_eigen_expression, torch.Tensor):
        processed_data = sample_eigen_expression.float()
    elif isinstance(sample_eigen_expression, list):
        temp_list = []
        for item in sample_eigen_expression:
            if isinstance(item, np.ndarray):
                temp_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor):
                temp_list.append(item.float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray or torch.Tensor, but got {type(item)}")
        processed_data = temp_list
    else:
        my_error(f"Unsupported input type for get_skip_circle_mse: {type(sample_eigen_expression)}.")

    def _calculate_single_sample_skip_mse(
        sample_data: torch.Tensor, 
        model: Union[Covariates, Order]
    ) -> torch.Tensor:
        if isinstance(model, Covariates):
            target = sample_data[:model.o]
            prediction = CovariatesSkipCircularNodeDecodingOH(sample_data, model)
            return F.mse_loss(target, prediction)
        elif isinstance(model, Order):
            target = sample_data[:model.o]
            prediction = OrderSkipCircularNode(sample_data, model)
            return F.mse_loss(target, prediction)
        else:
            my_error(f"Unsupported model type: {type(model)}. Expected Covariates or Order.")

    if isinstance(processed_data, torch.Tensor):
        if processed_data.ndim == 1:
            return _calculate_single_sample_skip_mse(processed_data, the_model)
        elif processed_data.ndim == 2:
            results = []
            for i in range(processed_data.shape[1]):
                sample_column = processed_data[:, i]
                mse_value = _calculate_single_sample_skip_mse(sample_column, the_model)
                results.append(mse_value)
            return torch.stack(results)
        else:
            my_error(f"Unsupported torch.Tensor dimensions: {processed_data.ndim}. Expected 1 or 2.")
    elif isinstance(processed_data, list):
        results = []
        for sample_item in processed_data:
            if not (isinstance(sample_item, torch.Tensor) and sample_item.ndim == 1):
                my_error("List elements must be 1D torch.Tensor.")
            mse_value = _calculate_single_sample_skip_mse(sample_item, the_model)
            results.append(mse_value)
        return results
    else:
        my_error("Unexpected internal type after input processing in get_skip_circle_mse.")

def get_out_of_plane_reconstruction_error(
    sample_eigen_expression: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray, List[float]]]],
    the_model: Union[Covariates, Order]
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(sample_eigen_expression, (torch.Tensor, np.ndarray)) and sample_eigen_expression.ndim == 1:
        x_tensor = torch.from_numpy(sample_eigen_expression).float() if isinstance(sample_eigen_expression, np.ndarray) else sample_eigen_expression.float()

        if isinstance(the_model, Covariates):
            prediction_main = the_model(x_tensor)
            prediction_skip = CovariatesSkipCircularNodeDecodingOH(x_tensor, the_model)
            return F.mse_loss(prediction_main, prediction_skip)
        elif isinstance(the_model, Order):
            prediction_main = the_model(x_tensor)
            prediction_skip = OrderSkipCircularNode(x_tensor, the_model)
            return F.mse_loss(prediction_main, prediction_skip)
        else:
            my_error(f"Unsupported model type: {type(the_model)}. Expected Covariates or Order.")

    processed_input_list = []
    if isinstance(sample_eigen_expression, np.ndarray):
        if sample_eigen_expression.ndim == 2:
            for i in range(sample_eigen_expression.shape[1]):
                processed_input_list.append(torch.from_numpy(sample_eigen_expression[:, i]).float())
        else:
             for item in sample_eigen_expression:
                 if isinstance(item, np.ndarray) and item.ndim == 1:
                     processed_input_list.append(torch.from_numpy(item).float())
                 else:
                     my_error(f"Unsupported list element type (numpy.ndarray) for sample_eigen_expression: {type(item)}")
    elif isinstance(sample_eigen_expression, torch.Tensor):
        if sample_eigen_expression.ndim == 2:
            for i in range(sample_eigen_expression.shape[1]):
                processed_input_list.append(sample_eigen_expression[:, i].float())
        else:
            my_error(f"Unsupported torch.Tensor dimensions: {sample_eigen_expression.ndim}. Expected 1 or 2.")
    elif isinstance(sample_eigen_expression, list):
        for item in sample_eigen_expression:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                processed_input_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                processed_input_list.append(item.float())
            elif isinstance(item, List):
                processed_input_list.append(torch.tensor(item).float())
            else:
                my_error(f"Unsupported list element type for sample_eigen_expression: {type(item)}")
    else:
        my_error(f"Unsupported input type: {type(sample_eigen_expression)}.")

    results = []
    for single_sample_tensor in processed_input_list:
        error_value = get_out_of_plane_reconstruction_error(single_sample_tensor, the_model)
        results.append(error_value)
    
    if isinstance(sample_eigen_expression, (torch.Tensor, np.ndarray)) and sample_eigen_expression.ndim == 2:
        return torch.stack(results)
    else:
        return results

def OrderDecoder(best_model: Any, gea: np.ndarray, OUT_TYPE: type = np.float64) -> Tuple[Any, np.ndarray]:
    eigenspace_magnitudes = sqrtsumofsquares(gea)
    magnitudes = OrderMagnitude(best_model, gea, np.float32)
    decoding_eigenspace_magnitudes = sqrtsumofsquares(gea, best_model)
    
    decoding_eigenspace_skip_circle_magnitudes = OrderSkipCircularNodeMagnitude(gea, best_model)
    
    projections_list = OrderThroughEncodingDense(gea, best_model)
    projections = np.vstack([p.T for p in projections_list])
    
    phases = OrderPhase(gea, best_model)

    model_errors = MSELoss(gea, best_model)
    circ_errors = get_circ_mse(gea, best_model)

    skip_circle_errors = get_skip_circle_mse(gea, best_model)

    out_of_plane_errors = get_out_of_plane_reconstruction_error(gea, best_model)

    phases = phases.reshape(-1, 1) if phases.ndim == 1 else phases
    model_errors = model_errors.reshape(-1, 1) if model_errors.ndim == 1 else model_errors
    magnitudes = magnitudes.reshape(-1, 1) if magnitudes.ndim == 1 else magnitudes
    circ_errors = circ_errors.reshape(-1, 1) if circ_errors.ndim == 1 else circ_errors
    skip_circle_errors = skip_circle_errors.reshape(-1, 1) if skip_circle_errors.ndim == 1 else skip_circle_errors
    out_of_plane_errors = out_of_plane_errors.reshape(-1, 1) if out_of_plane_errors.ndim == 1 else out_of_plane_errors
    eigenspace_magnitudes = eigenspace_magnitudes.reshape(-1, 1) if eigenspace_magnitudes.ndim == 1 else eigenspace_magnitudes
    decoding_eigenspace_magnitudes = decoding_eigenspace_magnitudes.reshape(-1, 1) if decoding_eigenspace_magnitudes.ndim == 1 else decoding_eigenspace_magnitudes
    decoding_eigenspace_skip_circle_magnitudes = decoding_eigenspace_skip_circle_magnitudes.reshape(-1, 1) if decoding_eigenspace_skip_circle_magnitudes.ndim == 1 else decoding_eigenspace_skip_circle_magnitudes

    metrics_hcat = np.hstack((
        phases,
        model_errors,
        magnitudes,
        projections,
        circ_errors,
        skip_circle_errors,
        out_of_plane_errors,
        eigenspace_magnitudes,
        decoding_eigenspace_magnitudes,
        decoding_eigenspace_skip_circle_magnitudes
    ))

    return best_model, metrics_hcat

def TrainCovariatesCircular(
    m: Covariates,
    gea_vectorized: List[torch.Tensor],
    MinSteps: int = 250,
    MaxSteps: int = 1000,
    μA: float = 0.0001,
    β: Tuple[float, float] = (0.9, 0.999),
    cutoff: int = 1000
) -> Tuple[Covariates, float]:

    μA_original = copy.deepcopy(μA)

    optimizer_L1_L2 = optim.Adam(list(m.L1.parameters()) + list(m.L2.parameters()), lr=μA, betas=β)
    
    optimizer_SB = optim.Adam([m.S_OH, m.B, m.B_OH], lr=μA, betas=β)

    c1 = 0
    while c1 < MinSteps:
        c1 += 1
        
        for x in gea_vectorized:
            circ_in_val = x[:m.o] * (1 + (m.S_OH @ x[m.o:])) + (m.B_OH @ x[m.o:]) + m.B
            circ_out_val = m.L2(m.c(m.L1(circ_in_val)))
            loss1 = F.mse_loss(circ_in_val, circ_out_val)
            
            optimizer_L1_L2.zero_grad()
            loss1.backward()
            optimizer_L1_L2.step()

            model_output = m(x)
            loss2 = F.mse_loss(model_output, x[:m.o])
            
            optimizer_SB.zero_grad()
            loss2.backward()
            optimizer_SB.step()

    c2 = 0
    c3 = 0

    while (c2 < MaxSteps) and (μA > μA_original / cutoff):
        c2 += 1
        μA = μA * 1.05

        for g in optimizer_L1_L2.param_groups:
            g['lr'] = μA
        for g in optimizer_SB.param_groups:
            g['lr'] = μA

        before_mse_values = [F.mse_loss(m(x), x[:m.o]) for x in gea_vectorized]
        before = torch.mean(torch.stack(before_mse_values))
        before_m_state_dict = copy.deepcopy(m.state_dict())

        for x in gea_vectorized:
            circ_in_val = x[:m.o] * (1 + (m.S_OH @ x[m.o:])) + (m.B_OH @ x[m.o:]) + m.B
            circ_out_val = m.L2(m.c(m.L1(circ_in_val)))
            loss1 = F.mse_loss(circ_in_val, circ_out_val)
            
            optimizer_L1_L2.zero_grad()
            loss1.backward()
            optimizer_L1_L2.step()

            model_output = m(x)
            loss2 = F.mse_loss(model_output, x[:m.o])
            
            optimizer_SB.zero_grad()
            loss2.backward()
            optimizer_SB.step()

        after_mse_values = [F.mse_loss(m(x), x[:m.o]) for x in gea_vectorized]
        after = torch.mean(torch.stack(after_mse_values))
        change = before - after

        c4 = 0
        while (change <= 0) and (μA > μA_original / cutoff):
            c3 += 1
            c4 += 1
            μA = μA * 0.5
            
            m.load_state_dict(before_m_state_dict)

            for g in optimizer_L1_L2.param_groups:
                g['lr'] = μA
            for g in optimizer_SB.param_groups:
                g['lr'] = μA
            
            for x in gea_vectorized:
                circ_in_val = x[:m.o] * (1 + (m.S_OH @ x[m.o:])) + (m.B_OH @ x[m.o:]) + m.B
                circ_out_val = m.L2(m.c(m.L1(circ_in_val)))
                loss1 = F.mse_loss(circ_in_val, circ_out_val)
                
                optimizer_L1_L2.zero_grad()
                loss1.backward()
                optimizer_L1_L2.step()

                model_output = m(x)
                loss2 = F.mse_loss(model_output, x[:m.o])
                
                optimizer_SB.zero_grad()
                loss2.backward()
                optimizer_SB.step()

            after_mse_values = [F.mse_loss(m(x), x[:m.o]) for x in gea_vectorized]
            after = torch.mean(torch.stack(after_mse_values))
            change = before - after

    return m, after


def TrainCovariatesTrueTimes(
    m: Covariates,
    gea_vectorized: List[torch.Tensor],
    collection_times_with_flag_vectorized: List[torch.Tensor],
    MinSteps: int = 250,
    MaxSteps: int = 1000,
    μA: float = 0.0001,
    β: Tuple[float, float] = (0.9, 0.999),
    cutoff: int = 1000,
    collection_time_balance: float = 4
) -> Tuple[Covariates, float]:

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0
    after = 0.0

    def EncodingOHLayer(x_l: torch.Tensor) -> torch.Tensor:
        return x_l[:m.o] * (1 + (m.S_OH @ x_l[m.o:])) + (m.B_OH @ x_l[m.o:]) + m.B
    
    def AfterCircularLayer(x_l: torch.Tensor) -> torch.Tensor:
        return m.c(m.L1(EncodingOHLayer(x_l)))
    
    def MSELoss(x_l: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(m(x_l), x_l[:m.o])

    def TrueTimeLoss2(l_sample_eigendata: torch.Tensor, l_true_radian: torch.Tensor) -> torch.Tensor:
        cyclops_projection = AfterCircularLayer(l_sample_eigendata)
        cyclops_phase = torch.atan2(cyclops_projection[1], cyclops_projection[0])
        cos_distance = 1 - torch.cos(cyclops_phase - l_true_radian[1])
        return 2 * cos_distance * l_true_radian[0]

    def TotalLoss2(x_l: torch.Tensor, y_l: torch.Tensor, balance: float) -> torch.Tensor:
        return MSELoss(x_l) + TrueTimeLoss2(x_l, y_l) * balance
    
    zip_data = list(zip(gea_vectorized, collection_times_with_flag_vectorized))

    try:
        μA_original = copy.deepcopy(μA)

        optimizer = optim.Adam(m.parameters(), lr=μA, betas=β)

        while c1 < MinSteps:
            c1 += 1
            for x, y in zip_data:
                optimizer.zero_grad()
                loss = TotalLoss2(x, y, collection_time_balance)
                loss.backward()
                optimizer.step()

        smallest_μA = μA
        largest_μA = μA

        c2 = 0
        c3 = 0

        while (c2 < MaxSteps) and (μA > μA_original / cutoff):
            c2 += 1
            μA *= 1.05
            for param_group in optimizer.param_groups:
                param_group['lr'] = μA
            
            before_losses = [TotalLoss2(x, y, collection_time_balance).item() for x, y in zip_data]
            before = sum(before_losses) / len(before_losses)
            
            before_m_state_dict = copy.deepcopy(m.state_dict())

            for x, y in zip_data:
                optimizer.zero_grad()
                loss = TotalLoss2(x, y, collection_time_balance)
                loss.backward()
                optimizer.step()

            after_losses = [TotalLoss2(x, y, collection_time_balance).item() for x, y in zip_data]
            after = sum(after_losses) / len(after_losses)
            change = before - after

            while (change <= 0) and (μA > μA_original / cutoff):
                c3 += 1
                c4 += 1
                μA *= 0.5
                
                m.load_state_dict(before_m_state_dict)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = μA
                
                for x, y in zip_data:
                    optimizer.zero_grad()
                    loss = TotalLoss2(x, y, collection_time_balance)
                    loss.backward()
                    optimizer.step()

                after_losses = [TotalLoss2(x, y, collection_time_balance).item() for x, y in zip_data]
                after = sum(after_losses) / len(after_losses)
                change = before - after
            
            if μA < μA_original:
                c5 += 1
                if μA < smallest_μA:
                    smallest_μA = round(μA, 4)
            elif μA > μA_original:
                c6 += 1
                if μA > largest_μA:
                    largest_μA = round(μA, 4)
            
            c2 += c4
            c4 = 0

    except Exception as e:
        my_info(f"An error occurred in training. {c1 + c2 + c4} steps ({c3 + c2} variable) were taken before the error occurred.")
        raise e

    my_info(f"Model took {c1 + c2} total training steps. Of these, {c2} were variable learning rate steps.\n"
          f"The learning rate was decreased {c3} times and was smaller than the original learning rate for {c5} steps.\n"
          f"The learning rate was increased {c2} times and was larger than the original learning rate for {c6} steps.\n"
          f"The final learning rate was {round(μA, 4)}; the smallest the learning rate became was {smallest_μA} and the largest it became was {largest_μA}.\n\n"
          "~~~~~~~~~~~~TRAINING COMPLETE~~~~~~~~~~~~\n\n")

    return m, after

def TrainCovariates(
    m: Covariates,
    gea_vectorized: List[torch.Tensor],
    MinSteps: int = 250,
    MaxSteps: int = 1000,
    μA: float = 0.0001,
    β: Tuple[float, float] = (0.9, 0.999),
    cutoff: int = 1000
) -> Tuple[Covariates, float]:

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0
    after = 0.0

    try:
        μA_original = μA
        
        optimizer = optim.Adam(m.parameters(), lr=μA, betas=β)

        while c1 < MinSteps:
            c1 += 1
            for x in gea_vectorized:
                optimizer.zero_grad()
                loss = F.mse_loss(m(x), x[:m.o])
                loss.backward()
                optimizer.step()
        
        smallest_μA = μA
        largest_μA = μA
        
        while (c2 < MaxSteps) and (μA > μA_original / cutoff):
            
            c2 += 1
            μA *= 1.05
            for param_group in optimizer.param_groups:
                param_group['lr'] = μA
            
            before_losses = [F.mse_loss(m(x), x[:m.o]).item() for x in gea_vectorized]
            before = sum(before_losses) / len(before_losses)
            
            before_m_state_dict = copy.deepcopy(m.state_dict())
            
            for x in gea_vectorized:
                optimizer.zero_grad()
                loss = F.mse_loss(m(x), x[:m.o])
                loss.backward()
                optimizer.step()
            
            after_losses = [F.mse_loss(m(x), x[:m.o]).item() for x in gea_vectorized]
            after = sum(after_losses) / len(after_losses)
            change = before - after

            while (change <= 0) and (μA > μA_original / cutoff):
                c3 += 1
                c4 += 1
                μA *= 0.5
                
                m.load_state_dict(before_m_state_dict)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = μA
                
                for x in gea_vectorized:
                    optimizer.zero_grad()
                    loss = F.mse_loss(m(x), x[:m.o])
                    loss.backward()
                    optimizer.step()
                
                after_losses = [F.mse_loss(m(x), x[:m.o]).item() for x in gea_vectorized]
                after = sum(after_losses) / len(after_losses)
                change = before - after
            
            if μA < μA_original:
                c5 += 1
                if μA < smallest_μA:
                    smallest_μA = round(μA, 4)
            elif μA > μA_original:
                c6 += 1
                if μA > largest_μA:
                    largest_μA = round(μA, 4)
            
            c2 += c4
            c4 = 0

    except Exception as e:
        my_info(f"An error occurred in training. {c1 + c2} steps ({c2} variable) were taken before the error occurred.")
        raise e

    my_info(f"Model took {c1 + c2} total training steps. Of these, {c2} were variable learning rate steps.\n"
          f"The learning rate was decreased {c3} times and was smaller than the original learning rate for {c5} steps.\n"
          f"The learning rate was increased {c2} times and was larger than the original learning rate for {c6} steps.\n"
          f"The final learning rate was {round(μA, 4)}; the smallest the learning rate became was {smallest_μA} and the largest it became was {largest_μA}.\n\n"
          "~~~~~~~~~~~~TRAINING COMPLETE~~~~~~~~~~~~\n\n")
    return m, after

def MultiTrainCovariates(m_array: List[Any], gea: np.ndarray, options: Dict[str, Any]) -> List[Any]:
    trained_models = []
    if isinstance(gea, np.ndarray):
        gea = torch.from_numpy(gea).float()
    gea_vectorized = [gea[:, i] for i in range(gea.shape[1])]

    if options.get("train_circular", False):
        for model in m_array:
            try:
                trained_models.append(
                    TrainCovariatesCircular(
                        model, 
                        gea_vectorized, 
                        μA=options["train_muA"], 
                        β=options["train_beta"],
                        MinSteps=options["train_min_steps"], 
                        MaxSteps=options["train_max_steps"],
                        cutoff=options["train_muA_scale_lim"]
                    )
                )
            except Exception as e:
                raise e
    elif options.get("train_sample_id", None) is not None and options.get("train_collection_times", False):
        if "o_column_ids" not in options:
            my_error("options['o_column_ids'] must be provided when using train_sample_id and train_collection_times.")
        
        known_sample_indices = findXinY(options["train_sample_id"], options["o_column_ids"])
        
        init_collection_time = np.zeros(len(gea_vectorized), dtype=np.float32)
        
        if "train_sample_phase" not in options:
             my_error("options['train_sample_phase'] must be provided when using train_sample_id and train_collection_times.")

        init_collection_time[known_sample_indices] = options["train_sample_phase"]

        init_collection_time_flag = np.full(len(init_collection_time), False, dtype=bool)
        init_collection_time_flag[known_sample_indices] = True
        
        collection_times_with_flag_vectorized_raw = np.vstack((init_collection_time_flag.astype(np.float32), init_collection_time)).T
        collection_times_with_flag_vectorized = [
            collection_times_with_flag_vectorized_raw[i, :] for i in range(collection_times_with_flag_vectorized_raw.shape[0])
        ]
        
        for model in m_array:
            try:
                trained_models.append(
                    TrainCovariatesTrueTimes(model, gea_vectorized, collection_times_with_flag_vectorized,
                        μA=options["train_μA"], β=options["train_β"],
                        MinSteps=options["train_min_steps"], MaxSteps=options["train_max_steps"],
                        cutoff=options["train_μA_scale_lim"],
                        collection_time_balance=options["train_collection_time_balance"]
                    )
                )
            except Exception as e:
                raise e
    else:
        for model in m_array:
            trained_models.append(
                TrainCovariates(model, gea_vectorized, μA=options["train_μA"], β=options["train_β"],
                                MinSteps=options["train_min_steps"], MaxSteps=options["train_max_steps"],
                                cutoff=options["train_μA_scale_lim"])
            )
    return trained_models

def CovariatesEncodingDenseMagnitude(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 1:
        return sqrtsumofsquares(CovariatesEncodingOH(x, m))
    
    processed_x_list = []
    if isinstance(x, np.ndarray) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(torch.from_numpy(x[:, i]).float())
    elif isinstance(x, torch.Tensor) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(x[:, i].float())
    elif isinstance(x, list):
        for item in x:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                processed_x_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                processed_x_list.append(item.float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray or torch.Tensor, but got {type(item)}")
    else:
        my_error(f"Unsupported input type for CovariatesEncodingDenseMagnitude: {type(x)}")

    results = []
    for single_sample_tensor in processed_x_list:
        results.append(CovariatesEncodingDenseMagnitude(single_sample_tensor, m))
    
    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 2:
        return torch.stack(results)
    else:
        return results

def CovariatesMagnitude(
    model: Covariates,
    ohEigenData: Union[torch.Tensor, np.ndarray],
    OUT_TYPE: Type = float
) -> Union[List[float], np.ndarray]:

    m = model

    def OH(x_l: torch.Tensor) -> torch.Tensor:
        return x_l[:m.o] * (1 + (m.S_OH @ x_l[m.o:])) + (m.B_OH @ x_l[m.o:]) + m.B

    def Lin(x_l: torch.Tensor) -> torch.Tensor:
        return m.L1(x_l)

    def M(x_l: torch.Tensor) -> torch.Tensor:
        return Lin(OH(x_l))
    
    if isinstance(ohEigenData, np.ndarray):
        ohEigenData_tensor = torch.from_numpy(ohEigenData).float()
    else:
        ohEigenData_tensor = ohEigenData.float()

    num_samples = ohEigenData_tensor.shape[1]
    magnitudes = torch.zeros(num_samples, dtype=torch.float32)

    for ii in range(num_samples):

        sample_data = ohEigenData_tensor[:, ii]

        magnitudes[ii] = torch.sqrt(torch.sum(M(sample_data) ** 2))
    
    if OUT_TYPE == float:
        return magnitudes.detach().tolist()
    elif OUT_TYPE == np.float64:
        return magnitudes.detach().numpy().astype(np.float64)
    elif OUT_TYPE == np.float32:
        return magnitudes.detach().numpy().astype(np.float32)
    elif OUT_TYPE == torch.Tensor:
        return magnitudes.detach()
    else:
        my_error("Unsupported OUT_TYPE. Must be float, numpy.float32, numpy.float64, or torch.Tensor.")

def CovariatesThroughDecodingDense(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 1:
        through_circular_node_output = CovariatesThroughCircularNode(x, m)
        return CovariatesDecodingDense(through_circular_node_output, m)
    
    processed_x_list = []
    if isinstance(x, np.ndarray) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(torch.from_numpy(x[:, i]).float())
    elif isinstance(x, torch.Tensor) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(x[:, i].float())
    elif isinstance(x, list):
        for item in x:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                processed_x_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                processed_x_list.append(item.float())
            elif isinstance(item, List):
                processed_x_list.append(torch.tensor(item).float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray, torch.Tensor or list, but got {type(item)}")
    else:
        my_error(f"Unsupported input type for CovariatesThroughDecodingDense: {type(x)}")

    results = []
    for single_sample_tensor in processed_x_list:
        results.append(CovariatesThroughDecodingDense(single_sample_tensor, m))
    
    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 2:
        return torch.stack(results, dim=1) 
    else:
        return results

def CovariatesDecodingDenseMagnitude(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 1:
        return sqrtsumofsquares(CovariatesThroughDecodingDense(x, m))
    
    processed_x_list = []
    if isinstance(x, np.ndarray) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(torch.from_numpy(x[:, i]).float())
    elif isinstance(x, torch.Tensor) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(x[:, i].float())
    elif isinstance(x, list):
        for item in x:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                processed_x_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                processed_x_list.append(item.float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray or torch.Tensor, but got {type(item)}")
    else:
        my_error(f"Unsupported input type for CovariatesDecodingDenseMagnitude: {type(x)}")

    results = []
    for single_sample_tensor in processed_x_list:
        results.append(CovariatesDecodingDenseMagnitude(single_sample_tensor, m))
    
    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 2:
        return torch.stack(results)
    else:
        return results

def CovariatesSkipCircularNodeDecodingDenseMagnitude(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 1:
        return sqrtsumofsquares(CovariatesSkipCircularNodeDecodingDense(x, m))
    
    processed_x_list = []
    if isinstance(x, np.ndarray) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(torch.from_numpy(x[:, i]).float())
    elif isinstance(x, torch.Tensor) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(x[:, i].float())
    elif isinstance(x, list):
        for item in x:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                processed_x_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                processed_x_list.append(item.float())
            elif isinstance(item, List):
                processed_x_list.append(torch.tensor(item).float()) # For Array{Array{Float32,1},2} flattened
            else:
                my_error(f"List elements must be 1D numpy.ndarray, torch.Tensor or list, but got {type(item)}")
    else:
        my_error(f"Unsupported input type for CovariatesSkipCircularNodeDecodingDenseMagnitude: {type(x)}")

    results = []
    for single_sample_tensor in processed_x_list:
        magnitude = CovariatesSkipCircularNodeDecodingDenseMagnitude(single_sample_tensor, m)
        results.append(magnitude)
    
    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 2:
        return torch.stack(results).squeeze()
    else:
        return results

def CovariatesSkipCircularNodeDecodingOHMagnitude(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> torch.Tensor:

    decoded_oh_output = CovariatesSkipCircularNodeDecodingOH(x, m)
    if isinstance(decoded_oh_output, list):
        concatenated_output = torch.cat(decoded_oh_output)
        return sqrtsumofsquares(concatenated_output)
    else:
        return sqrtsumofsquares(decoded_oh_output)
def CovariatesPhase(
    x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    m: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 1:
        CircularOutput = CovariatesThroughCircularNode(x, m)

        phase = torch.atan2(CircularOutput[1], CircularOutput[0])
        return torch.remainder(phase, 2 * math.pi)
    
    processed_x_list = []
    if isinstance(x, np.ndarray) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(torch.from_numpy(x[:, i]).float())
    elif isinstance(x, torch.Tensor) and x.ndim == 2:
        for i in range(x.shape[1]):
            processed_x_list.append(x[:, i].float())
    elif isinstance(x, list):
        for item in x:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                processed_x_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                processed_x_list.append(item.float())
            elif isinstance(item, List):
                processed_x_list.append(torch.tensor(item).float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray, torch.Tensor or list, but got {type(item)}")
    else:
        my_error(f"Unsupported input type for CovariatesPhase: {type(x)}")

    results = []
    for single_sample_tensor in processed_x_list:
        results.append(CovariatesPhase(single_sample_tensor, m))
    
    if isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 2:
        return torch.stack(results).squeeze()
    else:
        return results

def get_inner_mse(
    the_eigen_data: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    the_model: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(the_eigen_data, (torch.Tensor, np.ndarray)) and the_eigen_data.ndim == 1:
        encoded_oh_output = CovariatesEncodingOH(the_eigen_data, the_model)
        decoded_dense_output = CovariatesThroughDecodingDense(the_eigen_data, the_model)
        return F.mse_loss(encoded_oh_output, decoded_dense_output)
    
    processed_data_list = []
    if isinstance(the_eigen_data, np.ndarray) and the_eigen_data.ndim == 2:
        for i in range(the_eigen_data.shape[1]):
            processed_data_list.append(torch.from_numpy(the_eigen_data[:, i]).float())
    elif isinstance(the_eigen_data, torch.Tensor) and the_eigen_data.ndim == 2:
        for i in range(the_eigen_data.shape[1]):
            processed_data_list.append(the_eigen_data[:, i].float())
    elif isinstance(the_eigen_data, list):
        for item in the_eigen_data:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                processed_data_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                processed_data_list.append(item.float())
            elif isinstance(item, List):
                processed_data_list.append(torch.tensor(item).float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray, torch.Tensor or list, but got {type(item)}")
    else:
        my_error(f"Unsupported input type for get_inner_mse: {type(the_eigen_data)}")

    results = []
    for sample_tensor in processed_data_list:
        results.append(get_inner_mse(sample_tensor, the_model))
    
    if isinstance(the_eigen_data, (torch.Tensor, np.ndarray)) and the_eigen_data.ndim == 2:
        return torch.stack(results)
    else:
        return results

def get_skip_circle_inner_mse(
    sample_eigen_expression: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    the_model: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(sample_eigen_expression, (torch.Tensor, np.ndarray)) and sample_eigen_expression.ndim == 1:
        encoded_oh_output = CovariatesEncodingOH(sample_eigen_expression, the_model)
        decoded_dense_output = CovariatesSkipCircularNodeDecodingDense(sample_eigen_expression, the_model)
        return F.mse_loss(encoded_oh_output, decoded_dense_output)
    
    processed_data_list = []
    if isinstance(sample_eigen_expression, np.ndarray) and sample_eigen_expression.ndim == 2:
        for i in range(sample_eigen_expression.shape[1]):
            processed_data_list.append(torch.from_numpy(sample_eigen_expression[:, i]).float())
    elif isinstance(sample_eigen_expression, torch.Tensor) and sample_eigen_expression.ndim == 2:
        for i in range(sample_eigen_expression.shape[1]):
            processed_data_list.append(sample_eigen_expression[:, i].float())
    elif isinstance(sample_eigen_expression, list):
        for item in sample_eigen_expression:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                processed_data_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                processed_data_list.append(item.float())
            elif isinstance(item, List):
                processed_data_list.append(torch.tensor(item).float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray, torch.Tensor or list, but got {type(item)}")
    else:
        my_error(f"Unsupported input type for get_skip_circle_inner_mse: {type(sample_eigen_expression)}")

    results = []
    for sample_tensor in processed_data_list:
        results.append(get_skip_circle_inner_mse(sample_tensor, the_model))
    
    if isinstance(sample_eigen_expression, (torch.Tensor, np.ndarray)) and sample_eigen_expression.ndim == 2:
        return torch.stack(results)
    else:
        return results

def get_out_of_plane_error(
    sample_eigen_expression: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    the_model: Covariates
) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(sample_eigen_expression, (torch.Tensor, np.ndarray)) and sample_eigen_expression.ndim == 1:
        dense_decoded_output = CovariatesThroughDecodingDense(sample_eigen_expression, the_model)
        skip_decoded_output = CovariatesSkipCircularNodeDecodingDense(sample_eigen_expression, the_model)
        return F.mse_loss(dense_decoded_output, skip_decoded_output)
    
    processed_data_list = []
    if isinstance(sample_eigen_expression, np.ndarray) and sample_eigen_expression.ndim == 2:
        for i in range(sample_eigen_expression.shape[1]):
            processed_data_list.append(torch.from_numpy(sample_eigen_expression[:, i]).float())
    elif isinstance(sample_eigen_expression, torch.Tensor) and sample_eigen_expression.ndim == 2:
        for i in range(sample_eigen_expression.shape[1]):
            processed_data_list.append(sample_eigen_expression[:, i].float())
    elif isinstance(sample_eigen_expression, list):
        for item in sample_eigen_expression:
            if isinstance(item, np.ndarray) and item.ndim == 1:
                processed_data_list.append(torch.from_numpy(item).float())
            elif isinstance(item, torch.Tensor) and item.ndim == 1:
                processed_data_list.append(item.float())
            elif isinstance(item, List):
                processed_data_list.append(torch.tensor(item).float())
            else:
                my_error(f"List elements must be 1D numpy.ndarray, torch.Tensor or list, but got {type(item)}")
    else:
        my_error(f"Unsupported input type for get_out_of_plane_error: {type(sample_eigen_expression)}")

    results = []
    for sample_tensor in processed_data_list:
        results.append(get_out_of_plane_error(sample_tensor, the_model))
    
    if isinstance(sample_eigen_expression, (torch.Tensor, np.ndarray)) and sample_eigen_expression.ndim == 2:
        return torch.stack(results)
    else:
        return results

def CovariatesDecoder(
    trained_models_and_errors: List[Any], 
    gea: np.ndarray, 
    OUT_TYPE: type = np.float64
) -> Tuple[Any, np.ndarray]:
    
    losses = np.array([x[-1] for x in trained_models_and_errors], dtype=np.float64)
    losses[np.isnan(losses)] = np.inf
    lowest_loss_index = np.argmin(losses)

    models = [x[0] for x in trained_models_and_errors]
    best_model = models[lowest_loss_index]
    gea = torch.from_numpy(gea).float() if isinstance(gea, np.ndarray) else gea.float()
    eigenspace_magnitudes = sqrtsumofsquares(gea[:best_model.o, :])
    encoding_dense_magnitudes = CovariatesEncodingDenseMagnitude(gea, best_model)
    magnitudes = CovariatesMagnitude(best_model, gea, np.float32)
    decoding_dense_magnitudes = CovariatesDecodingDenseMagnitude(gea, best_model)
    decoding_eigenspace_magnitudes = sqrtsumofsquares(gea, best_model)
    
    decoding_dense_skip_circle_magnitudes = CovariatesSkipCircularNodeDecodingDenseMagnitude(gea, best_model)
    decoding_OH_skip_circle_magnitudes = CovariatesSkipCircularNodeDecodingOHMagnitude(gea, best_model)
    
    projections_list = CovariatesThroughEncodingDense(gea, best_model)
    projections = np.vstack([p.detach().T.numpy() for p in projections_list]).T
    
    phases = CovariatesPhase(gea, best_model)
    
    model_errors = MSELoss(gea, best_model)
    inner_errors = get_inner_mse(gea, best_model)
    circ_errors = get_circ_mse(gea, best_model)
    skip_circle_inner_errors = get_skip_circle_inner_mse(gea, best_model)
    skip_circle_errors = get_skip_circle_mse(gea, best_model)

    out_of_plane_errors = get_out_of_plane_error(gea, best_model)
    out_of_plane_reconstruction_errors = get_out_of_plane_reconstruction_error(gea, best_model)

    arrays_to_hcat = [
        phases,
        model_errors,
        magnitudes,
        projections,
        inner_errors,
        circ_errors,
        skip_circle_inner_errors,
        skip_circle_errors,
        out_of_plane_errors,
        out_of_plane_reconstruction_errors,
        eigenspace_magnitudes,
        encoding_dense_magnitudes,
        decoding_dense_magnitudes,
        decoding_eigenspace_magnitudes,
        decoding_dense_skip_circle_magnitudes,
        decoding_OH_skip_circle_magnitudes
    ]
    processed_arrays = []
    for arr in arrays_to_hcat:
        if isinstance(arr, np.ndarray):
            if arr.ndim == 1:
                processed_arrays.append(arr.reshape(-1, 1))
            else:
                processed_arrays.append(arr)
        elif isinstance(arr, torch.Tensor):
            numpy_arr = arr.detach().numpy()
            if numpy_arr.ndim == 1:
                processed_arrays.append(numpy_arr.reshape(-1, 1))
            else:
                processed_arrays.append(numpy_arr)
    metrics_hcat = np.hstack(processed_arrays)

    return best_model, metrics_hcat

def MetricCovariateConcatenator(
    metrics: pd.DataFrame, 
    options: Dict[str, Any], 
    _verbose: bool = False
) -> Tuple[pd.DataFrame, int]:
    
    if metrics.shape[1] > 12:
        metricNames = [
            "ID", "Phase", "Error", "Magnitude", "ProjectionX", "ProjectionY",
            "Inner_Error", "Circular_Error", "Modified_Inner_Error", "Modified_Error",
            "Out_of_Plane_Error", "Out_of_Plane_Reconstruction_Error", "Input_Magnitude",
            "Dense_Input_Magnitude", "Decoding_Dense_Output_Magnitude",
            "Model_Output_Magnitude", "Modified_Decoding_Dense_Output_Magnitude",
            "Modified_Model_Output_Magnitude"
        ]
    else:
        metricNames = [
            "ID", "Phase", "Error", "Magnitude", "ProjectionX", "ProjectionY",
            "Circular_Error", "Modified_Error", "Out_of_Plane_Reconstruction_Error",
            "Input_Magnitude", "Model_Output_Magnitude", "Modified_Model_Output_Magnitude"
        ]

    metric_covariate_names = []

    are_there_discontinuous_covariates = 'o_dc' in options and options['o_dc'] is not None and len(options['o_dc']) > 0
    are_there_continuous_covariates = 'o_cc' in options and options['o_cc'] is not None and len(options['o_cc']) > 0

    if options.get("o_fxr", 0) >= 2:
        if are_there_discontinuous_covariates and are_there_continuous_covariates:
            if _verbose:
                my_info("\tDISCONTINUOUS AND CONTINUOUS COVARIATES EXIST\n\n")
            metric_covariates = np.vstack((options['o_dc'], options['o_cc'])).T
            
            metric_covariate_names = (
                ["Covariate_D"] * options['o_dc'].shape[0] +
                ["Covariate_C"] * options['o_cc'].shape[0]
            )
        elif are_there_discontinuous_covariates:
            if _verbose:
                my_info("\tONLY DISCONTINUOUS COVARIATES EXIST\n\n")
            metric_covariates = options['o_dc'].T
            metric_covariate_names = ["Covariate_D"] * options['o_dc'].shape[0]
        elif are_there_continuous_covariates:
            if _verbose:
                my_info("\tONLY CONTINUOUS COVARIATES EXIST\n\n")
            metric_covariates = options['o_cc'].T
            metric_covariate_names = ["Covariate_C"] * options['o_cc'].shape[0]
        else:
            metric_covariates = np.empty((metrics.shape[0], 0))
            metric_covariate_names = []

        if _verbose:
            my_info("\tADD COVARIATES TO FIT OUTPUT\n\n")
        if metrics.shape[0] != metric_covariates.shape[0]:
            my_error(f"Number of rows in metrics ({metrics.shape[0]}) does not match number of rows in covariates ({metric_covariates.shape[0]}).")

        cov_df = pd.DataFrame(metric_covariates, index=metrics.index, columns=metric_covariate_names)
        metrics = pd.concat([metrics, cov_df], axis=1)

        if _verbose:
            print("\tADD COVARIATE NAMES TO FIT OUTPUT\n\n")
        metricNames.extend(metric_covariate_names)

    if len(metricNames) != metrics.shape[1]:
        my_warn(f"Number of generated metric names ({len(metricNames)}) does not match number of columns in metrics DataFrame ({metrics.shape[1]}). Adjusting.")
        metricDataframe = pd.DataFrame(metrics.values, columns=pd.unique(metricNames).tolist())
    else:
        metricDataframe = metrics.copy()
        metricDataframe.columns = pd.unique(metricNames).tolist()

    return metricDataframe, len(metric_covariate_names)

def Fit(
    dataFile: pd.DataFrame, 
    genesOfInterest: List[str], 
    alternateOps: Dict[str, Any] = None,
    _verbose: bool = False
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, Covariates, Dict[str, Any]]:
    if alternateOps is None:
        alternateOps = {}

    options = update_DefaultDict(alternateOps)
    
    eigen_data = Eigengenes(dataFile, genesOfInterest, options) 
    
    random.seed(1234)
    np.random.seed(1234)

    initialized_models = InitializeModel(eigen_data, options)

    no_covariates = (eigen_data.shape[0] == options["o_svd_n_dims"])
    
    if _verbose:
        print(f"Number of eigen rows: {eigen_data.shape[0]}")
        print(f"Option o_svd_n_dims: {options.get('o_svd_n_dims')}")
        print(f"No covariates scenario: {no_covariates}")

    if no_covariates:
        if _verbose:
            print("No covariates detected. Proceeding with Order training.")
        trained_models, metrics_array = MultiTrainOrder(initialized_models, eigen_data, options)
        best_model, metrics_array = OrderDecoder(trained_models, eigen_data)
    else:
        if _verbose:
            print("Covariates detected. Proceeding with Covariates training.")
        trained_models_and_errors = MultiTrainCovariates(initialized_models, eigen_data, options)
        best_model, metrics_array = CovariatesDecoder(trained_models_and_errors, eigen_data)
    
    sample_names = dataFile.columns[1:].tolist()
    
    if metrics_array.ndim == 1:
        metrics_array = metrics_array.reshape(1, -1)
    
    if metrics_array.shape[0] != len(sample_names):
        if metrics_array.shape[0] < len(sample_names) and metrics_array.shape[0] == 1:
            if _verbose:
                print(f"Warning: metrics_array shape {metrics_array.shape[0]} doesn't match number of samples {len(sample_names)}. Tiling for demo.")
            metrics_array = np.tile(metrics_array, (len(sample_names), 1))
        else:
            my_error(f"Mismatched metrics_array shape {metrics_array.shape} and number of samples {len(sample_names)}. Cannot proceed.")

    metric_column_names = [f'MetricType{i+1}' for i in range(metrics_array.shape[1])]
    metrics = pd.DataFrame(metrics_array, columns=metric_column_names)
    metrics.insert(0, 'SampleID', sample_names)
    
    metricDataframe, n_cov = MetricCovariateConcatenator(metrics, options, _verbose)
    
    if n_cov > 0:
        cols_for_corr_metrics = metricDataframe.iloc[:, 1:-n_cov].values.astype(np.float32)
    else:
        cols_for_corr_metrics = metricDataframe.iloc[:, 1:].values.astype(np.float32)

    eigengenes_for_cor = eigen_data[:best_model.o, :].T.astype(np.float32)
    combined_data_for_corr = np.hstack((cols_for_corr_metrics, eigengenes_for_cor))
    correlation_matrix = np.corrcoef(combined_data_for_corr, rowvar=False)

    num_metrics_cols = cols_for_corr_metrics.shape[1]
    
    eigengene_metric_pearson_correlations = correlation_matrix[
        :num_metrics_cols, num_metrics_cols:
    ]

    eigengene_col_names = [f"Eigengene{i+1}" for i in range(best_model.o)]
    correlationDataframe_no_row_names = pd.DataFrame(
        eigengene_metric_pearson_correlations,
        columns=eigengene_col_names
    )

    if n_cov > 0:
        metric_names_for_cor = metricDataframe.columns[1:-n_cov].tolist()
    else:
        metric_names_for_cor = metricDataframe.columns[1:].tolist()
    
    correlationDataframe = pd.concat([
        pd.DataFrame({'MetricName': metric_names_for_cor}),
        correlationDataframe_no_row_names
    ], axis=1)

    return eigen_data, metricDataframe, correlationDataframe, best_model, options

def CheckPath_bang(path_to_check: str):
    os.makedirs(path_to_check, exist_ok=True)

def CheckPath(path_to_check: str) -> str:
    if not os.path.isdir(path_to_check):
        os.makedirs(path_to_check)
        return path_to_check
    else:
        new_path = f"{path_to_check}_1"
        return CheckPath(new_path)


def OutputFolders(output_path: str, ops: Dict[str, Any]) -> Tuple[str, List[str]]:
    my_info("\tCREATING OUTPUT FOLDER\n\n")
    CheckPath_bang(output_path)
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    master_output_folder_path_base = os.path.join(output_path, f"CYCLOPS_{todays_date}")
    master_output_folder_path = CheckPath(master_output_folder_path_base)

    my_info(f"\tOUTPUTS WILL BE SAVED IN {master_output_folder_path}\n\n")

    all_subfolder_paths: List[str] = []
    
    for folder_name in subfolders:
        sub_output_folder_path = os.path.join(master_output_folder_path, folder_name)
        CheckPath_bang(sub_output_folder_path)
        all_subfolder_paths.append(sub_output_folder_path)
        
    return todays_date, all_subfolder_paths

def Circular_Mean(phases: Union[np.ndarray, list]) -> float:
    if len(phases) < 1:
        return np.nan
    
    phases_np = np.asarray(phases)

    sinterm = np.sum(np.sin(phases_np))
    costerm = np.sum(np.cos(phases_np))

    return np.mod(np.arctan2(sinterm, costerm), 2 * math.pi)

def plus_minus(value: float, delta: float) -> tuple[float, float]:
    return value + delta, value - delta


def GetLinSSE(eP: Union[pd.Series, np.ndarray], gea: np.ndarray, ops: Dict[str, Any], s: float = 0) -> np.ndarray:
    eP_np = np.asarray(eP)
    x = np.mod(eP_np - s, 2 * math.pi)
    x_col = x.reshape(-1, 1)
    if "o_covariates" in ops and ops["cosine_covariate_offset"]:
        usable_covariates = covariates_0_check(ops)
        b_terms = np.hstack((np.ones((len(eP_np), 1)), usable_covariates))
    else:
        b_terms = np.ones((len(eP_np), 1))
    llsq_terms = np.hstack((x_col, b_terms))
    line_of_best_fit, residuals, rank, s_values = np.linalg.lstsq(llsq_terms, gea.T, rcond=None)
    m_coeffs = line_of_best_fit[0, :]
    b_coeffs_raw = line_of_best_fit[1:, :] 
    b_coeffs = b_coeffs_raw.T
    term1 = np.outer(m_coeffs, x)
    term2 = b_coeffs @ b_terms.T
    predicted_values = term1 + term2
    SSE = np.sum((gea - predicted_values)**2, axis=1, keepdims=True)
    return SSE

def covariates_0_check(
    arg1: Any, 
    arg2: Any = None, 
    arg3: Any = None, 
    arg4: Any = None
) -> np.ndarray:
    def _handle_bool_2d_array_case(onehotmatrix: np.ndarray) -> np.ndarray:
        if onehotmatrix.shape[1] > 1:
            return onehotmatrix[:, 1:].astype(float)
        else:
            return np.zeros((onehotmatrix.shape[0], 0))

    def _handle_str_1d_array_case(covariate_row: np.ndarray) -> np.ndarray:
        encoded_df = pd.get_dummies(covariate_row, dtype=bool)
        return _handle_bool_2d_array_case(encoded_df.values)

    def _handle_str_2d_array_all_cols_case(raw_covariates: np.ndarray) -> np.ndarray:
        processed_cols_results = []
        for col_idx in range(raw_covariates.shape[1]):
            processed_cols_results.append(_handle_str_1d_array_case(raw_covariates[:, col_idx]))
        
        if len(processed_cols_results) > 0:
            return np.hstack(processed_cols_results)
        else:
            return np.zeros((raw_covariates.shape[0], 0))


    if isinstance(arg1, dict) and arg2 is None and arg3 is None and arg4 is None:
        ops = arg1
        out_all_disc_cov_cond = (ops["out_all_disc_cov"] and not (ops.get("o_dco") is None))
        return covariates_0_check(
            ops["o_dc"], 
            out_all_disc_cov_cond or None, 
            ops["out_disc_cov"], 
            len(ops["o_column_ids"])
        )

    if arg1 is None and arg2 is None:
        n_samples = arg4
        return np.zeros((n_samples, 0))

    if isinstance(arg1, np.ndarray) and arg1.ndim == 2 and arg1.dtype == bool and arg2 is None and arg3 is None and arg4 is None:
        return _handle_bool_2d_array_case(arg1)

    if isinstance(arg1, np.ndarray) and arg1.ndim == 1 and arg1.dtype == object and arg2 is None and arg3 is None and arg4 is None:
        return _handle_str_1d_array_case(arg1)
    
    if isinstance(arg1, np.ndarray) and arg1.ndim == 2 and arg1.dtype == object and arg2 is None and arg3 is None and arg4 is None:
        return _handle_str_2d_array_all_cols_case(arg1)

    if isinstance(arg1, np.ndarray) and arg1.ndim == 2 and arg1.dtype == object and \
       (arg2 is None or isinstance(arg2, bool)) and \
       (isinstance(arg3, int) or isinstance(arg3, list)) and \
       isinstance(arg4, int):
        
        raw_covariates = arg1
        use_all = arg2
        which_cov = arg3
        n_samples = arg4
        raw_covariates = raw_covariates.T
        if isinstance(which_cov, int):
            selected_cols = raw_covariates[:, which_cov - 1].reshape(-1, 1)
        else:
            selected_cols_indices = [idx - 1 for idx in which_cov]
            selected_cols = raw_covariates[:, selected_cols_indices]
        
        return _handle_str_2d_array_all_cols_case(selected_cols)

    my_error(f"未找到与提供的参数相匹配的 `covariates_0_check` 签名。参数: {type(arg1), type(arg2), type(arg3), type(arg4)}")

def GetCosLinSSE(eP: Union[np.ndarray, pd.Series], gea_row: np.ndarray, ops: Dict[str, Any], s: float = 0) -> float:
    eP_np = np.asarray(eP)
    lin_x_terms = np.mod(eP_np - s, 2 * math.pi)
    cos_x_terms = np.cos(lin_x_terms)
    sin_x_terms = np.sin(lin_x_terms)
    if "o_covariates" in ops and ops["cosine_covariate_offset"]:
        usable_covariates = covariates_0_check(ops)
        b_terms = np.hstack((np.ones((len(eP_np), 1)), usable_covariates))
    else:
        b_terms = np.ones((len(eP_np), 1))
    gradient_terms = np.hstack((
        lin_x_terms.reshape(-1, 1), 
        sin_x_terms.reshape(-1, 1), 
        cos_x_terms.reshape(-1, 1)
    ))
    llsq_terms = np.hstack((gradient_terms, b_terms))
    line_of_best_fit = np.linalg.lstsq(llsq_terms, gea_row.reshape(-1, 1), rcond=None)[0]
    m_coeffs = line_of_best_fit[0:3].flatten()
    b_coeffs = line_of_best_fit[3:].flatten()
    term1 = gradient_terms @ m_coeffs
    term2 = b_terms @ b_coeffs
    predicted_values = term1 + term2
    SSE = np.sum((gea_row - predicted_values)**2)
    return SSE

def GetCosSSELineAttributes(
    eP: Union[pd.Series, np.ndarray], 
    gea: np.ndarray, 
    ops: Dict[str, Any], 
    some_value: float
) -> Tuple[np.ndarray, pd.DataFrame]:
    num_genes = gea.shape[0]
    dummy_cos_sse = np.random.rand(num_genes, 1) * 10
    
    line_attributes_df = pd.DataFrame({
        'Amplitude': np.random.rand(num_genes),
        'Acrophase': np.random.rand(num_genes) * 2 * math.pi,
        'Offset': np.random.rand(num_genes)
    })
    return dummy_cos_sse, line_attributes_df

def CosineFit(eP: Union[pd.Series, np.ndarray], dataFile: pd.DataFrame, ops: Dict[str, Any]) -> pd.DataFrame:
    eP_np = np.asarray(eP)
    gea = MakeFloat(dataFile.iloc[ops["o_fxr"]:, 1:], float)
    
    lin_range_shifts = np.linspace(0, 2 * math.pi, ops["cosine_shift_iterations"])
    
    all_lin_SSEs_list = [GetLinSSE(eP_np, gea, ops, shift_val) for shift_val in lin_range_shifts]
    all_lin_SSEs = np.hstack(all_lin_SSEs_list)

    LinSSEs = np.min(all_lin_SSEs, axis=1, keepdims=True)
    best_shift_index = np.argmin(all_lin_SSEs, axis=1)
    
    best_shift = lin_range_shifts[best_shift_index]

    gene_rows = [gea[i, :] for i in range(gea.shape[0])]
    
    cosLinSSEs = []
    for each_shift, each_gene in zip(best_shift, gene_rows):
        cosLinSSEs.append(GetCosLinSSE(eP_np, each_gene, ops, each_shift))
    cosLinSSEs = np.array(cosLinSSEs).reshape(-1, 1)
    
    ops["eP_dummy_len"] = len(eP_np) 

    if "o_covariates" in ops and ops["cosine_covariate_offset"]:
        usable_covariates = covariates_0_check(ops)
        full_param_number = 3 + usable_covariates.shape[1]
    else:
        full_param_number = 3
    
    df_denominator = len(eP_np) - full_param_number
    if df_denominator <= 0:
        each_gene_f_statistic = np.full(LinSSEs.shape, np.nan)
    else:
        each_gene_f_statistic = ((LinSSEs - cosLinSSEs) / 2) / ((cosLinSSEs / df_denominator))
    
    nan_f = np.isnan(each_gene_f_statistic)
    each_gene_f_statistic[nan_f] = 0
    
    if df_denominator <= 0:
        my_cdf = np.full(each_gene_f_statistic.shape, np.nan)
    else:
        my_cdf = f.cdf(each_gene_f_statistic, 2.0, df_denominator)
    
    p_statistic = (1 - my_cdf).flatten()
    
    bhq_statistic = multipletests(p_statistic, method='fdr_bh')[1]
    
    bonferroni_statistic = multipletests(p_statistic, method='bonferroni')[1]
    
    cos_SSE, line_attributes = GetCosSSELineAttributes(eP_np, gea, ops, 0.0)
    
    SSE_base = np.sum((gea - np.mean(gea, axis=1, keepdims=True))**2, axis=1, keepdims=True)
    
    r2 = 1 - (cos_SSE / SSE_base)
    r2[nan_f] = 0
    
    gene_symbols_df = pd.DataFrame({'Gene_Symbols': dataFile.iloc[ops["o_fxr"]:, 0].values})
    
    if not isinstance(line_attributes, pd.DataFrame):
        line_attributes = pd.DataFrame(line_attributes)

    stats_df = pd.DataFrame({
        'F_Statistic': each_gene_f_statistic.flatten(),
        'P_Statistic': p_statistic,
        'BHQ_Statistic': bhq_statistic,
        'Bonferroni_Statistic': bonferroni_statistic,
        'R_Squared': r2.flatten()
    })
    
    line_attributes_final = pd.concat([gene_symbols_df, line_attributes, stats_df], axis=1)
    
    return line_attributes_final


def cosShift(
    estimate_list: Union[np.ndarray, List[float]], 
    ideal_list: Union[np.ndarray, List[float]], 
    additional_list: Union[np.ndarray, List[float]], 
    base: str = "radians"
) -> Tuple[np.ndarray, np.ndarray]:
    
    estimate_list_np = np.asarray(estimate_list)
    ideal_list_np = np.asarray(ideal_list)
    additional_list_np = np.asarray(additional_list)

    if base == "hours":
        ideal_radian_list = np.mod(ideal_list_np, 24) * (math.pi / 12)
    elif base == "radians":
        ideal_radian_list = np.mod(ideal_list_np, 2 * math.pi)
    else:
        print("FLAG ERROR")
        return np.array([]), np.array([])

    best_error = 2 * math.pi
    shifted_estimate_list = np.copy(estimate_list_np)
    shifted_additional_list = np.copy(additional_list_np)

    for a in np.linspace(-math.pi, math.pi, 192):
        new_estimate_list = np.mod(estimate_list_np + a, 2 * math.pi)
        current_error = np.mean(1 - np.cos(new_estimate_list - ideal_radian_list))
        if best_error > current_error:
            best_error = np.copy(current_error)
            shifted_estimate_list = np.copy(new_estimate_list)
            shifted_additional_list = np.mod(additional_list_np + a, 2 * math.pi)

    for a in np.linspace(-math.pi, math.pi, 192):
        new_estimate_list = np.mod(-1 * (estimate_list_np + a), 2 * math.pi)
        current_error = np.mean(1 - np.cos(new_estimate_list - ideal_radian_list))
        if best_error > current_error:
            best_error = np.copy(current_error)
            shifted_estimate_list = np.copy(new_estimate_list)
            shifted_additional_list = np.mod(-1 * (additional_list_np + a), 2 * math.pi)

    return shifted_estimate_list, shifted_additional_list


def AlignAcrophases(
    dataFile: pd.DataFrame, 
    Fit_Output: pd.DataFrame, 
    ops: Dict[str, Any], 
    align_genes: List[str], 
    align_acrophases: List[float]
) -> Tuple[pd.Series, pd.DataFrame, List[Union[np.ndarray, float]]]:
    goi_index = findXinY(align_genes, dataFile.iloc[:, 0])
    
    each_gene_how_many_times = [len(findXinY([gene], dataFile.iloc[:, 0])) for gene in align_genes]
    
    rows_to_select = list(range(ops["o_fxr"] - 1)) + goi_index
    goi_dataframe = dataFile.iloc[rows_to_select, :].copy()
    
    all_phases = Fit_Output["Phase"]
    Cosine_output_first = CosineFit(all_phases, goi_dataframe, ops)
    
    p_statistic = Cosine_output_first.P_Statistic
    p_logical = p_statistic < ops["align_p_cutoff"]
    
    all_acrophases = Cosine_output_first.Acrophase
    
    gene_group_upper_bounds = np.cumsum(each_gene_how_many_times)
    gene_group_lower_bounds = np.concatenate(([1], gene_group_upper_bounds[:-1] + 1))
    
    grouped_acrophases = [all_acrophases[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    grouped_p_logical = [p_logical[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    
    grouped_significant_acrophases = [grouped_acrophases[ii][grouped_p_logical[ii]] for ii in range(len(each_gene_how_many_times))]
    
    mean_grouped_significant_acrophases = np.array([Circular_Mean(group) if len(group) > 0 else np.nan for group in grouped_significant_acrophases])
    usable_genes_logical = ~np.isnan(mean_grouped_significant_acrophases)
    
    grouped_p_statistic = [p_statistic[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    grouped_significant_p_statistic = [grouped_p_statistic[ii][grouped_p_logical[ii]] for ii in range(len(each_gene_how_many_times))]
    
    mean_grouped_significant_p_statistic = np.array([np.mean(group) if len(group) > 0 else np.nan for group in grouped_significant_p_statistic])[usable_genes_logical]

    usable_genes = np.array(align_genes)[usable_genes_logical].tolist()
    
    usable_ideal_genes = []
    if align_genes == human_homologue_gene_symbol:
        usable_ideal_genes = np.array(mouse_gene_symbol)[usable_genes_logical].tolist()
    else:
        usable_ideal_genes = usable_genes
    
    usable_ideal_gene_acrophases = np.array(align_acrophases)[usable_genes_logical]
    usable_gene_acrophases = mean_grouped_significant_acrophases[usable_genes_logical]
    
    _, shifted_sample_phases = cosShift(usable_gene_acrophases, usable_ideal_gene_acrophases, Fit_Output["Phase"], ops["align_base"])
    
    shifted_cosine_output = CosineFit(shifted_sample_phases, dataFile, ops)
    Cosine_output = CosineFit(shifted_sample_phases, goi_dataframe, ops)

    p_statistic = Cosine_output.P_Statistic
    p_logical = p_statistic < ops["align_p_cutoff"]

    r_squared = Cosine_output.R_Squared
    
    all_acrophases = Cosine_output.Acrophase
    
    gene_group_upper_bounds = np.cumsum(each_gene_how_many_times)
    gene_group_lower_bounds = np.concatenate(([1], gene_group_upper_bounds[:-1] + 1))
    
    grouped_acrophases = [all_acrophases[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    grouped_p_logical = [p_logical[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    
    grouped_significant_acrophases = [grouped_acrophases[ii][grouped_p_logical[ii]] for ii in range(len(each_gene_how_many_times))]
    mean_grouped_significant_acrophases = np.array([Circular_Mean(group) if len(group) > 0 else np.nan for group in grouped_significant_acrophases])
    usable_genes_logical = ~np.isnan(mean_grouped_significant_acrophases)

    grouped_p_statistic = [p_statistic[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    grouped_significant_p_statistic = [grouped_p_statistic[ii][grouped_p_logical[ii]] for ii in range(len(each_gene_how_many_times))]
    mean_grouped_significant_p_statistic = np.array([np.mean(group) if len(group) > 0 else np.nan for group in grouped_significant_p_statistic])[usable_genes_logical]

    grouped_r_squared = [r_squared[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    grouped_significant_r_squared = [grouped_r_squared[ii][grouped_p_logical[ii]] for ii in range(len(each_gene_how_many_times))]
    mean_grouped_significant_r_squared = np.array([np.mean(group) if len(group) > 0 else np.nan for group in grouped_significant_r_squared])[usable_genes_logical]

    usable_genes = np.array(align_genes)[usable_genes_logical].tolist()
    if np.array_equal(align_genes, human_homologue_gene_symbol):
        usable_ideal_genes = np.array(mouse_gene_symbol)[usable_genes_logical].tolist()
    else:
        usable_ideal_genes = usable_genes
    usable_ideal_gene_acrophases = np.array(align_acrophases)[usable_genes_logical]
    usable_gene_acrophases = mean_grouped_significant_acrophases[usable_genes_logical]

    Acrophase_Plot_Info_Array = [usable_genes, usable_ideal_genes, usable_ideal_gene_acrophases, mean_grouped_significant_r_squared, usable_gene_acrophases, mean_grouped_significant_p_statistic]

    return shifted_sample_phases, shifted_cosine_output, Acrophase_Plot_Info_Array

def AlignSamples(dataFile: pd.DataFrame, Fit_Output: pd.DataFrame, ops: Dict[str, Any], align_samples: List[str], align_phases: List[float]) -> Tuple[pd.Series, pd.DataFrame, List[Union[np.ndarray, float]]]:
    known_sample_indices = findXinY(align_samples, ops["o_column_ids"])
    estimates_for_known_samples = Fit_Output["Phase"].iloc[known_sample_indices].values
    _, shifted_sample_phases_np = cosShift(estimates_for_known_samples, align_phases, Fit_Output["Phase"].values, ops["align_base"])
    shifted_sample_phases = pd.Series(shifted_sample_phases_np, index=Fit_Output.index)
    shifted_Cosine_Sample_output = CosineFit(shifted_sample_phases, dataFile, ops)
    goi_index = findXinY(human_homologue_gene_symbol, dataFile.iloc[ops["o_fxr"]:, 0])
    each_gene_how_many_times = [len(findXinY([gene], dataFile.iloc[:, 0])) for gene in human_homologue_gene_symbol]
    Cosine_output = shifted_Cosine_Sample_output.iloc[goi_index, :].copy()
    p_statistic = Cosine_output["P_Statistic"].values
    p_logical = p_statistic < ops["align_p_cutoff"]
    r_squared = Cosine_output["R_Squared"].values
    all_acrophases = Cosine_output["Acrophase"].values
    gene_group_upper_bounds = np.cumsum(each_gene_how_many_times)
    gene_group_lower_bounds = np.concatenate(([1], gene_group_upper_bounds[:-1] + 1))
    grouped_acrophases = [all_acrophases[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    grouped_p_logical = [p_logical[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    grouped_significant_acrophases = [group_acro[group_p_log] for group_acro, group_p_log in zip(grouped_acrophases, grouped_p_logical)]
    mean_grouped_significant_acrophases = np.array([Circular_Mean(group) if len(group) > 0 else np.nan for group in grouped_significant_acrophases])
    usable_genes_logical = ~np.isnan(mean_grouped_significant_acrophases)
    grouped_p_statistic = [p_statistic[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    grouped_significant_p_statistic = [group_p_stat[group_p_log] for group_p_stat, group_p_log in zip(grouped_p_statistic, grouped_p_logical)]
    mean_grouped_significant_p_statistic = np.array([np.mean(group) if len(group) > 0 else np.nan for group in grouped_significant_p_statistic])[usable_genes_logical]
    grouped_r_squared = [r_squared[gene_group_lower_bounds[ii]-1:gene_group_upper_bounds[ii]] for ii in range(len(each_gene_how_many_times))]
    grouped_significant_r_squared = [group_r_sq[group_p_log] for group_r_sq, group_p_log in zip(grouped_r_squared, grouped_p_logical)]
    mean_grouped_significant_r_squared = np.array([np.mean(group) if len(group) > 0 else np.nan for group in grouped_significant_r_squared])[usable_genes_logical]
    usable_genes = np.array(human_homologue_gene_symbol)[usable_genes_logical].tolist()
    usable_ideal_genes = np.array(mouse_gene_symbol)[usable_genes_logical].tolist()
    usable_ideal_gene_acrophases = np.array(mouse_acrophases)[usable_genes_logical]
    usable_gene_acrophases = mean_grouped_significant_acrophases[usable_genes_logical]
    Acrophase_Plot_Info_Array = [usable_genes, usable_ideal_genes, usable_ideal_gene_acrophases, mean_grouped_significant_r_squared, usable_gene_acrophases, mean_grouped_significant_p_statistic]
    return shifted_sample_phases, shifted_Cosine_Sample_output, Acrophase_Plot_Info_Array


def UniversalModelSaver(model: Any, dir: str = None, name: str = "Model1"):
    if dir is None:
        dir = os.getcwd()

    all_model_field_names_before = list(model.__dict__.keys())
    all_model_field_names = list(model.__dict__.keys())
    all_model_field_values = [getattr(model, field_name) for field_name in all_model_field_names]

    dense_layer_logical = [isinstance(val, nn.Linear) for val in all_model_field_values]
    
    dense_layer_W_and_b = []
    for i, is_dense in enumerate(dense_layer_logical):
        if is_dense:
            layer = all_model_field_values[i]
            dense_layer_W_and_b.append(layer.weight)
            dense_layer_W_and_b.append(layer.bias)

    dense_layer_key_names = []
    for i, is_dense in enumerate(dense_layer_logical):
        if is_dense:
            original_name = all_model_field_names[i]
            dense_layer_key_names.append(f"{original_name}_W")
            dense_layer_key_names.append(f"{original_name}_b")

    all_model_field_names.extend(dense_layer_key_names)
    all_model_field_values.extend(dense_layer_W_and_b)

    model_dict = dict(zip(all_model_field_names, all_model_field_values))

    for i, is_dense in enumerate(dense_layer_logical):
        if is_dense:
            original_name = all_model_field_names_before[i]
            del model_dict[original_name]

    df_to_save = pd.DataFrame([model_dict])
    
    file_path = os.path.join(dir, f"{name}.csv")
    df_to_save.to_csv(file_path, index=False)

def Align(
    dataFile: pd.DataFrame, 
    Fit_Output: pd.DataFrame, 
    Eigengene_Correlation: pd.DataFrame, 
    Model: Any, 
    ops: Dict[str, Any], 
    output_path: str
) -> Tuple[str, Tuple[str, str, str, str]]:
    
    todays_date, all_subfolder_paths = OutputFolders(output_path, ops)
    plot_path_l, fit_path_l, model_path_l, parameter_path_l = all_subfolder_paths
    
    if "align_genes" in ops and "align_acrophases" in ops:
        my_info("ALIGNMENT ACROPHASES FOR GENES OTHER THAN MOUSE ATLAS GENES HAVE BEEN SPECIFIED.")
        align_genes, align_acrophases = ops["align_genes"], ops["align_acrophases"]
        Fit_Output["Phases_AG"], Align_Genes_Cosine_Fit, _ = AlignAcrophases(
            dataFile, 
            Fit_Output, 
            ops, 
            align_genes, 
            align_acrophases
        )
        
        Align_Genes_Cosine_Fit.to_csv(
            os.path.join(fit_path_l, f"Genes_of_Interest_Aligned_Cosine_Fit_{todays_date}.csv"), 
            index=False
        )
        my_info("COSINE FIT SAVED.")
    
    if "align_samples" in ops and "align_phases" in ops:
        my_info("ALIGNMENT PHASES FOR SAMPLES HAVE BEEN SPECIFIED.")
        align_samples, align_phases = ops["align_samples"], ops["align_phases"]
        Fit_Output["Phases_SA"], Align_Samples_Cosine_Fit, _ = AlignSamples(
            dataFile, 
            Fit_Output, 
            ops, 
            align_samples, 
            align_phases
        )
        
        x_max = 0.0
        if ops["align_base"] == "radians":
            x_max = 2 * math.pi
            plt.xticks(
                [0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi], 
                ["0", r'$\frac{\pi}{2}$', "π", r'$\frac{3\pi}{2}$', "2π"]
            )
        elif ops["align_base"] == "hours":
            x_max = 24
            plt.xticks([0, 6, 12, 18, 24], ["0", "6", "12", "18", "24"])
        
        plt.ylabel("Predicted Phases (radians)")
        plt.yticks(
            [0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi], 
            ["0", r'$\frac{\pi}{2}$', "π", r'$\frac{3\pi}{2}$', "2π"]
        )
        plt.axis([0, x_max, 0, 2 * math.pi])
        plt.grid(True)
        
        known_sample_indices = findXinY(ops["align_samples"], Fit_Output["ID"])
        plt.scatter(ops["align_phases"], Fit_Output["Phases_SA"].iloc[known_sample_indices], s=22)
        
        my_info("SAVING FIGURE")
        plt.savefig(
            os.path.join(plot_path_l, f"Sample_Phases_Compared_To_Predicted_Phases_Plot_{todays_date}.png"), 
            bbox_inches="tight", dpi=300
        )
        my_info("FIGURE SAVED. CLOSING FIGURE.")
        plt.close()
        my_info("FIGURE CLOSED. SAVING COSINE FIT.")
        Align_Samples_Cosine_Fit.to_csv(
            os.path.join(fit_path_l, f"Sample_Phase_Aligned_Cosine_Fit_{todays_date}.csv"), 
            index=False
        )
        my_info("COSINE FIT SAVED.")
    
    my_info("ALIGNMENT TO MOUSE ATLAS ACROPHASES.")
    align_genes, align_acrophases = human_homologue_gene_symbol, mouse_acrophases
    
    if len(findXinY(align_genes, dataFile.iloc[:, 0])) > 0:
        Fit_Output["Phases_MA"], Align_Genes_Cosine_Fit, _ = AlignAcrophases(
            dataFile, 
            Fit_Output, 
            ops, 
            align_genes, 
            align_acrophases
        )
        
        Align_Genes_Cosine_Fit.to_csv(
            os.path.join(fit_path_l, f"Mouse_Atlas_Aligned_Cosine_Fit_{todays_date}.csv"), 
            index=False
        )
        my_info("COSINE FIT SAVED. SAVING FIT OUTPUT.")
    
    Fit_Output.to_csv(os.path.join(fit_path_l, f"Fit_Output_{todays_date}.csv"), index=False)
    my_info("FIT OUTPUT SAVED. SAVING METRIC CORRELATIONS TO EIGENGENES.")
    Eigengene_Correlation.to_csv(
        os.path.join(fit_path_l, f"Metric_Correlation_to_Eigengenes_{todays_date}.csv"), 
        index=False
    )
    my_info("METRIC CORRELATIONS TO EIGENGENES SAVED. SAVING TRAINING PARAMETERS.")

    csv_compatible_ops = {}
    for k, v in ops.items():
        if isinstance(v, np.ndarray):
            csv_compatible_ops[k] = repr(v.tolist())
        elif isinstance(v, list):
            temp_list = []
            for item in v:
                if isinstance(item, np.ndarray):
                    temp_list.append(repr(item.tolist()))
                else:
                    temp_list.append(item)
            csv_compatible_ops[k] = temp_list
        else:
            csv_compatible_ops[k] = v

    ops_df = pd.DataFrame([csv_compatible_ops])

    csv_output_path = os.path.join(parameter_path_l, f"Trained_Parameter_Dictionary_{todays_date}.csv")

    ops_df.to_csv(csv_output_path, index=False)

    my_info("TRAINING PARAMETERS SAVED TO CSV.")

    UniversalModelSaver(Model, dir=model_path_l, name=f"Trained_Model_{todays_date}")

    return todays_date, all_subfolder_paths