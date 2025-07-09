import os
import pandas as pd
from typing import Dict, Any
import CYCLOPS

base_path = "/home/xuzhen/CYCLOPS-2.0"
data_path = os.path.join(base_path, "data")
dataset_path = "rna5.Subclass_TimePoint"
path_to_cyclops = os.path.join(base_path, "CYCLOPS.jl")
output_path = os.path.join(base_path, "output")

os.makedirs(output_path, exist_ok=True)

expression_data = pd.read_csv(os.path.join(data_path, dataset_path, "filtered_expression.csv"), low_memory=False)
with open(os.path.join(data_path, dataset_path, "seed_genes.txt"), 'r') as f:
    seed_genes = [line.strip() for line in f if line.strip()]

sample_ids_with_collection_times = ["Sample_6"]
sample_collection_times = [0.0]

if (len(sample_ids_with_collection_times) > 0 or len(sample_collection_times) > 0) and \
   (len(sample_ids_with_collection_times) != len(sample_collection_times)):
    raise ValueError("ATTENTION REQUIRED! Number of sample ids provided ('sample_ids_with_collection_times') must match number of collection times ('sample_collection_times').")

training_parameters: Dict[str, Any] = {
    "regex_cont": r".*_C",
    "regex_disc": r".*_D",
    "blunt_percent": 0.975,
    "seed_min_CV": 0.14,
    "seed_max_CV": 0.9,
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
    "eigen_contr_var": 0.05,
    "eigen_var_override": True,
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
    "train_n_models": 2,
    "train_μA": 0.001,
    "train_β": (0.9, 0.999),
    "train_min_steps": 1500,
    "train_max_steps": 2050,
    "train_μA_scale_lim": 1000,
    "train_circular": False,
    "train_collection_times": True,
    "train_collection_time_balance": 1.0,
    "cosine_shift_iterations": 192,
    "cosine_covariate_offset": True,
    "align_p_cutoff": 0.05,
    "align_base": "radians",
    "align_disc": False,
    "align_disc_cov": 1,
    "align_other_covariates": False,
    "align_batch_only": False,
    "align_samples": sample_ids_with_collection_times,
    "align_phases": sample_collection_times,
    "X_Val_k": 10,
    "X_Val_omit_size": 0.1,
    "plot_use_o_cov": True,
    "plot_correct_batches": True,
    "plot_disc": False,
    "plot_disc_cov": 1,
    "plot_separate": False,
    "plot_color": ["b", "orange", "g", "r", "m", "y", "k"],
    "plot_only_color": True,
    "plot_p_cutoff": 0.05
}

rorc_index = [i for i, gene in enumerate(CYCLOPS.human_homologue_gene_symbol) if gene == "RORC"]
filtered_human_genes = [gene for i, gene in enumerate(CYCLOPS.human_homologue_gene_symbol) if i not in rorc_index]
filtered_mouse_acrophases = [acrophase for i, acrophase in enumerate(CYCLOPS.mouse_acrophases) if i not in rorc_index]

training_parameters["align_genes"] = filtered_human_genes
training_parameters["align_acrophases"] = filtered_mouse_acrophases

eigendata, modeloutputs, correlations, bestmodel, parameters = CYCLOPS.Fit(expression_data, seed_genes, training_parameters)
CYCLOPS.Align(expression_data, modeloutputs, correlations, bestmodel, parameters, output_path)