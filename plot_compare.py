import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def compare_csv_files(csv_file_1, csv_file_2, save_path):
    def normalize_id(id_val):
        return re.sub(r'([_.])\d+$', '', str(id_val))

    df1 = pd.read_csv(csv_file_1)
    df2 = pd.read_csv(csv_file_2)

    df1["ID_norm"] = df1["ID"].apply(normalize_id)
    df2["ID_norm"] = df2["ID"].apply(normalize_id)

    merged = pd.merge(
        df1, df2,
        on="ID_norm",
        suffixes=('_1', '_2')
    )

    x = merged["Phase_1"]
    y = merged["Phase_2"]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.7)
    
    file_name_1 = os.path.basename(csv_file_1).replace('.csv', '')
    file_name_2 = os.path.basename(csv_file_2).replace('.csv', '')
    
    plt.xlabel(f"Phase {file_name_1}")
    plt.ylabel(f"Phase {file_name_2}")
    plt.grid(True)
    plt.axis("equal")

    plt.xlim(0, 2 * np.pi)
    plt.ylim(0, 2 * np.pi)
    xticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    xtick_labels = ["0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$", r"$2\pi$"]
    plt.xticks(xticks, xtick_labels)
    plt.yticks(xticks, xtick_labels)

    save_path_file = f"{save_path}/{file_name_1}_{file_name_2}_phase_comparison.png"
    os.makedirs(os.path.dirname(save_path_file), exist_ok=True)

    plt.savefig(save_path_file)
    plt.close()
    print(f"Figure saved in {save_path_file}")
    print(f"Compared {len(merged)} samples based on ID matching")


def compare(path_1, path_2, save_path):
    def find_first_fit_csv(base_path):
        search_pattern = os.path.join("result", base_path, "phase_predictions.csv")
        files = glob.glob(search_pattern, recursive=True)
        if not files:
            raise FileNotFoundError(f"No phase_predictions.csv files found in {search_pattern}")
        return files[0]

    def normalize_id(id_val):
        return re.sub(r'([.])\d+$', '', str(id_val))

    file_path_1 = find_first_fit_csv(path_1)
    file_path_2 = find_first_fit_csv(path_2)
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    df1["ID_norm"] = df1["Sample_ID"].apply(normalize_id)
    df2["ID_norm"] = df2["Sample_ID"].apply(normalize_id)

    merged = pd.merge(
        df1, df2,
        left_on=["ID_norm", "Cell_Type"],
        right_on=["ID_norm", "Cell_Type"],
        suffixes=('_1', '_2')
    )

    x = merged["Predicted_Phase_Radians_1"]
    y = merged["Predicted_Phase_Radians_2"]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel(f"Phase {path_1}")
    plt.ylabel(f"Phase {path_2}")
    plt.grid(True)
    plt.axis("equal")

    plt.xlim(0, 2 * np.pi)
    plt.ylim(0, 2 * np.pi)
    xticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    xtick_labels = ["0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$", r"$2\pi$"]
    plt.xticks(xticks, xtick_labels)
    plt.yticks(xticks, xtick_labels)

    save_path_file = f"{save_path}/{path_1}_{path_2}_phase_comparison.png"
    os.makedirs(os.path.dirname(save_path_file), exist_ok=True)

    plt.savefig(save_path_file)
    print(f"figure saved in {save_path_file}")

path_1 = "Zhang_CancerCell_2025.Sample_MajorCluster"
path_2 = "Zhang_CancerCell_2025.Sample_SubCluster_CD4_CD8"
save_path = "plot"

if __name__ == "__main__":
    # Example usage of the new function
    # compare_csv_files(file_1, file_2, save_path)
    
    # Original function still available
    compare(path_1, path_2, save_path)