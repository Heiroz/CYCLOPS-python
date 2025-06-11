import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt


def compare(path_1, path_2, save_path):
    """
    Compare two CSV files by merging them on 'ID' (ignoring trailing _{number}) and 'Covariate_D',
    and plotting the 'Phase' values from both files.
    """
    def find_first_fit_csv(base_path):
        search_pattern = os.path.join(base_path, "Fits", "Fit*.csv")
        files = glob.glob(search_pattern, recursive=True)
        if not files:
            raise FileNotFoundError(f"No Fit*.csv files found in {search_pattern}")
        return files[0]

    def normalize_id(id_val):
        return re.sub(r'_\d+$', '', str(id_val))

    file_path_1 = find_first_fit_csv(path_1)
    file_path_2 = find_first_fit_csv(path_2)
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    df1["ID_norm"] = df1["ID"].apply(normalize_id)
    df2["ID_norm"] = df2["ID"].apply(normalize_id)

    merged = pd.merge(
        df1, df2,
        left_on=["ID_norm", "Covariate_D"],
        right_on=["ID_norm", "Covariate_D"],
        suffixes=('_1', '_2')
    )

    x = merged["Phase_1"]
    y = merged["Phase_2"]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel(f"Phase {path_1}")
    plt.ylabel(f"Phase {path_2}")
    plt.grid(True)
    plt.axis("equal")

    save_path_file = f"{save_path}/{path_1}_{path_2}_phase_comparison.png"
    os.makedirs(os.path.dirname(save_path_file), exist_ok=True)

    plt.savefig(save_path_file)

path_1 = "Zhang_CancerCell_2025.Sample_MajorCluster"
path_2 = "Zhang_CancerCell_2025.Sample_SubCluster"
save_path = "plot"

if __name__ == "__main__":
    compare(path_1, path_2, save_path)