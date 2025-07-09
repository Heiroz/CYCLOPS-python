import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np


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
        # Remove trailing _{number} or .{number} at the end of the string
        return re.sub(r'([_.])\d+$', '', str(id_val))

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

    # 设置坐标范围为 [0, 2π] 并调整刻度
    plt.xlim(0, 2 * np.pi)
    plt.ylim(0, 2 * np.pi)
    xticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    xtick_labels = ["0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$", r"$2\pi$"]
    plt.xticks(xticks, xtick_labels)
    plt.yticks(xticks, xtick_labels)

    save_path_file = f"{save_path}/{path_1}_{path_2}_phase_comparison.png"
    os.makedirs(os.path.dirname(save_path_file), exist_ok=True)

    plt.savefig(save_path_file)
    print(f"图像已保存到 {save_path_file}")


path_1 = "Murrow_Cell_systems_2022.mean"
path_2 = "Murrow_Cell_systems_2022.and.ZhangYY_CancerCell.pseudobulk"
save_path = "plot"

if __name__ == "__main__":
    compare(path_1, path_2, save_path)