import csv
from collections import defaultdict
import os
import glob

def find_first_fit_csv(base_path):
    search_pattern = os.path.join(base_path, "Fits", "Fit_*.csv")
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No Fit_*.csv files found in {search_pattern}")
    return files[0]

def split_by_celltype(base_path, celltype_col, output_dir):
    input_csv = find_first_fit_csv(base_path)
    print(f"找到输入文件: {input_csv}")

    with open(input_csv, newline='') as f:
        reader = list(csv.DictReader(f))
        header = reader[0].keys()

    celltype_dict = defaultdict(list)
    for row in reader:
        celltype = row[celltype_col]
        celltype_dict[celltype].append(row)

    os.makedirs(output_dir, exist_ok=True)

    for celltype, rows in celltype_dict.items():
        outname = os.path.join(output_dir, f"Fit_Output_{celltype}.csv")
        with open(outname, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"写入文件: {outname}")

if __name__ == "__main__":
    base_path = "Filtered_HTAN_HTAPP"
    celltype_col = "Covariate_D"
    output_dir = f"{base_path}/Fit_celltype"

    split_by_celltype(base_path, celltype_col, output_dir)