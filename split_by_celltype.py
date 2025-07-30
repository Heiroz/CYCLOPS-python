import csv
from collections import defaultdict
import os
import glob

def find_first_fit_csv(base_path):
    search_pattern = os.path.join("result", base_path, "phase_predictions.csv")
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No phase_predictions.csv files found in {search_pattern}")
    return files[0]

def process_id(original_id):
    parts = original_id.split(".")
    if len(parts) > 1:
        return ".".join(parts[:-1])
    return original_id

def split_by_celltype(base_path, celltype_col, output_dir, id_col="ID", default_celltype="Unknown"):
    input_csv = find_first_fit_csv(base_path)
    print(f"Found input file: {input_csv}")

    with open(input_csv, newline='') as f:
        reader = list(csv.DictReader(f))
        header = list(reader[0].keys())

    has_celltype_col = celltype_col in header
    if not has_celltype_col:
        print(f"Warning: Column '{celltype_col}' not found, treating all data as one celltype: '{default_celltype}'")

    celltype_dict = defaultdict(list)
    processed_count = 0
    id_conversion_count = 0
    
    for row in reader:
        if has_celltype_col:
            celltype = row[celltype_col]
        else:
            celltype = default_celltype
        
        if id_col in row:
            original_id = row[id_col]
            processed_id = process_id(original_id)
            
            if original_id != processed_id:
                print(f"ID conversion: {original_id} -> {processed_id}")
                row[id_col] = processed_id
                id_conversion_count += 1
        
        celltype_dict[celltype].append(row)
        processed_count += 1

    print(f"Processed {processed_count} records")
    print(f"Converted {id_conversion_count} IDs")
    
    if has_celltype_col:
        print(f"Found {len(celltype_dict)} different celltypes based on column '{celltype_col}'")
    else:
        print(f"All data grouped as one celltype: '{default_celltype}'")

    os.makedirs(output_dir, exist_ok=True)

    for celltype, rows in celltype_dict.items():
        outname = os.path.join(output_dir, f"Fit_Output_{celltype}.csv")
        with open(outname, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Written file: {outname} ({len(rows)} records)")

def split_by_celltype_with_custom_id_col(base_path, celltype_col, output_dir, id_col="ID"):
    split_by_celltype(base_path, celltype_col, output_dir, id_col)

if __name__ == "__main__":
    base_path = "Zhang_CancerCell_2025.Sample_SubCluster"
    celltype_col = "Cell_Type"
    output_dir = f"result/{base_path}/Fit_celltype"
    id_col = "Sample_ID"
    default_celltype = "AllCells"

    split_by_celltype(base_path, celltype_col, output_dir, id_col, default_celltype)