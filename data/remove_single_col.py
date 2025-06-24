import pandas as pd
from collections import Counter

def remove_single_occurrence_celltype(input_path, output_path=None):
    df = pd.read_csv(input_path, header=0, low_memory=False)
    celltype_row = df.iloc[0]
    samples = df.columns[1:]
    celltype_values = celltype_row[samples]
    celltype_counts = Counter(celltype_values)
    single_occurrence_celltypes = [
        celltype for celltype, count in celltype_counts.items()
        if count == 1 and celltype != "celltype_D"
    ]
    columns_to_delete = [
        sample for sample in samples
        if celltype_values[sample] in single_occurrence_celltypes
    ]
    df = df.drop(columns=columns_to_delete)
    if output_path is None:
        output_path = input_path
    df.to_csv(output_path, index=False)

remove_single_occurrence_celltype(
    "HTAN_HTAPP/expression.csv"
)