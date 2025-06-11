import csv

def remove_single_occurrence_columns_from_csv(input_csv, output_csv, covariate_row_name):
    with open(input_csv, newline='') as f:
        reader = list(csv.reader(f))
    covariate_idx = None
    for i, row in enumerate(reader):
        if row[0] == covariate_row_name:
            covariate_idx = i
            break
    if covariate_idx is None:
        raise ValueError(f"未找到协变量行: {covariate_row_name}")

    covariate_row = reader[covariate_idx][1:]
    from collections import Counter
    counts = Counter(covariate_row)
    valid_types = {k for k, v in counts.items() if v > 1}
    keep_indices = [0] + [i+1 for i, v in enumerate(covariate_row) if v in valid_types]

    new_reader = []
    for row in reader:
        new_row = [row[i] for i in keep_indices]
        new_reader.append(new_row)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_reader)

if __name__ == "__main__":
    input_csv = "expression.csv"
    output_csv = "expression.csv"
    covariate_row_name = "celltype_D"
    remove_single_occurrence_columns_from_csv(input_csv, output_csv, covariate_row_name)