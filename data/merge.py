import pandas as pd

def merge_expression_files(file1, file2, output_file):

    df1 = pd.read_csv(file1, sep=None, engine='python')
    df2 = pd.read_csv(file2, sep=None, engine='python')

    common_genes = set(df1['Gene_Symbol']).intersection(set(df2['Gene_Symbol']))

    df1_filtered = df1[df1['Gene_Symbol'].isin(common_genes)]
    df2_filtered = df2[df2['Gene_Symbol'].isin(common_genes)]

    merged_df = pd.merge(df1_filtered, df2_filtered, on='Gene_Symbol')

    merged_df.to_csv(output_file, index=False)


csv_path_1 = "Zhang_CancerCell_2025.Sample_MajorCluster/expression.csv"
csv_path_2 = "Zhang_CancerCell_2025.Sample_SubCluster/expression.csv"
merged_csv_path = f"{csv_path_1}_{csv_path_2}_merged_expression.csv"

if __name__ == "__main__":
    print(f"Merging {csv_path_1} and {csv_path_2} into {merged_csv_path}")
    merge_expression_files(csv_path_1, csv_path_2, merged_csv_path)
    print("Merging completed successfully.")