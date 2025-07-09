import pandas as pd

path = "pseudobulk_PFC.projid_CellClass_log1p/expression.csv"

import pandas as pd

def filter_celltypes_by_occurrence(df, n):
    celltypes = df.iloc[0, 1:]
    valid_celltypes = celltypes.value_counts()[celltypes.value_counts() >= n].index
    print(f"符合条件的 celltypes: {valid_celltypes.tolist()}")

    valid_columns = ['Gene_Symbol'] + list(df.columns[1:][celltypes.isin(valid_celltypes)])

    filtered_df = df.loc[:, valid_columns]

    return filtered_df

def check_missing_values(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    return missing_values

if __name__ == "__main__":
    missing_values = check_missing_values(path)
    if not missing_values.empty:
        print("Missing values found in the following columns:")
        print(missing_values)
    else:
        print("No missing values found in the dataset.")

    df = pd.read_csv(path, low_memory=False)
    # df = df[~df['Gene_Symbol'].str.match(r'^\d', na=False)]
    df = df[~df['Gene_Symbol'].isna()]
    df = filter_celltypes_by_occurrence(df, 2)
    df.to_csv(path, index=False)
    print("Rows with invalid Gene_Symbol values have been removed and saved to the file.")
    print("Columns corresponding to celltypes appearing only once have been removed.")
