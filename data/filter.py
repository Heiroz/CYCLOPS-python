import pandas as pd
import os
from os.path import exists

def keep_sample(df: pd.DataFrame, keep_samples: list = None):
    if keep_samples:
        # Always keep the 'Gene_Symbol' column if present
        columns_to_keep = ['Gene_Symbol'] if 'Gene_Symbol' in df.columns else []
        columns_to_keep += keep_samples
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        df = df[columns_to_keep]
    return df

def keep_gene(df: pd.DataFrame, keep_genes: list = None):
    # 应该保留celltype_D这一行！
    if keep_genes and "celltype_D" not in keep_genes:
        keep_genes = keep_genes + ["celltype_D"]

    if keep_genes:
        df = df[df["Gene_Symbol"].isin(keep_genes)]
    return df

def delete_sample(df: pd.DataFrame, delete_samples: list = None):
    if delete_samples:
        df = df.drop(columns=delete_samples, errors='ignore')
    return df

def delete_gene(df: pd.DataFrame, delete_genes: list = None):
    if delete_genes:
        df = df[~df["Gene_Symbol"].isin(delete_genes)]
    return df

def keep_celltype(df: pd.DataFrame, celltypes: list):
    """
    保留celltype_D行为指定celltype列表的列（除了Gene_Symbol）
    """
    if df.shape[0] < 2:
        return df
    base_cols = ['Gene_Symbol'] if 'Gene_Symbol' in df.columns else []
    celltype_row = df.iloc[0]
    celltype_cols = [col for col in df.columns[1:] if celltype_row[col] in celltypes]
    cols = base_cols + celltype_cols
    cols = [col for col in cols if col in df.columns]
    return df[cols]

def delete_celltype(df: pd.DataFrame, celltypes: list):
    """
    删除celltype_D行为指定celltype列表的列（除了Gene_Symbol）
    """
    if df.shape[0] < 2:
        return df
    celltype_row = df.iloc[0]
    celltype_cols = [col for col in df.columns[1:] if celltype_row[col] in celltypes]
    celltype_cols = [col for col in celltype_cols if col != 'Gene_Symbol']
    return df.drop(columns=celltype_cols, errors='ignore')


csv_path = "rna5.Subclass_TimePoint/filtered_expression.csv"
saved_path = "rna5.Subclass_TimePoint/filtered_expression.csv"
if __name__ == "__main__":
    if not exists(saved_path):
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    df = pd.read_csv(csv_path, low_memory=False)
    df = keep_celltype(df, celltypes=['CD4Tcell', 'Bcell', 'Macro_Mono', 'CD8Tcell', 'NKcell'])
    df.to_csv(saved_path, index=False)