import pandas as pd

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


csv_path = "Zhang_CancerCell_2025.Sample_SubCluster/expression.csv"
saved_path = "Zhang_CancerCell_2025.Sample_SubCluster/expression_hat.csv"
if __name__ == "__main__":

    df = pd.read_csv(csv_path)
    # df = keep_gene(df, keep_genes=["CLOCK", "NFIL3", "ARNTL", "NPAS2", "NR1D1", "CRY1", "CRY2", "PER1", "PER2", "PER3"])
    df = keep_sample(df, keep_samples=['P018_Post', 'P037_Pre', 'P025_Pre', 'P051_Pre', 'P023_Post', 'P039_Post', 'P045_Pre', 'P034_Post', 'P016_Pre', 'P045_Post', 'P019_Post', 'P060_Post', 'P016_Post', 'P047_Post', 'P057_Pre', 'P002_Post', 'P056_Post', 'P040_Pre', 'P012_Post', 'P052_Pre', 'P017_Pre', 'P018_Pre', 'P053_Pre', 'P049_Pre', 'P010_Pre', 'P059_Pre', 'P039_Pre', 'P058_Pre', 'P062_Pre', 'P061_Pre', 'P007_Pre', 'P031_Pre', 'P038_Pre', 'P040_Post', 'P054_Post', 'P019_Pre', 'P017_Post', 'P058_Post', 'P055_Pre', 'P032_Pre', 'P051_Post', 'P023_Pre', 'P002_Pre', 'P040-L_Pre', 'P044_Pre', 'P060_Pre', 'P047_Pre', 'P059_Post', 'P057_Post', 'P035_Pre', 'P046_Pre', 'P013_Pre', 'P037_Post', 'P038_Post', 'P003_Post', 'P025_Post', 'P043_Pre', 'P034_Pre', 'P042_Post', 'P013_Post', 'P041_Pre', 'P046_Post', 'P012_Pre', 'P042_Pre', 'P020_Pre', 'P052_Post', 'P005_Post', 'P041_Post', 'P005_Pre', 'P049_Post', 'P004_Pre', 'P022_Pre', 'P022_Post', 'P040-L_Post', 'P054_Pre', 'P053_Post', 'P056_Pre', 'P020_Post', 'P003_Pre', 'P021_Pre', 'P021_Post', 'P006_Pre', 'P006_Post', 'P035_Post', 'P048_Pre', 'P048_Post', 'P011_Pre', 'P011_Post', 'P024_Pre', 'P024_Post', 'P026_Pre', 'P026_Post', 'P027_Pre', 'P027_Post', 'P028_Pre', 'P028_Post', 'P029_Pre', 'P029_Post'])
    df.to_csv(saved_path, index=False)