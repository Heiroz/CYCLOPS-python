import pandas as pd

def get_seed_genes(expression_csv, output_txt, skip_covariates=True):
    df = pd.read_csv(expression_csv, sep=None, engine='python')
    gene_symbols = df['Gene_Symbol']

    if skip_covariates:
        gene_symbols = gene_symbols.astype(str)
        mask = ~(gene_symbols.str.endswith('_D') | gene_symbols.str.endswith('_C'))
        gene_symbols = gene_symbols[mask]

    gene_symbols = gene_symbols.dropna()

    gene_symbols = gene_symbols.drop_duplicates()

    gene_symbols.to_csv(output_txt, index=False, header=False)

dataset_path = "Kumar_Nature.Sample_MajorCluster"
if __name__ == "__main__":
    csv_path = f"{dataset_path}/expression.csv"
    output_path = f"{dataset_path}/seed_genes.txt"
    get_seed_genes(csv_path, output_path, skip_covariates=True)