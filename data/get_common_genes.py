import pandas as pd

def get_common_seed_genes(expression_csv1, expression_csv2, output_txt, skip_covariates=True):
    def extract_genes(expression_csv):
        df = pd.read_csv(expression_csv, sep=None, engine='python')
        df['Gene_Symbol'] = df['Gene_Symbol'].astype(str).str.upper()  # 转为大写
        # 保存修改后的文件，文件名加后缀 _UPPER.csv
        new_csv = expression_csv.replace('.csv', '_UPPER.csv')
        df.to_csv(new_csv, index=False)
        gene_symbols = df['Gene_Symbol']
        if skip_covariates:
            mask = ~(gene_symbols.str.endswith('_D') | gene_symbols.str.endswith('_C'))
            gene_symbols = gene_symbols[mask]
        gene_symbols = gene_symbols.dropna().drop_duplicates()
        return set(gene_symbols)

    genes1 = extract_genes(expression_csv1)
    genes2 = extract_genes(expression_csv2)

    common_genes = genes1.intersection(genes2)

    pd.Series(list(common_genes)).to_csv(output_txt, index=False, header=False)

# 用法示例
get_common_seed_genes("rna5.Subclass_TimePoint/expression.csv", 
                      "Zhang_CancerCell_2025.Sample_MajorCluster/expression.csv", 
                      "seed_genes.txt")
