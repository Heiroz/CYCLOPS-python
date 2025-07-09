import pandas as pd

def filter_genes(expression_path, genes_path, output_path):
    """
    过滤expression.csv文件，只保留在seed_genes.txt中出现的gene_symbol，并保留celltype_D这一行。

    Args:
        expression_path (str): 表达数据文件路径。
        genes_path (str): 基因列表文件路径（无列名的txt文件，每行一个基因符号）。
        output_path (str): 输出过滤后的文件路径。
    """
    # 读取表达数据
    expression_df = pd.read_csv(expression_path)

    # 读取基因列表（无列名，每行一个基因符号）
    with open(genes_path, 'r') as f:
        gene_symbols = f.read().splitlines()

    # 过滤表达数据，只保留基因符号匹配的行，或Gene_Symbol为celltype_D的行
    filtered_df = expression_df[
        (expression_df["Gene_Symbol"].isin(gene_symbols)) |
        (expression_df["Gene_Symbol"] == "celltype_D") |
        (expression_df["Gene_Symbol"] == "time_C")
    ]

    # 保存过滤后的数据
    filtered_df.to_csv(output_path, index=False)
    print(f"Filtered data saved to {output_path}")

# 示例调用
path = "rna5.Subclass_TimePoint"
expression_path = f"{path}/expression.csv"
genes_path = f"{path}/seed_genes.txt"
output_path = f"{path}/filtered_expression.csv"

filter_genes(expression_path, genes_path, output_path)