import pandas as pd

def rename_celltype_row(expression_path, output_path, rename_dict):
    """
    对expression.csv文件中以celltype_D开头的行进行更名。

    Args:
        expression_path (str): 表达数据文件路径。
        output_path (str): 输出文件路径。
        rename_dict (dict): 更名映射字典，键为原celltype名，值为新celltype名。
    """
    # 读取表达数据
    expression_df = pd.read_csv(expression_path, header=None)

    # 找到以celltype_D开头的行
    celltype_row_index = expression_df[0].eq("celltype_D").idxmax()
    if celltype_row_index == 0 and expression_df.iloc[0, 0] != "celltype_D":
        raise ValueError("文件中没有找到以 'celltype_D' 开头的行")

    # 获取该行的celltype名并进行更名
    celltypes = expression_df.iloc[celltype_row_index, 1:]
    renamed_celltypes = celltypes.replace(rename_dict)

    # 更新数据
    expression_df.iloc[celltype_row_index, 1:] = renamed_celltypes

    # 保存更名后的数据
    expression_df.to_csv(output_path, index=False, header=False)
    print(f"Renamed celltypes saved to {output_path}")

# 示例调用
expression_path = "rna5.Subclass_TimePoint/filtered_expression.csv"
output_path = "rna5.Subclass_TimePoint/filtered_expression.csv"
rename_dict = {
    # "B": "Bcell",
    "CD4_T": "CD4Tcell",
    # "CD8_T": "CD8Tcell",
    # "NKT": "NKcell"
}

rename_celltype_row(expression_path, output_path, rename_dict)