import pandas as pd
import os

path = "."

def preprocess_for_cyclops(input_csv, output_csv):
    input_path = os.path.join(path, input_csv)

    df = pd.read_csv(input_path, sep=None, engine='python')

    df.columns = ['Gene_Symbol'] + list(df.columns[1:])

    sample_names = df.columns[1:]
    # 分离出celltype_D，同时去除sample名中的.celltype_D部分
    celltypes = [s.split('.')[-1] for s in sample_names]
    sample_names_clean = [s.rsplit('.', 1)[0] if '.' in s else s for s in sample_names]

    # 构造celltype_D行
    celltype_row = ['celltype_D'] + celltypes

    # 替换列名为Gene_Symbol + 处理后的sample名
    df.columns = ['Gene_Symbol'] + sample_names_clean

    # 拼接celltype_D行和原始数据
    df_str = df.astype(str)
    new_df = pd.DataFrame([celltype_row], columns=df.columns)
    new_df = pd.concat([new_df, df_str], ignore_index=True)

    # 保存为expression.csv
    new_df.to_csv(os.path.join(path, "expression.csv"), index=False)

preprocess_for_cyclops("HTAN_HTAPP_Frozen/HTAN_HTAPP_Frozen.pseudobulk_matrix.tsv", "expression.csv")