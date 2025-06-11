import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

# 定义基因名
gene = "CLOCK"

# 遍历当前目录下的所有 Fit_Output_*.csv 文件
for file_name in os.listdir("."):
    if file_name.startswith("Fit_Output_") and file_name.endswith(".csv"):
        print(f"处理文件: {file_name}")
        
        # 读取 Fit_Output 文件
        fit_df = pd.read_csv(file_name)
        id_phase = dict(zip(fit_df["ID"], fit_df["Phase"]))

        # 读取表达矩阵
        expr_df = pd.read_csv("/home/xuzhen/CYCLOPS-2.0/CIRCADIA/HTAN_HTAPP_Freash.pseudobulk_matrix.tsv/expression.csv")

        # 查找目标基因
        gene_row = expr_df[expr_df["Gene_Symbol"] == gene]
        if gene_row.empty:
            print(f"未找到基因 {gene}，跳过文件 {file_name}")
            continue

        expr_dict = gene_row.iloc[0].to_dict()
        expr_dict.pop("Gene_Symbol")

        # 收集样本的相位和表达量
        plot_data = []
        for sample_id, expr in expr_dict.items():
            if sample_id in id_phase:
                plot_data.append((id_phase[sample_id], float(expr)))

        if not plot_data:
            print(f"文件 {file_name} 中没有匹配的样本，跳过")
            continue

        plot_data.sort()
        phases, exprs = zip(*plot_data)

        # 绘制散点图和拟合曲线
        plt.figure(figsize=(8, 6))
        plt.scatter(phases, exprs, label="Samples", alpha=0.7)

        # 拟合平滑曲线
        spline = UnivariateSpline(phases, exprs, s=1)
        xs = np.linspace(0, 2 * np.pi, 200)
        ys = spline(xs)
        plt.plot(xs, ys, color="red", label="Spline fit")

        # 自定义横坐标刻度和范围
        xticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        xtick_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
        plt.xticks(xticks, xtick_labels)
        plt.xlim(0, 2 * np.pi)  # 设置横坐标范围为 0 到 2π

        # 图形设置
        plt.xlabel("Phase (radian)")
        plt.ylabel(f"{gene} expression")
        plt.title(f"{gene} Expression in {file_name}")
        plt.legend()
        plt.grid(True)

        # 保存图像
        output_file = f"{gene}_{file_name.replace('.csv', '')}.png"
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"图像已保存: {output_file}")