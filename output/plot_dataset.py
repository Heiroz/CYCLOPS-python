import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import argparse
from scipy.optimize import curve_fit

def load_fit_output(file_name):
    """加载 Fit_Output 文件并返回 ID 和 Phase 的字典"""
    fit_df = pd.read_csv(file_name, low_memory=False)
    return dict(zip(fit_df["ID"], fit_df["Phase"]))

def load_expression_data(expression_file, gene):
    """加载表达矩阵并返回目标基因的表达数据"""
    expr_df = pd.read_csv(expression_file, low_memory=False)
    gene_row = expr_df[expr_df["Gene_Symbol"] == gene]
    if gene_row.empty:
        return None
    expr_dict = gene_row.iloc[0].to_dict()
    expr_dict.pop("Gene_Symbol")
    return expr_dict

def collect_plot_data(expr_dict, id_phase):
    """收集样本的相位和表达量"""
    plot_data = []
    for sample_id, expr in expr_dict.items():
        if sample_id in id_phase:
            plot_data.append((id_phase[sample_id], float(expr)))
    return plot_data

def plot_data_and_save(phases, exprs, gene, file_name, saved_path):
    """绘制散点图和正弦拟合曲线，并保存图像"""
    plt.figure(figsize=(8, 6))
    plt.scatter(phases, exprs, label="Samples", alpha=0.7)

    # 正弦拟合
    def sin_func(x, A, w, phi, C):
        return A * np.sin(w * x + phi) + C


    # 初始参数估计
    A_guess = (max(exprs) - min(exprs)) / 2
    w_guess = 1
    phi_guess = 0
    C_guess = np.mean(exprs)
    try:
        popt, _ = curve_fit(sin_func, phases, exprs, p0=[A_guess, w_guess, phi_guess, C_guess], maxfev=10000)
        xs = np.linspace(0, 2 * np.pi, 200)
        ys = sin_func(xs, *popt)
        plt.plot(xs, ys, color="red", label="Sine fit")
    except Exception as e:
        print(f"正弦拟合失败: {e}")

    xticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    xtick_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    plt.xticks(xticks, xtick_labels)
    plt.xlim(0, 2 * np.pi)

    plt.xlabel("Phase (radian)")
    plt.ylabel(f"{gene} expression")
    plt.title(f"{gene} Expression in {file_name}")
    plt.legend()
    plt.grid(True)

    output_file = f"{saved_path}/{gene}/{file_name.replace('.csv', '')}.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"图像已保存: {output_file}")

def process_files(dataset, result_path, saved_path, gene):
    for file_name in os.listdir(f"{result_path}/Fit_celltype/"):
        if file_name.startswith("Fit_Output_") and file_name.endswith(".csv"):
            file_path = os.path.join(result_path, "Fit_celltype", file_name)
            print(f"处理文件: {file_name}")
            id_phase = load_fit_output(file_path)
            expr_dict = load_expression_data(f"../data/{dataset}/expression.csv", gene)
            if expr_dict is None:
                print(f"未找到基因 {gene}，跳过文件 {file_name}")
                continue

            plot_data = collect_plot_data(expr_dict, id_phase)
            if not plot_data:
                print(f"文件 {file_name} 中没有匹配的样本，跳过")
                continue

            plot_data.sort()
            phases, exprs = zip(*plot_data)
            plot_data_and_save(phases, exprs, gene, file_name, saved_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制基因表达与相位的关系图")
    parser.add_argument("--gene", required=True, help="目标基因名称")
    parser.add_argument("--dataset", required=True, help="数据集名称")
    parser.add_argument("--result_path", required=True, help="结果文件路径")
    args = parser.parse_args()

    gene = args.gene
    dataset = args.dataset
    result_path = args.result_path
    saved_path = f"{result_path}/expression_level_plot"
    process_files(dataset, result_path, saved_path, gene)