#!/bin/bash

# 定义数据集和结果路径
DATASET="Zhang_CancerCell_2025.Sample_MajorCluster"
RESULT_PATH="Zhang_CancerCell_2025.Sample_MajorCluster"

# 定义基因列表
GENES=("CLOCK" "NFIL3" "ARNTL" "NPAS2" "NR1D1" "CRY1" "CRY2" "PER1" "PER2" "PER3")

# 循环运行 Python 脚本
for GENE in "${GENES[@]}"; do
    echo "正在处理基因: $GENE"
    python plot_dataset.py --gene "$GENE" --dataset "$DATASET" --result_path "$RESULT_PATH"
done

echo "所有基因处理完成！"