import os
import pandas as pd
from zero_shot_1 import run_zero_shot

# 1. 定义你想评估的疾病标签 (必须与 CSV 中的列名对应)
labels = [
    'Atelectasis', 'Cardiomegaly', 'Edema', 
    'Pleural', 'Effusion'
]

# 2. 定义正负提示词模版
# 模型会对比 "Atelectasis" 和 "no Atelectasis" 的概率
templates = [("{}", "no {}")]

# 3. 指定路径
MODEL_PATH = "F:/5207_project/model_2/checkpoints/model_2/checkpoint_4500.pt"
TEST_H5_PATH = "F:/5207_project/BN5207_data-main/stage0_training_handoff/data/test.h5"
LABEL_CSV_PATH = "F:/5207_project/BN5207_data-main/stage0_training_handoff/test_with_labels.csv"

SAVE_DIR = "F:/5207_project/model_2/eval" # 文件夹路径
SAVE_NAME = "checkpoint_4500_eval.csv"           # 文件名

full_save_path = os.path.join(SAVE_DIR, SAVE_NAME)

# 4. 执行零样本评估
# 注意：如果你训练时使用了预训练权重，pretrained 设为 True
results, y_pred = run_zero_shot(
    cxr_labels=labels,
    cxr_templates=templates,
    model_path=MODEL_PATH,
    cxr_filepath=TEST_H5_PATH,
    final_label_path=LABEL_CSV_PATH,
    pretrained=False,     # 如果是你从头训练或微调的选 False
    use_bootstrap=True    # 是否计算置信区间 (1000次采样，较慢)
)

# 5. 查看结果
# results[0] 包含原始采样数据, results[1] 包含各指标的 mean, lower, upper bound
boot_stats, stats_df = results[0]
print("\n评估结果 (AUROC):")
print(stats_df)

stats_df.to_csv(full_save_path)
print(f"评估结果已成功保存至: {full_save_path}")
