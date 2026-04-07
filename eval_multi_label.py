import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 导入 CheXzero 核心函数
import zero_shot_1 

# --- 1. 配置基础路径 ---
# 填入你本次要测试的 checkpoint 路径（例如 3k_real 组或合成组）
model_paths = ["F:/5207_project/model_3/checkpoints/model_3/checkpoint_3400.pt"] 
cxr_filepath = "F:/5207_project/BN5207_data-main/stage0_training_handoff/data/test.h5"
test_csv_path = "F:/5207_project/BN5207_data-main/stage0_training_handoff/test_with_labels.csv"
cache_dir = "F:/5207_project/model_3/cache"

# --- 2. 定义五个病灶标签（严格对应 label 1-5 的顺序） ---
# 1:Atelectasis, 2:Cardiomegaly, 3:Edema, 4:Effusion, 5:Pleural
cxr_labels = ["Atelectasis", "Cardiomegaly", "Edema", "Pleural", "Effusion"]

# --- 3. 执行推理获取原始概率 ---
print("正在执行模型推理，这可能需要几分钟...")
# ensemble_models 会利用正负模板计算初始概率
_, y_pred_raw = zero_shot_1.ensemble_models(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=("{}", "no {}"), 
    cache_dir=cache_dir
)

# --- 4. 核心转换：从独立判断转为“五选一” ---
# 使用 softmax 使每一行的 5 个标签概率之和为 1
y_pred_multiclass = softmax(y_pred_raw, axis=1) 
# 取概率最大的索引作为预测结果 (0-4)
y_pred_final = np.argmax(y_pred_multiclass, axis=1) 

# --- 5. 加载真实标签 ---
df_test = pd.read_csv(test_csv_path)
# 将 label 列 (1-5) 转换为索引 (0-4) 以便比对
y_true_final = df_test['label'].values - 1 

# --- 6. 计算多分类指标 ---
print("\n" + "="*30)
print("STAGE 3 多分类评估结果")
print("="*30)

acc = accuracy_score(y_true_final, y_pred_final)
report = classification_report(y_true_final, y_pred_final, target_names=cxr_labels)

from sklearn.metrics import roc_auc_score
auc_scores = roc_auc_score(y_true_final, y_pred_multiclass, multi_class='ovr', average=None)

report_dict = classification_report(y_true_final, y_pred_final, target_names=cxr_labels, output_dict=True)
df_eval = pd.DataFrame(report_dict).transpose()

# C. 将 AUC 整合进表格
# 为前 5 个病灶行添加对应的 AUC 值
for i, label in enumerate(cxr_labels):
    df_eval.loc[label, 'AUC'] = auc_scores[i]

# 为整体平均行 (macro avg) 添加平均 AUC
df_eval.loc['macro avg', 'AUC'] = np.mean(auc_scores)

# D. 格式化输出：移除 support 列，保留 precision, recall, f1-score, AUC
df_eval = df_eval[['precision', 'recall', 'f1-score', 'AUC']]

print(f"总体准确率 (Accuracy): {acc:.4f}")
print("\n详细分类报告：\n", report)
print(df_eval)

# --- 7. 绘制并保存混淆矩阵 ---
cm = confusion_matrix(y_true_final, y_pred_final)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=cxr_labels, yticklabels=cxr_labels, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Multi-class Confusion Matrix')
plt.savefig("F:/5207_project/stage3_confusion_matrix.png")
plt.show()

# 保存量化指标到 CSV
df_eval.to_csv("F:/5207_project/model_3/eval/checkpoint_3400_eval.csv")
