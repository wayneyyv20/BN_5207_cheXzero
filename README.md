# BN_5207_cheXzero
1. eval.py (核心底层逻辑)
作用：定义了所有评估相关的函数（计算指标、画图、Bootstrap 采样）
关键改进：
  动态阈值：不再使用硬编码的 0.5，而是通过 Youden's Index (约登指数) 自动寻找最优分类截断点。
  指标扩充：除了原有的 AUC，新增了 Precision, Recall, F1-score 和 Accuracy 的计算。

2. zero_shot.py (模型推理)
作用：加载预训练模型（CLIP/CheXzero）并对测试集进行推理。
流程：将图像特征与文本提示词（Prompts）进行对比，生成原始的预测概率分数（Probability scores）。

3. eval_task.py (任务运行入口)
流程：
   调用 zero_shot.py 获取预测结果。
   调用 eval.py 进行性能分析。
   结果输出：将最终的评估指标汇总并保存为 CSV 文件（例如：checkpoint_epo30_eval.csv）。
