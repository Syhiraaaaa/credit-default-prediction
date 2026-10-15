# 信贷违约预测模型

基于 Lending Club 公开信贷数据构建违约预测模型，输出可解释的风险特征与可执行的风险分层策略，辅助贷前审批决策。

- 数据：Lending Club 信贷数据集（阿里天池版），800,000 行 × 47 列
- 预测目标：`isDefault`（0 = 正常还款，1 = 违约）
- 主指标：AUC-ROC；辅助指标：KS
- 周期：4 周 6 阶段
- 整体违约率：19.95%（正负样本约 4:1）

原始数据集约 300MB，未纳入本仓库。下载 `credit_data(1).csv` 后置于 `项目资料/` 目录即可。

## 代码文件

| 文件 | 说明 |
|---|---|
| data_loading.py | 数据加载与整体概览 |
| data_preprocessing.py | 缺失值、异常值、重复值处理 |
| feature_engineering.py | 类别编码、数值变换、特征创建与选择 |
| model_training.py | 逻辑回归 / 随机森林 / XGBoost 训练与评估 |
| model_optimization.py | 超参搜索、交叉验证与集成 |
| dashboard/credit_dashboard.py | Streamlit + Plotly 在线业务仪表板 |
| main_pipeline.ipynb | 完整项目流程 Notebook |

各脚本自包含、零本地依赖，可单独运行。结果表写入 `outputs/tables/`，图表写入 `outputs/figures/`。

`dashboard/outputs/tables/` 为本期运行产出的结果表，随仓库提供，无需重跑建模即可启动看板。

## 环境

Python 3.13，conda 环境 `credit`。依赖包版本见 `requirements.txt`。

```bash
conda create -n credit python=3.13
conda activate credit
pip install -r requirements.txt
python -m ipykernel install --user --name credit --display-name "Python (credit)"
```

## 复现步骤

1. 下载 `credit_data(1).csv`，放入 `项目资料/`
2. 按编号顺序执行：`data_loading.py` → `data_preprocessing.py` → `feature_engineering.py`
   → `model_training.py` → `model_optimization.py`
3. 启动仪表板查看交互式结果

也可直接打开 `main_pipeline.ipynb` 顺序执行全部流程，内核选择 `Python (credit)`。

## 业务仪表板

`credit_dashboard.py` 读取 `dashboard/outputs/tables/` 下已产出的结果表，不重新训练模型。

```bash
streamlit run dashboard/credit_dashboard.py
```

浏览器打开 `http://localhost:8501`，左侧边栏切换 8 个页面。

在线版本：https://credit-risk-dashboard.app.workbuddy.host/

| 页面 | 内容 |
|---|---|
| 业务概览 | 业务背景、数据集特点、两种切分方式、业务侧已知风险规律 |
| 特征解释 | SHAP 全局重要性，89 个特征已全部替换为业务名称与业务含义 |
| 模型选型 | 四组候选配置横评、最优配置参数一览、逐参数取值敏感度 |
| 模型表现 | ROC / PR / KS / 五折交叉验证、调优前后指标对照 |
| 阈值决策 | 阈值扫描、金额成本与笔数成本双口径、漏放误拒代价比对 |
| 风险分层 | 十档实际违约率与相对倍数、累计捕获率 |
| 稳定性与校准 | 整体 PSI、概率校准曲线、逐特征 PSI 漂移、时间外推验证 |
| 结论与局限 | 现有能力边界、天花板成因、四条提升路径 |

所有结果表均为聚合指标，不含逐笔借款人记录，可对外提供。

## 核心结果

| 指标 | 数值 |
|---|---|
| 测试集 AUC-ROC（XGBoost 最优配置） | 0.7395 |
| 测试集 KS | 0.3514 |
| Brier 分数 | 0.1400 |
| 时间外样本测试 AUC-ROC | 0.7226 |
| 前 30% 人群捕获违约占比 | 56.6% |
| 最危险一档 Lift | 2.492 |
| 业务最优阈值 | 0.17 |
| 超参数搜索规模 | 693 组 |

口径说明：上表为最优配置（树深 6 / 学习率 0.02 / L2 正则 30 / 最小子样本权重 24，
随机种子 42，峰值轮 2920）在 random 切分下的测试集结果，与在线业务仪表板同源。
另有「训练集 + 验证集合并重训」口径（测试 AUC 0.7403、首档 Lift 2.507），
两套口径不同源，不可混用。
