# -*- coding: utf-8 -*-
"""信贷违约预测项目：特征工程（第 3 阶段）

输入：db/processed/credit_clean.parquet（第 2 阶段清洗产物，800,000 行 x 49 列）
输出：db/processed/credit_fe_{part}_{tag}.parquet（{part}=train/valid/test，{tag}=random 或 oot_y2013）
      outputs/tables/04_*.csv、outputs/figures/04_*.png
      补充实验另产出 outputs/tables/10_*.csv

功能：
1. 数据加载与切分（random 分层抽样 / oot 时间切分）
2. 类别特征编码（序数、独热、K 折 OOF 目标编码）
3. 数值特征标准化/归一化、等频分箱、多项式与组合特征
4. 特征选择（过滤法、包装法、嵌入法）与特征重要性分析
5. 补充实验：未使用字段补漏、组合特征构造与分组增量验证（对应特征工程报告第 6 章）

运行环境：pandas / numpy / scikit-learn / lightgbm / xgboost / pyarrow / matplotlib
运行方式：python feature_engineering.py
"""

import os

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', 60)

RANDOM_STATE = 42
TARGET = 'isDefault'

# 强右偏数值列（偏度 > 2），做 log1p 变换
LOG1P_COLS = ['loanAmnt', 'annualIncome', 'revolBal', 'installment', 'delinquency_2years']

# 参与 StandardScaler 的连续数值列
SCALE_COLS = ['loanAmnt_log', 'interestRate', 'installment_log', 'annualIncome_log',
              'dti', 'delinquency_2years_log', 'openAcc', 'pubRec', 'pubRecBankruptcies',
              'revolBal_log', 'revolUtil', 'totalAcc', 'n0', 'n1', 'n2', 'n4', 'n5',
              'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14',
              'employmentLength', 'fico_mean', 'credit_hist_days',
              'loan_to_income', 'installment_to_income', 'revolBal_to_income',
              'issue_year', 'issue_month']

# 目标编码的高基数类别列
TE_COLS = ['title', 'employmentTitle', 'postCode', 'regionCode']
TE_SMOOTHING = 100   # 平滑权重
TE_FOLDS = 5         # OOF 交叉编码折数

# 分箱列（等频 5 箱）
BIN_COLS = ['interestRate', 'dti', 'fico_mean']
BIN_LABELS = [0, 1, 2, 3, 4]

# 多项式特征的输入列
POLY_INPUT_COLS = ['interestRate', 'dti', 'revolUtil', 'fico_mean']
POLY_DEGREE = 2

# 建模不使用的原始列
DROP_COLS = ['id', 'issueDate', 'earliesCreditLine', 'grade', 'subGrade',
             'ficoRangeLow', 'ficoRangeHigh', 'title', 'employmentTitle',
             'postCode', 'regionCode']


# ---------------------------------------------------------------------------
# 路径定位
# ---------------------------------------------------------------------------
def find_base():
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(cur, '项目资料', 'credit_data(1).csv')):
            return cur
        cur = os.path.dirname(cur)
    raise FileNotFoundError('未找到项目根目录')


BASE = find_base()
IN_PATH = os.path.join(BASE, 'db', 'processed', 'credit_clean.parquet')
OUT_DIR = os.path.join(BASE, 'db', 'processed')
TAB_DIR = os.path.join(BASE, 'outputs', 'tables')
FIG_DIR = os.path.join(BASE, 'outputs', 'figures')
SPLIT_DIR = os.path.join(BASE, 'db', 'processed', 'splits')

# 切分比例（train / val / test）
SPLIT_RATIO = (0.70, 0.20, 0.10)


# ---------------------------------------------------------------------------
# 数据集划分
# ---------------------------------------------------------------------------
def split_data(df, strategy='random', seed=42, oot_train_start_year=None):
    """生成并持久化一种划分策略，返回 (train_idx, val_idx, test_idx)。

    random：按 isDefault 分层抽样 70/20/10；
    oot   ：按 issueDate 升序行百分位切 70/20/10（train 早 / test 晚），
            可用 oot_train_start_year 限定训练窗口起点。
    切分结果持久化到 db/processed/splits/。
    """
    df = df.reset_index(drop=True)
    n = len(df)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    if strategy == 'random':
        y = df['isDefault']
        tr, tmp = train_test_split(df.index, test_size=SPLIT_RATIO[1] + SPLIT_RATIO[2],
                                   stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2]),
                                  stratify=y.loc[tmp], random_state=seed)
        train_idx, val_idx, test_idx = tr, va, te
    elif strategy == 'oot':
        d = pd.to_datetime(df['issueDate'], errors='coerce')
        order = d.sort_values().index
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]
        if oot_train_start_year is not None:
            yr = d.loc[train_idx].dt.year
            train_idx = train_idx[yr >= oot_train_start_year]
    else:
        raise ValueError('unknown strategy: ' + str(strategy))

    os.makedirs(SPLIT_DIR, exist_ok=True)
    suffix = strategy if oot_train_start_year is None else 'oot_y%d' % oot_train_start_year
    out = pd.DataFrame({
        'orig_index': list(train_idx) + list(val_idx) + list(test_idx),
        'split': (['train'] * len(train_idx) + ['val'] * len(val_idx)
                  + ['test'] * len(test_idx)),
    })
    out.to_csv(os.path.join(SPLIT_DIR, '%s_split.csv' % suffix), index=False)
    return train_idx, val_idx, test_idx


def describe_split(name, data):
    """打印各集合的规模、违约率与时间跨度。"""
    print('%-6s %7d 行 | 违约率 %.4f%% | 时间 %s ~ %s'
          % (name, len(data), data[TARGET].mean() * 100,
             data['issueDate'].min().date(), data['issueDate'].max().date()))


# ---------------------------------------------------------------------------
# 2. 类别特征编码
# ---------------------------------------------------------------------------
def encode_ordinal(train, others):
    """序数编码：grade/subGrade 按等级映射为整数。"""
    grade_map = {g: i for i, g in enumerate('ABCDEFG')}
    sub_cats = sorted(train['subGrade'].unique())  # A1..G5 按字典序即等级序
    sub_map = {s: i for i, s in enumerate(sub_cats)}
    for d in [train] + others:
        d['grade_ord'] = d['grade'].map(grade_map)
        d['subGrade_ord'] = d['subGrade'].map(sub_map)
    print('序数编码：grade A~G -> 0~6，subGrade %s~%s -> 0~%d'
          % (sub_cats[0], sub_cats[-1], len(sub_cats) - 1))
    return grade_map, sub_map


def encode_onehot(train, others):
    """独热编码：homeOwnership / verificationStatus / purpose 三个低基数类别列。"""
    onehot_cols = ['homeOwnership', 'verificationStatus', 'purpose']
    cats = {c: sorted(train[c].unique()) for c in onehot_cols}
    for c in onehot_cols:
        print('独热编码：%-18s %2d 个取值 -> %2d 列' % (c, len(cats[c]), len(cats[c])))
    frames = []
    for d in [train] + others:
        parts = []
        for c in onehot_cols:
            cat = pd.Categorical(d[c], categories=cats[c])
            parts.append(pd.get_dummies(cat, prefix=c).astype('int8'))
        base = d.reset_index(drop=True)
        dm = pd.concat(parts, axis=1).reset_index(drop=True)
        assert len(base) == len(dm), '独热编码列数与样本数不一致'
        frames.append(pd.concat([base, dm], axis=1))
    return frames, cats


# ---------------------------------------------------------------------------
# 目标编码（K 折 OOF）
# ---------------------------------------------------------------------------
def _smooth_mapping(group_df, col, global_mean, smoothing):
    """先验收缩的目标编码：enc = (sum_y + m*global) / (count + m)。"""
    grp = group_df.groupby(col)[TARGET].agg(['sum', 'count'])
    return (grp['sum'] + smoothing * global_mean) / (grp['count'] + smoothing)


def fit_target_encoding(train, n_splits=TE_FOLDS, smoothing=TE_SMOOTHING):
    """目标编码：训练集用 K 折 OOF 编码，返回映射供验证/测试集 transform。"""
    global_mean = train[TARGET].mean()
    full_mapping = {c: _smooth_mapping(train, c, global_mean, smoothing) for c in TE_COLS}
    for c in TE_COLS:
        print('目标编码（平滑 m=%d）：%-18s %6d 个取值 | 编码值范围 %.4f ~ %.4f'
              % (smoothing, c, len(full_mapping[c]),
                 full_mapping[c].min(), full_mapping[c].max()))

    oof = {c: pd.Series(np.nan, index=train.index) for c in TE_COLS}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, va_idx in kf.split(train):
        tr = train.iloc[tr_idx]
        for c in TE_COLS:
            enc = _smooth_mapping(tr, c, global_mean, smoothing)
            oof[c].iloc[va_idx] = train.iloc[va_idx][c].map(enc)
    for c in TE_COLS:
        train[c + '_te'] = oof[c].fillna(global_mean).astype('float32')
        print('  %-18s OOF 折内未见、回退全局先验 %d 行' % (c, int(oof[c].isna().sum())))
    return full_mapping, global_mean


def apply_target_encoding(others, mapping, global_mean):
    """目标编码应用：验证/测试集用训练集映射 transform，未见取值回退全局先验。"""
    for name, d in others:
        unseen = {}
        for c in TE_COLS:
            d[c + '_te'] = d[c].map(mapping[c]).fillna(global_mean).astype('float32')
            unseen[c] = int(d[~d[c].isin(mapping[c].index)][c].nunique())
        print('%s未见过、回退为先验的取值个数：%s' % (name, unseen))
    return [d for _, d in others]


# ---------------------------------------------------------------------------
# 3. 数值特征标准化/归一化 + 特征创建
# ---------------------------------------------------------------------------
def create_features(train, others):
    """log1p 变换、组合特征、信用历史长度与放款时间拆分。"""
    for d in [train] + others:
        # log1p 变换
        for c in LOG1P_COLS:
            d[c + '_log'] = np.log1p(d[c].clip(lower=0)).astype('float32')
        # 合并 FICO 上下限（两列完全共线）
        d['fico_mean'] = ((d['ficoRangeLow'] + d['ficoRangeHigh']) / 2).astype('float32')
        # 收入比类组合特征
        income = d['annualIncome'].where(d['annualIncome'] > 0)
        d['loan_to_income'] = (d['loanAmnt'] / income).astype('float32')
        d['installment_to_income'] = (d['installment'] * 12 / income).astype('float32')
        d['revolBal_to_income'] = (d['revolBal'] / income).astype('float32')
        # 信用历史长度（天）与放款时间拆分
        d['credit_hist_days'] = (d['issueDate'] - d['earliesCreditLine']).dt.days.astype('float32')
        d['issue_year'] = d['issueDate'].dt.year.astype('int16')
        d['issue_month'] = d['issueDate'].dt.month.astype('int8')
    ratio_cols = ['loan_to_income', 'installment_to_income', 'revolBal_to_income']
    med = train[ratio_cols].median()   # 填充统计量只取训练集
    for d in [train] + others:
        for c in ratio_cols:
            d[c] = d[c].fillna(med[c])
    print('log1p 变换列：%s' % LOG1P_COLS)
    print('组合特征：fico_mean、loan_to_income、installment_to_income、revolBal_to_income、'
          'credit_hist_days、issue_year、issue_month')
    print('年收入为 0 的行（比率置缺失后填充中位数）：%d 行' % int((train['annualIncome'] == 0).sum()))
    return [train] + others


def fit_binning(train):
    """等频分箱拟合（仅用训练集分位数），返回各列分箱边界。"""
    edges = {}
    for c in BIN_COLS:
        _, e = pd.qcut(train[c], q=5, retbins=True, duplicates='drop')
        edges[c] = e
        print('等频分箱：%-12s 边界 %s' % (c, np.round(e, 2).tolist()))
    return edges


def apply_binning(train, others, edges):
    """按训练集边界分箱为有序整数（0~4），输出训练集各箱违约率验证单调性。"""
    for c in BIN_COLS:
        e = edges[c]
        for d in [train] + others:
            d[c + '_bin'] = pd.cut(d[c], bins=e, labels=BIN_LABELS,
                                   include_lowest=True).astype('float32')
        rate = train.groupby(c + '_bin')[TARGET].mean() * 100
        print('%-12s 各箱违约率%%：%s' % (c, rate.round(2).tolist()))
    return [train] + others


def fit_scaler(train):
    """标准化缩放器拟合（仅用训练集统计量）。"""
    scaler = StandardScaler().fit(train[SCALE_COLS])
    print('StandardScaler 拟合 %d 列' % len(SCALE_COLS))
    print('标准化前后对比（训练集）：')
    print('%-22s %10s %10s %12s %12s'
          % ('列', '原均值', '原标准差', '变换后均值', '变换后标准差'))
    for i, c in enumerate(SCALE_COLS[:6]):
        mu, sd = train[c].mean(), train[c].std()
        print('%-22s %10.3f %10.3f %12.4f %12.4f'
              % (c, mu, sd, scaler.mean_[i], scaler.scale_[i]))
    return scaler


def apply_scaler(train, others, scaler):
    """应用标准化，数值列统一转为 float32。"""
    for d in [train] + others:
        d[SCALE_COLS] = scaler.transform(d[SCALE_COLS]).astype('float32')
    print('标准化完成：%d 列' % len(SCALE_COLS))
    return [train] + others


def build_polynomial(train, others):
    """多项式特征：4 个关键列（已标准化）再做 MinMax 归一化后二次展开，只保留平方项与交互项。"""
    mm = MinMaxScaler().fit(train[POLY_INPUT_COLS])   # MinMax 也只在训练集上拟合
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
    out_names = []
    for d in [train] + others:
        x_mm = mm.transform(d[POLY_INPUT_COLS])
        x_poly = poly.fit_transform(x_mm)  # 多项式展开无统计量，fit_transform 等价 transform
        full_names = poly.get_feature_names_out(POLY_INPUT_COLS)
        new_names = ['poly_' + n.replace(' ', '_x_').replace('^', '_sq_')
                     for n in full_names[len(POLY_INPUT_COLS):]]
        for j, name in enumerate(new_names):
            d[name] = x_poly[:, len(POLY_INPUT_COLS) + j].astype('float32')
        out_names = new_names
    print('多项式特征：MinMax 归一化 %d 列 -> 二次展开，新增 %d 列（平方项 + 交互项）'
          % (len(POLY_INPUT_COLS), len(out_names)))
    return [train] + others, out_names


# ---------------------------------------------------------------------------
# 4. 特征选择（过滤法 / 包装法 / 嵌入法）
# ---------------------------------------------------------------------------
def feature_matrix(df):
    """组装最终特征矩阵：剔除已替代的原始列，特征与标签分离。"""
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = X.pop(TARGET)
    return X, y


# 高相关剔除的豁免列（组合特征，业务含义不同、不可互相替代）
CORR_KEEP_COLS = ['loan_to_income', 'installment_to_income', 'revolBal_to_income']


def select_filter(X, y):
    """过滤法：低方差检查、基础列中 |r|>=0.98 高相关对剔除（保留与目标相关更强的一列）、互信息排序。"""
    vt = VarianceThreshold(threshold=0.0)
    vt.fit(X.fillna(X.median()))
    constant_cols = X.columns[~vt.get_support()].tolist()
    print('方差过滤（threshold=0）：常量列 %d 个 %s' % (len(constant_cols), constant_cols))

    base_cols = [c for c in X.columns if not c.startswith('poly_')]
    corr = X[base_cols].corr()
    cols = list(corr.columns)
    drop_by_corr = set()
    pairs = []
    exempted = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= 0.98:
                a, b = cols[i], cols[j]
                pairs.append((a, b, round(r, 4)))
                if a in CORR_KEEP_COLS and b in CORR_KEEP_COLS:
                    exempted.append((a, b, round(r, 4)))
                    continue
                ra, rb = abs(X[a].corr(y)), abs(X[b].corr(y))
                loser = b if ra >= rb else a
                drop_by_corr.add(loser)
    print('|r| >= 0.98 的高相关对 %d 组，各保留与目标相关更强的一列：' % len(pairs))
    for a, b, r in pairs:
        tag = '（组合特征豁免，两者均保留）' if (a, b, r) in exempted else ''
        print('  %s ~ %s (r=%s) %s' % (a, b, r, tag))
    print('剔除列：%s' % sorted(drop_by_corr))

    sample = X.drop(columns=list(drop_by_corr)).sample(n=100_000, random_state=RANDOM_STATE)
    ys = y.loc[sample.index]
    sample = sample.fillna(sample.median())
    mi = mutual_info_classif(sample, ys, random_state=RANDOM_STATE)
    mi_rank = pd.Series(mi, index=sample.columns).sort_values(ascending=False)
    print('互信息 TOP10（抽样 10 万行）：')
    print(mi_rank.head(10).round(5).to_string())
    return sorted(drop_by_corr), mi_rank, pairs


def select_wrapper(X, y, n_select=40):
    """包装法：RFE 递归特征消除（LogisticRegression 评估器，抽样 10 万行控制耗时）。"""
    sample = X.sample(n=100_000, random_state=RANDOM_STATE)
    ys = y.loc[sample.index]
    sample = sample.fillna(sample.median())
    estimator = LogisticRegression(max_iter=1000, solver='lbfgs')
    rfe = RFE(estimator, n_features_to_select=n_select, step=8)
    rfe.fit(sample, ys)
    rank = pd.Series(rfe.ranking_, index=X.columns).sort_values()
    selected = list(rank[rank == 1].index)
    print('RFE（LogisticRegression，抽样 10 万行）选出 %d 列：' % len(selected))
    print(', '.join(selected))
    return selected, rank


def select_embedded(X, y):
    """嵌入法：LightGBM 训练并按 gain 汇总特征重要性。"""
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    model.fit(X, y)
    imp = pd.Series(model.booster_.feature_importance(importance_type='gain'),
                    index=X.columns).sort_values(ascending=False)
    print('LightGBM（300 棵树）gain 重要性 TOP20：')
    print(imp.head(20).round(0).to_string())
    return model, imp


# ---------------------------------------------------------------------------
# 5. 特征重要性分析 + 切分对比图
# ---------------------------------------------------------------------------
def importance_analysis(model, X_train, y_train, evals, imp, tag):
    """特征重要性图 + 训练/验证/测试 AUC 校验。"""
    top = imp.head(25)[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top.index, top.values, color='#2c7fb8')
    for i, v in enumerate(top.values):
        ax.text(v, i, ' %.0f' % v, va='center', fontsize=8)
    ax.set_title('LightGBM 特征重要性 TOP25（gain，%s）' % tag, fontsize=12)
    ax.set_xlabel('gain 重要性')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '04_importance_top25_%s.png' % tag), dpi=150)
    plt.close(fig)

    print('特征集校验（LightGBM，%s）：' % tag)
    print('训练集 AUC：%.4f' % roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    for name, X_e, y_e in evals:
        print('%s AUC：%.4f' % (name, roc_auc_score(y_e, model.predict_proba(X_e)[:, 1])))


def binning_figure(train, tag):
    """分箱单调性图：三个分箱列的各箱违约率。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    base = train[TARGET].mean() * 100
    for ax, c in zip(axes, BIN_COLS):
        rate = train.groupby(c + '_bin')[TARGET].mean() * 100
        ax.bar(range(1, len(rate) + 1), rate.values, color='#2c7fb8', width=0.6)
        ax.axhline(base, color='gray', linestyle='--', linewidth=1)
        ax.set_title('%s 各箱违约率' % c, fontsize=11)
        ax.set_xlabel('分箱（1~5，由低到高）')
        ax.set_ylabel('违约率（%）')
    fig.suptitle('等频分箱后的违约率（虚线为训练集基准 %.2f%%）' % base, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '04_binning_default_rate_%s.png' % tag), dpi=150)
    plt.close(fig)
    print('已输出分箱单调性图：04_binning_default_rate_%s.png' % tag)


def category_default_rate_figure(train, tag):
    """类别特征各取值的违约率。"""
    specs = [('grade', None), ('homeOwnership', None), ('verificationStatus', None), ('purpose', 8)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    base = train[TARGET].mean() * 100
    for ax, (c, topn) in zip(axes.ravel(), specs):
        rate = train.groupby(c)[TARGET].mean().mul(100).sort_values(ascending=False)
        if topn:
            rate = rate.head(topn)
        ax.bar([str(i) for i in rate.index], rate.values, color='#2c7fb8')
        ax.axhline(base, color='gray', linestyle='--', linewidth=1)
        ax.set_title('%s（基准 %.2f%%）' % (c, base), fontsize=11)
        ax.set_ylabel('违约率（%）')
        ax.tick_params(axis='x', rotation=30, labelsize=8)
    fig.suptitle('类别特征各取值违约率（%s）' % tag, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '04_category_default_rate_%s.png' % tag), dpi=150)
    plt.close(fig)
    print('已输出类别违约率图：04_category_default_rate_%s.png' % tag)


def scaling_compare_figure(raw_df, scaled_df, tag, cols=('interestRate', 'dti')):
    """标准化 / 归一化前后分布对比：原始 vs StandardScaler vs MinMaxScaler。"""
    cols = list(cols)
    mm_vals = MinMaxScaler().fit_transform(raw_df[cols])
    fig, axes = plt.subplots(3, len(cols), figsize=(5.5 * len(cols), 11))
    if len(cols) == 1:
        axes = axes.reshape(3, 1)
    for j, c in enumerate(cols):
        raw = raw_df[c].dropna().values
        std = scaled_df[c].dropna().values
        axes[0, j].hist(raw, bins=50, color='#9ecae1')
        axes[0, j].set_title('%s 原始\n均值 %.3f，标准差 %.3f' % (c, raw.mean(), raw.std()), fontsize=10)
        axes[1, j].hist(std, bins=50, color='#2c7fb8')
        axes[1, j].set_title('%s StandardScaler\n均值 %.3f，标准差 %.3f' % (c, std.mean(), std.std()), fontsize=10)
        axes[2, j].hist(mm_vals[:, j], bins=50, color='#fe9929')
        axes[2, j].set_title('%s MinMaxScaler\n范围 %.3f ~ %.3f' % (c, mm_vals[:, j].min(), mm_vals[:, j].max()), fontsize=10)
    fig.suptitle('数值特征标准化 / 归一化前后分布（%s）' % tag, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '04_scaling_compare_%s.png' % tag), dpi=150)
    plt.close(fig)
    print('已输出标准化对比图：04_scaling_compare_%s.png' % tag)


def feature_sources_figure(counts, tag):
    """特征来源构成：各处理方法新增的特征列数。"""
    names = list(counts.keys())[::-1]
    vals = list(counts.values())[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, vals, color='#41b6c4')
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, ' %d' % v, va='center', fontsize=9)
    ax.set_xlabel('新增特征列数')
    ax.set_title('特征来源构成（%s）' % tag, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '04_feature_sources_%s.png' % tag), dpi=150)
    plt.close(fig)
    print('已输出特征来源构成图：04_feature_sources_%s.png' % tag)


def selection_compare_figure(mi_rank, rfe_rank, imp, tag, topk=40):
    """三种特征选择方法对比：选中集合重叠 + TOP15 排名。"""
    mi_sel = set(mi_rank.head(topk).index)
    rf_sel = set(rfe_rank[rfe_rank == 1].index)
    lb_sel = set(imp.head(topk).index)
    inter = {
        '互信息 ∩ RFE': len(mi_sel & rf_sel),
        '互信息 ∩ LightGBM': len(mi_sel & lb_sel),
        'RFE ∩ LightGBM': len(rf_sel & lb_sel),
        '三者交集': len(mi_sel & rf_sel & lb_sel),
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    axes[0].bar(list(inter.keys()), list(inter.values()), color='#2c7fb8')
    for i, v in enumerate(inter.values()):
        axes[0].text(i, v, str(v), ha='center', va='bottom', fontsize=10)
    axes[0].set_title('三种方法选中集合重叠（各取 TOP%d）' % topk, fontsize=11)
    axes[0].set_ylabel('共同特征数')
    axes[0].tick_params(axis='x', rotation=15, labelsize=9)

    top = list(imp.head(15).index)[::-1]
    n = max(len(mi_rank), 1)
    y = np.arange(len(top))
    h = 0.26
    lb_rank = imp.rank(ascending=False)
    axes[1].barh(y + h, (mi_rank.reindex(top) / n).values, h, label='互信息排名', color='#41b6c4')
    axes[1].barh(y, (rfe_rank.reindex(top) / n).values, h, label='RFE 排名', color='#fe9929')
    axes[1].barh(y - h, (lb_rank.reindex(top) / n).values, h, label='LightGBM 排名', color='#2c7fb8')
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(top, fontsize=8)
    axes[1].set_xlabel('归一化排名（越小越优）')
    axes[1].set_title('TOP15 特征在三种方法下的排名', fontsize=11)
    axes[1].legend(fontsize=9)
    fig.suptitle('特征选择三种方法对比（%s）' % tag, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '04_selection_compare_%s.png' % tag), dpi=150)
    plt.close(fig)
    pd.DataFrame([{'对比': k, '共同特征数': v} for k, v in inter.items()]).to_csv(
        os.path.join(TAB_DIR, '04_selection_overlap_%s.csv' % tag), index=False, encoding='utf-8-sig')
    print('已输出特征选择对比图：04_selection_compare_%s.png' % tag)


def plot_split_comparison(stats_list):
    """双策略切分对比图：违约率柱状图 + 样本量柱状图，并落盘统计表。"""
    tags = [s['tag'] for s in stats_list]
    parts = ['train', 'valid', 'test']
    colors = {'train': '#2c7fb8', 'valid': '#41b6c4', 'test': '#fe9929'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(tags))
    w = 0.25
    for i, p in enumerate(parts):
        vals = [s['%s_def' % p] for s in stats_list]
        axes[0].bar(x + (i - 1) * w, vals, w, label=p, color=colors[p])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tags)
    axes[0].set_ylabel('违约率 (%)')
    axes[0].set_title('各切分策略 train/val/test 违约率')
    axes[0].legend()
    for i, p in enumerate(parts):
        ns = [s['%s_n' % p] for s in stats_list]
        axes[1].bar(x + (i - 1) * w, ns, w, label=p, color=colors[p])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tags)
    axes[1].set_ylabel('样本量')
    axes[1].set_title('各切分策略样本量分布')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, '04_split_comparison.png'), dpi=150)
    plt.close(fig)
    print('已输出双策略切分对比图：04_split_comparison.png')

    pd.DataFrame(stats_list).to_csv(os.path.join(TAB_DIR, '04_split_stats.csv'),
                                    index=False, encoding='utf-8-sig')
    print('已落盘切分统计：04_split_stats.csv')


# ---------------------------------------------------------------------------
# 7. 补充实验：未使用字段补漏 + 组合特征构造与分组验证
# ---------------------------------------------------------------------------
SUPP_STRATEGIES = [('random', '随机分层切分'), ('oot_y2013', '时间切分（OOT）')]

# 补充实验验证模型（XGBoost，加大树数并早停）
SUPP_XGB = dict(max_depth=6, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=3, reg_lambda=5,
                n_estimators=2000, tree_method='hist', eval_metric='auc',
                early_stopping_rounds=60, n_jobs=-1,
                random_state=RANDOM_STATE)


def load_split(tag):
    """读取已持久化的切分索引，返回 (train_idx, val_idx, test_idx)；缺失时按该标签重新生成。"""
    p = os.path.join(SPLIT_DIR, '%s_split.csv' % tag)
    if not os.path.exists(p):
        df = pd.read_parquet(IN_PATH)
        if tag == 'random':
            split_data(df, strategy='random')
        elif tag.startswith('oot_y'):
            split_data(df, strategy='oot', oot_train_start_year=int(tag[5:]))
        else:
            raise ValueError('未知切分标签: ' + tag)
    s = pd.read_csv(p)
    out = []
    for part in ('train', 'val', 'test'):
        out.append(s.loc[s['split'] == part, 'orig_index'].values)
    return tuple(out)


def add_missing_features(df):
    """为原字段构造衍生特征（FICO 区间 / 首次放款 / n9、n10）。"""
    out = df.copy()
    # 1) FICO 区间：区间宽度反映评估不确定度
    if 'ficoRangeLow' in out.columns:
        low = out['ficoRangeLow']
        high = out['ficoRangeHigh']
        out['fico_low'] = low
        out['fico_high'] = high
        out['fico_range_width'] = high - low
        out['fico_low_log'] = np.log1p(low.clip(lower=0))
        out['fico_high_log'] = np.log1p(high.clip(lower=0))
        # 相对位置：借款人处在本区间内的偏低还是偏高端
        out['fico_pos_in_range'] = (out.get('fico_mean', low) - low) / (
            (high - low).replace(0, np.nan))
    # 2) earliesCreditLine：首次放款时间（信用历史长度）
    if 'earliesCreditLine' in out.columns:
        e = pd.to_datetime(out['earliesCreditLine'], errors='coerce')
        out['earliest_year'] = e.dt.year.astype('float32')
        out['earliest_year_bin'] = pd.cut(
            e.dt.year, bins=[0, 1995, 2000, 2005, 2010, 2015, 2100],
            labels=[0, 1, 2, 3, 4, 5]).astype('float32')
        if 'issueDate' in out.columns:
            iy = pd.to_datetime(out['issueDate'], errors='coerce').dt.year
            out['credit_age_years'] = (iy - e.dt.year).astype('float32')
        out['earliest_missing'] = e.isna().astype('int8')
    # 3) n9 / n10：缺失本身即信号
    for c in ['n9', 'n10']:
        if c in out.columns:
            out[c] = out[c].astype('float32')
            out['isna_%s' % c] = out[c].isna().astype('int8')
            out['%s_log' % c] = np.log1p(out[c].clip(lower=0))
    return out


def build_combo_features(df, ref=None):
    """按信贷业务逻辑构造组合特征（还款压力 / 信用历史 / 额度使用 / 贷款条件等）。

    ref 为用于确定分位阈值的参考集（只传训练集）：所有 quantile 阈值只在 ref 上
    计算后套用到 df。
    """
    out = df.copy()
    ref = out if ref is None else ref

    def qref(col, k, series):
        """等频二值标记，阈值只在 ref 上取。"""
        thr = ref[col].quantile(k)
        return (series > thr).astype('int8')

    # ---- 组合1：还款压力（收入侧）----
    if 'loan_to_income' in out.columns:
        out['combo_loan_income_hi'] = qref('loan_to_income', 0.7, out['loan_to_income'])
    if 'installment_to_income' in out.columns:
        out['combo_installment_income_hi'] = qref('installment_to_income', 0.7,
                                                  out['installment_to_income'])
    if 'annualIncome' in out.columns:
        out['combo_low_income'] = (out['annualIncome'] <
                                   ref['annualIncome'].quantile(0.3)).astype('int8')
    if 'loanAmnt' in out.columns:
        out['combo_high_loan'] = qref('loanAmnt', 0.7, out['loanAmnt'])
    if 'combo_low_income' in out and 'combo_high_loan' in out:
        out['combo_lowinc_highloan'] = (out['combo_low_income'] &
                                        out['combo_high_loan']).astype('int8')

    # ---- 组合2：信用历史质量 ----
    if 'delinquency_2years' in out.columns:
        out['combo_any_delinq'] = (out['delinquency_2years'] > 0).astype('int8')
    if 'pubRec' in out.columns:
        out['combo_has_pubrec'] = (out['pubRec'] > 0).astype('int8')
    if 'combo_any_delinq' in out and 'combo_has_pubrec' in out:
        out['combo_delinq_and_pubrec'] = (out['combo_any_delinq'] &
                                          out['combo_has_pubrec']).astype('int8')
    if 'credit_age_years' in out.columns:
        out['combo_short_history'] = (out['credit_age_years'] < 6).astype('int8')

    # ---- 组合3：信用额度使用 ----
    if 'revolUtil' in out.columns:
        out['combo_high_revolutil'] = qref('revolUtil', 0.7, out['revolUtil'])
    if 'openAcc' in out.columns and 'totalAcc' in out.columns:
        out['open_acc_ratio'] = (out['openAcc'] /
                                 out['totalAcc'].replace(0, np.nan)).astype('float32')
        out['combo_many_open_acc'] = qref('openAcc', 0.7, out['openAcc'])
    if 'revolBal_to_income' in out.columns:
        out['combo_high_revbal_income'] = (
            out['revolBal_to_income'] >
            ref['revolBal_to_income'].quantile(0.7)).astype('int8')

    # ---- 组合4：贷款条件（利率 x 期限的联合定价）----
    if 'term' in out.columns and 'interestRate' in out.columns:
        out['combo_longterm'] = (out['term'] > ref['term'].median()).astype('int8')
        out['combo_high_rate'] = qref('interestRate', 0.7, out['interestRate'])
        out['combo_worst_terms'] = (out['combo_longterm'] &
                                    out['combo_high_rate']).astype('int8')
    if 'dti' in out.columns:
        out['combo_high_dti'] = qref('dti', 0.7, out['dti'])
    if 'combo_high_dti' in out and 'combo_worst_terms' in out:
        out['combo_dti_worst'] = (out['combo_high_dti'] &
                                  out['combo_worst_terms']).astype('int8')

    # ---- 组合5：风险因子计数（聚合型：坏信号越多风险越高）----
    risk_flags = [c for c in out.columns if c.startswith('combo_') and
                  c not in ('combo_delinq_and_pubrec', 'combo_dti_worst',
                            'combo_lowinc_highloan', 'combo_worst_terms')]
    if risk_flags:
        out['risk_flag_count'] = out[risk_flags].sum(axis=1).astype('float32')
        out['risk_flag_count_log'] = np.log1p(out['risk_flag_count'])

    # ---- 组合6：FICO 相关交互 ----
    if 'fico_mean' in out.columns and 'combo_high_dti' in out:
        out['combo_lowfico_highdti'] = ((out['fico_mean'] <
                                         ref['fico_mean'].quantile(0.3)) &
                                        out['combo_high_dti']).astype('int8')
    if 'fico_mean' in out.columns and 'annualIncome' in out.columns:
        out['combo_lowfico_lowinc'] = (
            (out['fico_mean'] < out['fico_mean'].quantile(0.3)) &
            (out['annualIncome'] < ref['annualIncome'].quantile(0.3))).astype('int8')

    # ---- 组合7：缺失聚集度（缺失本身就是风险信号）----
    na_cols = [c for c in ['n9', 'n10', 'n11', 'n2', 'n4', 'n5', 'n6', 'n7', 'n8',
                           'n12', 'n13', 'n14', 'employmentLength'] if c in out.columns]
    if na_cols:
        out['na_cluster_count'] = out[na_cols].isna().sum(axis=1).astype('float32')
    return out


def univariate_table(df, new_cols):
    """新特征的单变量区分度：类别型取组内违约率极差，数值型按十分位分组给单变量 AUC。"""
    y = df[TARGET]
    rows = []
    for c in new_cols:
        s = df[c]
        try:
            if s.nunique() <= 12:
                g = df.groupby(c, observed=True)[TARGET].agg(['mean', 'size'])
                spread = float(g['mean'].max() - g['mean'].min()) if len(g) > 1 else 0.0
                rows.append(dict(feature=c, kind='categorical', nunique=int(s.nunique()),
                                 def_rate=round(float(y.mean()), 4),
                                 group_spread=round(spread, 4), n_groups=len(g)))
            else:
                b = pd.qcut(s, 10, duplicates='drop')
                g = df.groupby(b, observed=True)[TARGET].agg(['mean', 'size'])
                spread = float(g['mean'].max() - g['mean'].min()) if len(g) > 1 else 0.0
                auc = float(roc_auc_score(y, g.loc[b.values, 'mean'].values))
                rows.append(dict(feature=c, kind='numeric', nunique=int(s.nunique()),
                                 def_rate=round(float(y.mean()), 4),
                                 group_spread=round(spread, 4),
                                 univ_auc=round(max(auc, 1 - auc), 4)))
        except Exception as exc:                                   # noqa: BLE001
            rows.append(dict(feature=c, kind='error', nunique=0, def_rate=0.0,
                             group_spread=0.0, note=str(exc)[:40]))
    return pd.DataFrame(rows)


def build_and_ablate(tag, label):
    """补充实验主流程：补漏 + 组合特征，并在验证集上做分组增量验证。"""
    print('\n' + '=' * 74)
    print('【补充实验 %s】%s' % (tag, label))
    print('=' * 74)
    old = {}
    for part in ('train', 'valid', 'test'):
        old[part] = pd.read_parquet(
            os.path.join(OUT_DIR, 'credit_fe_%s_%s.parquet' % (part, tag)))
    # 旧矩阵没有的原始字段，从 credit_clean 按切分索引取回
    tr_i, va_i, te_i = load_split(tag)
    clean = pd.read_parquet(IN_PATH)
    parts = [clean.iloc[tr_i].reset_index(drop=True),
             clean.iloc[va_i].reset_index(drop=True),
             clean.iloc[te_i].reset_index(drop=True)]
    # 校验切分索引与主流程特征矩阵行序一致
    for name, p, o in zip(('train', 'valid', 'test'), parts, old.values()):
        assert len(p) == len(o), '%s 行数不一致 %d vs %d' % (name, len(p), len(o))
        assert int(p[TARGET].sum()) == int(o[TARGET].sum()), \
            '%s 标签不一致，索引未对齐' % name
    print('  切分索引与特征矩阵对齐校验通过')
    new = {}
    for part, raw in zip(('train', 'valid', 'test'), parts):
        d = add_missing_features(raw)
        # 组合特征的分箱阈值只在 train 上确定，valid/test 套用同一阈值
        d = build_combo_features(d, ref=new['train'] if part != 'train' else None)
        new[part] = d

    old_cols = [c for c in old['train'].columns if c != TARGET]
    # 新增列 = 新构造列 + 补漏进来的原始数值列（排除目标与未编码的非数值列）
    banned = {TARGET, 'id', 'issueDate', 'earliesCreditLine', 'grade',
              'subGrade', 'title', 'employmentTitle', 'postCode',
              'regionCode', 'ficoRangeLow', 'ficoRangeHigh'}
    new_only = []
    for c in new['train'].columns:
        if c in old_cols or c == TARGET or c in banned:
            continue
        if new['train'][c].dtype == object or str(
                new['train'][c].dtype).startswith('datetime'):
            continue
        if c in ('ficoRangeLow', 'ficoRangeHigh'):
            continue
        new_only.append(c)
    print('新增特征 %d 个：%s' % (len(new_only), ', '.join(new_only)))

    tab = univariate_table(new['train'], new_only)
    tab.to_csv(os.path.join(TAB_DIR, '10_new_feature_univariate_%s.csv' % tag),
               index=False, encoding='utf-8-sig')
    print('\n新增特征区分度 TOP10（组内违约率极差）：')
    for _, r in tab.sort_values('group_spread', ascending=False).head(10).iterrows():
        print('  %-28s %.4f' % (r['feature'], r['group_spread']))

    # ---- 分组增量验证：只在验证集上比较 ----
    def build_matrix(part, extra_cols):
        """基线 = 主流程特征矩阵原样；增强 = 基线 + 指定新增列。"""
        base_part = old[part].drop(columns=[TARGET], errors='ignore')
        if not extra_cols:
            return base_part
        add_part = new[part][[c for c in extra_cols if c in new[part].columns]]
        return pd.concat([base_part, add_part], axis=1)

    def run(extra_cols):
        Xtr = build_matrix('train', extra_cols)
        Xva = build_matrix('valid', extra_cols)
        ytr = old['train'][TARGET].values
        yva = old['valid'][TARGET].values
        num = Xtr.select_dtypes(include=[np.number]).columns
        Xtr, Xva = Xtr[num], Xva[num]
        imp = SimpleImputer(strategy='median')
        Xtr_i = pd.DataFrame(imp.fit_transform(Xtr), columns=Xtr.columns)
        Xva_i = pd.DataFrame(imp.transform(Xva), columns=Xva.columns)
        clf = XGBClassifier(**SUPP_XGB)
        clf.fit(Xtr_i, ytr, eval_set=[(Xva_i, yva)], verbose=False)
        return roc_auc_score(yva, clf.predict_proba(Xva_i)[:, 1]), Xtr.shape[1]

    auc_base, n_base = run([])
    print('\n增量验证（判据 = 验证集 AUC-ROC，测试集不参与选择）：')
    print('  基线特征 %d 个        valid AUC = %.4f' % (n_base, auc_base))

    groups = {
        '补漏(n9/n10)': [c for c in new_only if c.startswith(('n9', 'n10', 'isna_n9', 'isna_n10'))],
        '补漏(FICO区间)': [c for c in new_only if c.startswith('fico_')],
        '补漏(首次放款)': [c for c in new_only if c in
                      ('earliest_year', 'earliest_year_bin', 'credit_age_years',
                       'earliest_missing')],
        '组合(还款压力)': [c for c in new_only if c in
                      ('combo_low_income', 'combo_high_loan',
                       'combo_lowinc_highloan', 'open_acc_ratio')],
        '组合(信用历史)': [c for c in new_only if c in
                      ('combo_any_delinq', 'combo_has_pubrec',
                       'combo_delinq_and_pubrec', 'combo_short_history')],
        '组合(贷款条件)': [c for c in new_only if c.startswith(('combo_longterm',
                             'combo_high_rate', 'combo_worst_terms',
                             'combo_high_dti', 'combo_dti_worst'))],
        '组合(FICO×负债)': [c for c in new_only if c in
                       ('combo_lowfico_highdti', 'combo_lowfico_lowinc')],
        '组合(额度使用)': [c for c in new_only if c in
                      ('combo_high_revolutil', 'combo_many_open_acc',
                       'combo_high_revbal_income')],
        '组合(风险计数)': [c for c in new_only if c.startswith(
            ('risk_flag_count', 'na_cluster_count'))],
    }
    group_rows = []
    print('\n分组增量验证（只加该组，其余保持基线）：')
    for gname, cols in groups.items():
        cols = [c for c in cols if c in new['train'].columns]
        if not cols:
            continue
        a, n = run(cols)
        group_rows.append(dict(group=gname, n_feat=len(cols),
                               auc_valid=round(a, 4), delta=round(a - auc_base, 4)))
        print('  %-18s +%2d 个   valid AUC = %.4f  (%+.4f)'
              % (gname, len(cols), a, a - auc_base))
    gdf = pd.DataFrame(group_rows).sort_values('delta', ascending=False)
    gdf.to_csv(os.path.join(TAB_DIR, '10_group_increment_%s.csv' % tag),
               index=False, encoding='utf-8-sig')

    # 只叠加"有正贡献"的组，得到最优子集
    good = list(gdf[gdf['delta'] > 0]['group'])
    if good:
        keep_cols = []
        for g in good:
            keep_cols += groups[g]
        a, n = run(keep_cols)
        print('\n只加正贡献组（%s）：valid AUC = %.4f (%+.4f)'
              % (', '.join(good), a, a - auc_base))
    else:
        a, n = auc_base, n_base
        print('\n无分组产生正贡献，新特征整体无效')

    auc_all, n_all = run(new_only)
    print('  基线 %d 个 + 全部新特征 %d 个  valid AUC = %.4f  (%+.4f)'
          % (n_base, len(new_only), auc_all, auc_all - auc_base))
    print('\n单变量区分度最高的 3 个新特征：')
    for _, r in tab.sort_values('group_spread', ascending=False).head(3).iterrows():
        print('  %-24s 组内违约率极差 %.4f' % (r['feature'], r['group_spread']))

    return dict(strategy=tag, strategy_name=label, n_base=n_base,
                auc_valid_base=round(auc_base, 4),
                auc_valid_all=round(auc_all, 4),
                delta_all=round(auc_all - auc_base, 4),
                auc_valid_best=round(a, 4),
                delta_best=round(a - auc_base, 4),
                good_groups='; '.join(good) if good else '无',
                n_new=len(new_only))


def run_supplement():
    """对两种切分策略执行补充实验并落盘汇总表（10_ablation_valid.csv）。"""
    rows = [build_and_ablate(tag, label) for tag, label in SUPP_STRATEGIES]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB_DIR, '10_ablation_valid.csv'),
              index=False, encoding='utf-8-sig')
    print('\n' + '=' * 74)
    print('补充实验汇总（验证集口径）')
    print('=' * 74)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(strategy='random', oot_train_start_year=None):
    """按指定切分策略执行完整特征工程，返回该策略的统计信息 dict。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    tag = strategy if oot_train_start_year is None else f"{strategy}_y{oot_train_start_year}"

    df = pd.read_parquet(IN_PATH)
    print('\n' + '=' * 70)
    print('信贷违约预测 · 第 3 阶段 特征工程（切分策略：%s）' % tag)
    print('=' * 70)
    print('输入数据：%s' % (df.shape,))
    print('全量违约率：%.4f%%' % (df[TARGET].mean() * 100))

    # 获取切分索引（RangeIndex，对应 credit_clean 行序）
    train_idx, val_idx, test_idx = split_data(
        df, strategy=strategy, oot_train_start_year=oot_train_start_year)
    train = df.loc[train_idx].reset_index(drop=True)
    valid = df.loc[val_idx].reset_index(drop=True)
    test = df.loc[test_idx].reset_index(drop=True)
    print('\n按 %s 切分（70/15/15）：' % tag)
    describe_split('训练集', train)
    describe_split('验证集', valid)
    describe_split('测试集', test)
    if oot_train_start_year is not None:
        print('oot 训练窗口自 %d 年起，测试集为时间上最近的样本' % oot_train_start_year)

    # 2. 类别特征编码
    print('\n--- 任务 3.1 类别特征编码 ---')
    encode_ordinal(train, [valid, test])
    (train, valid, test), onehot_cats = encode_onehot(train, [valid, test])
    te_mapping, te_global = fit_target_encoding(train)
    valid, test = apply_target_encoding([('验证集', valid), ('测试集', test)],
                                        te_mapping, te_global)
    category_default_rate_figure(train, tag)
    onehot_total = int(sum(len(v) for v in onehot_cats.values()))
    enc_summary = pd.DataFrame([
        {'方法': '序数编码', '作用列': 'grade, subGrade', '新增列数': 2},
        {'方法': '独热编码', '作用列': ', '.join(onehot_cats.keys()), '新增列数': onehot_total},
        {'方法': '目标编码（K 折 OOF）', '作用列': ', '.join(TE_COLS), '新增列数': len(TE_COLS)},
    ])
    enc_summary.to_csv(os.path.join(TAB_DIR, '04_encoding_summary_%s.csv' % tag),
                       index=False, encoding='utf-8-sig')

    # 3. 数值特征标准化/归一化 + 特征创建
    print('\n--- 任务 3.2 数值特征标准化/归一化 + 任务 3.3 特征创建 ---')
    train, valid, test = create_features(train, [valid, test])
    bin_edges = fit_binning(train)
    train, valid, test = apply_binning(train, [valid, test], bin_edges)
    binning_figure(train, tag)
    raw_snapshot = train[['interestRate', 'dti']].copy()   # 缩放前快照，用于前后分布对比图
    scaler = fit_scaler(train)
    train, valid, test = apply_scaler(train, [valid, test], scaler)
    scaling_compare_figure(raw_snapshot, train, tag)
    scale_rows = []
    for i, c in enumerate(SCALE_COLS):
        scale_rows.append({'列': c, '原始均值': round(float(raw_snapshot[c].mean()), 4)
                           if c in raw_snapshot else None,
                           '原始标准差': round(float(raw_snapshot[c].std()), 4)
                           if c in raw_snapshot else None,
                           '缩放后均值': round(float(scaler.mean_[i]), 4),
                           '缩放后标准差': round(float(scaler.scale_[i]), 4)})
    pd.DataFrame(scale_rows).to_csv(os.path.join(TAB_DIR, '04_scaling_stats_%s.csv' % tag),
                                    index=False, encoding='utf-8-sig')
    (train, valid, test), poly_names = build_polynomial(train, [valid, test])

    # 4. 特征选择（只在训练集上进行）
    print('\n--- 任务 3.4 特征选择 ---')
    X_train, y_train = feature_matrix(train)
    X_valid, y_valid = feature_matrix(valid)
    X_test, y_test = feature_matrix(test)
    print('训练集特征矩阵：%s（特征 %d 列）' % (X_train.shape, X_train.shape[1]))
    print('验证集：%s｜测试集：%s' % (X_valid.shape, X_test.shape))

    corr_drop, mi_rank, corr_pairs = select_filter(X_train, y_train)
    X_train_sel = X_train.drop(columns=corr_drop)
    X_valid_sel = X_valid.drop(columns=corr_drop)
    X_test_sel = X_test.drop(columns=corr_drop)
    print('剔除高相关列后：%s' % (X_train_sel.shape,))

    rfe_selected, rfe_rank = select_wrapper(X_train_sel, y_train)
    model, imp = select_embedded(X_train_sel, y_train)
    selection_compare_figure(mi_rank, rfe_rank, imp, tag)

    # 三种方法对比表
    compare = pd.DataFrame({
        '互信息': mi_rank.reindex(X_train_sel.columns),
        'RFE入选': ['是' if c in rfe_selected else '' for c in X_train_sel.columns],
        'RFE排名': rfe_rank.reindex(X_train_sel.columns),
        'LGBM_gain': imp.reindex(X_train_sel.columns),
    })
    compare['LGBM排名'] = compare['LGBM_gain'].rank(ascending=False).astype(int)
    compare = compare.sort_values('LGBM_gain', ascending=False)
    compare.index.name = '特征'
    compare.to_csv(os.path.join(TAB_DIR, '04_selection_compare_%s.csv' % tag),
                   encoding='utf-8-sig')
    print('三种方法 TOP15 对比（按 LGBM gain 排序，%s）：' % tag)
    print(compare.head(15).round(5).to_string())

    # 5. 特征重要性分析
    print('\n--- 特征重要性分析 ---')
    imp.to_csv(os.path.join(TAB_DIR, '04_importance_lgbm_%s.csv' % tag),
               encoding='utf-8-sig', header=['gain重要性'])
    importance_analysis(model, X_train_sel, y_train,
                       [('验证集', X_valid_sel, y_valid), ('测试集', X_test_sel, y_test)], imp, tag)

    # 6. 结果保存与回读验证
    edges_df = pd.DataFrame({c: pd.Series(e) for c, e in bin_edges.items()})
    edges_df.to_csv(os.path.join(TAB_DIR, '04_bin_edges_%s.csv' % tag), index=False,
                    encoding='utf-8-sig')
    te_rows = []
    for c in TE_COLS:
        for value, enc in te_mapping[c].items():
            te_rows.append({'列': c, '取值': value, '目标编码': round(float(enc), 6)})
    pd.DataFrame(te_rows).to_csv(os.path.join(TAB_DIR, '04_target_encoding_%s.csv' % tag),
                                 index=False, encoding='utf-8-sig')

    keep_cols = list(X_train_sel.columns) + [TARGET]
    paths = []
    for name, d in [('train', train), ('valid', valid), ('test', test)]:
        p = os.path.join(OUT_DIR, 'credit_fe_%s_%s.parquet' % (name, tag))
        d[keep_cols].to_parquet(p, index=False)
        paths.append(p)
        print('已保存：%s（%.1f MB）' % (p, os.path.getsize(p) / 1024 / 1024))

    print('\n回读验证（%s）：' % tag)
    for name, p in zip(['train', 'valid', 'test'], paths):
        chk = pd.read_parquet(p)
        print('  %-6s %s｜违约率 %.4f%%' % (name, chk.shape, chk[TARGET].mean() * 100))
    print('最终特征数：%d' % X_train_sel.shape[1])

    source_counts = {
        '序数编码': 2,
        '独热编码': onehot_total,
        '目标编码': len(TE_COLS),
        'log1p 变换': len(LOG1P_COLS),
        '组合/派生特征': 7,
        '等频分箱': len(BIN_COLS),
        '多项式特征': len(poly_names),
    }
    feature_sources_figure(source_counts, tag)
    pd.DataFrame([{'特征来源': k, '新增列数': v} for k, v in source_counts.items()]).to_csv(
        os.path.join(TAB_DIR, '04_feature_sources_%s.csv' % tag), index=False, encoding='utf-8-sig')

    return {
        'tag': tag,
        'n_features': int(X_train_sel.shape[1]),
        'train_n': len(train), 'valid_n': len(valid), 'test_n': len(test),
        'train_def': round(train[TARGET].mean() * 100, 4),
        'valid_def': round(valid[TARGET].mean() * 100, 4),
        'test_def': round(test[TARGET].mean() * 100, 4),
        'train_ymin': int(train['issueDate'].dt.year.min()),
        'train_ymax': int(train['issueDate'].dt.year.max()),
        'test_ymin': int(test['issueDate'].dt.year.min()),
        'test_ymax': int(test['issueDate'].dt.year.max()),
    }


if __name__ == '__main__':
    all_stats = []
    for s, kw in [('random', {}), ('oot', {'oot_train_start_year': 2013})]:
        all_stats.append(main(s, **kw))
    plot_split_comparison(all_stats)
    # 补充实验（特征工程报告第 6 章）：补漏 + 组合特征构造与分组验证
    run_supplement()
