# -*- coding: utf-8 -*-
"""信贷违约预测项目：模型训练、评估与业务解释（第 4—5 阶段）

输入：db/processed/credit_fe_{part}_{tag}.parquet（第 3 阶段双策略特征产物）
输出：outputs/tables/05_*.csv    多模型性能、扩展指标、Lift、PSI、两策略落差
      outputs/tables/08b_*.csv   特征组合消融
      outputs/tables/09_*.csv    SHAP 全局与局部解释、业务规则、跨期重要性差异
      outputs/figures/05_*.png   ROC/PR、指标对比、KS、混淆矩阵、阈值、XGBoost 重要性
      outputs/figures/08b_*.png  特征组合消融
      outputs/figures/09_*.png   SHAP 条形图、蜂群图、瀑布图、依赖图、规则画像

功能：
1. 读取两套切分特征，分别独立建模（逻辑回归 / 随机森林 / XGBoost）
2. 缺失值中位数填补（仅在训练集 fit）
3. 性能评估：准确率/精确率/召回率/F1/AUC-ROC/KS/PR-AUC/Gini/LogLoss/Brier/混淆矩阵
4. 业务评估：十等分 Lift 与累计增益、阈值扫描
5. 稳定性评估：PSI（验证集→测试集、随机测试 vs OOT 测试）
6. 图表：ROC/PR、指标对比、KS、混淆矩阵、阈值扫描、XGBoost 重要性、Lift、双策略对比
7. 特征贡献分析：XGBoost_base 口径的特征组合消融
8. 模型解释：TreeExplainer 全局与局部归因、业务规则挖掘、跨期重要性对比

运行环境：pandas / numpy / scikit-learn / xgboost / shap / pyarrow / matplotlib
运行方式：python model_training.py            # 训练评估 + SHAP 解释
          python model_training.py random     # 只跑随机切分
          python model_training.py shap       # 只跑 SHAP 解释
"""

import json
import os
import re
import sys
import time
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def pick_device(verbose=True):
    """返回 'cuda' 或 'cpu'，GPU 不可用时回退 CPU。"""
    try:
        import xgboost as xgb

        X = np.random.RandomState(0).rand(2000, 20)
        y = (X[:, 0] > 0.5).astype(int)
        p = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 3,
            'nthread': 1,
        }
        xgb.train(p, xgb.DMatrix(X, label=y), num_boost_round=2)
        if verbose:
            print('计算设备：GPU（cuda）')
        return 'cuda'
    except Exception as e:
        if verbose:
            print('计算设备：CPU（GPU 不可用：%s）' % str(e)[:90])
        return 'cpu'


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def find_base():
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(cur, '项目资料', 'credit_data(1).csv')):
            return cur
        cur = os.path.dirname(cur)
    raise FileNotFoundError('未找到项目根目录')


DEVICE = pick_device(verbose=False)

BASE = find_base()
PROC_DIR = os.path.join(BASE, 'db', 'processed')
TAB_DIR = os.path.join(BASE, 'outputs', 'tables')
FIG_DIR = os.path.join(BASE, 'outputs', 'figures')


RANDOM_STATE = 42
TARGET = 'isDefault'
THRESHOLD = 0.5
MODEL_COLORS = {'逻辑回归': '#2563EB', '随机森林': '#16A34A', 'XGBoost': '#DC2626'}
# XGBoost 基线参数（表 3.1 与第 5 章特征贡献分析共用）
XGB_BASE_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    eval_metric='auc',
    device=DEVICE,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
# 策略标签 -> 中文名
STRATEGIES = [
    ('random', '随机分层切分'),
    ('oot_y2013', '时间切分（OOT）'),
]
# ---------------------------------------------------------------------------
# 特征贡献分析（报告第 5 章）：特征组合消融，XGBoost_base 口径
# ---------------------------------------------------------------------------
# 消融分组识别用的原始字段集合
ABL_RAW_COLS = {
    'loanAmnt',
    'term',
    'interestRate',
    'installment',
    'employmentLength',
    'homeOwnership',
    'annualIncome',
    'verificationStatus',
    'purpose',
    'dti',
    'delinquency_2years',
    'openAcc',
    'pubRec',
    'pubRecBankruptcies',
    'revolBal',
    'revolUtil',
    'totalAcc',
    'initialListStatus',
    'applicationType',
    'n0',
    'n1',
    'n2',
    'n4',
    'n5',
    'n6',
    'n7',
    'n8',
    'n11',
    'n12',
    'n13',
    'n14',
}
# 全特征基线的 XGBoost 测试集 AUC（表 3.1）
ABL_EXPECT_BASE_AUC = 0.7370

# 最终采纳配置（深 6 / λ30 / mcw24，由泛化间隙控制实验选出）
FINAL = {
    'max_depth': 6,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_lambda': 30,
    'min_child_weight': 24,
    'gamma': 0,
}
FINAL_N_EST = 3000
SPARSE_THRESHOLD = 0.995


# 最终配置在 SHAP 解释中的参数写法（early_stopping_rounds 换为固定轮数）
SHAP_XGB_TUNED = dict(
    max_depth=6,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=24,
    reg_lambda=30,
    gamma=0,
    n_estimators=FINAL_N_EST,
    tree_method='hist',
    eval_metric='auc',
    early_stopping_rounds=60,
    device=DEVICE,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
# 业务命名：把技术特征名翻译成风控能看懂的话
SHAP_BIZ_NAME = {
    'grade_ord': '信用评级等级',
    'subGrade_ord': '信用评级细分等级',
    'term': '贷款期限',
    'interestRate': '实际利率',
    'installment': '月还款额',
    'loanAmnt': '贷款金额',
    'annualIncome': '年收入',
    'employmentLength': '就业年限',
    'homeOwnership': '住房所有权',
    'verificationStatus': '收入验证方式',
    'purpose': '贷款用途',
    'dti': '负债收入比',
    'delinquency_2years': '近2年逾期次数',
    'openAcc': '未结账户数',
    'pubRec': '公共记录数',
    'pubRecBankruptcies': '破产公共记录数',
    'revolBal': '循环信贷余额',
    'revolUtil': '循环信贷使用率',
    'totalAcc': '总账户数',
    'initialListStatus': '初始清单状态',
    'applicationType': '申请类型',
    'fico_mean': 'FICO评分均值',
    'loan_to_income': '贷款额/年收入',
    'installment_to_income': '月还款/月收入',
    'revolBal_to_income': '循环信贷余额/年收入',
    'credit_hist_days': '信用历史天数',
    'loanAmnt_log': '贷款金额(对数)',
    'annualIncome_log': '年收入(对数)',
    'revolBal_log': '循环信贷余额(对数)',
    'delinquency_2years_log': '近2年逾期(对数)',
    'issue_year': '发放年份',
    'issue_month': '发放月份',
    'title_te': '职位目标编码',
    'employmentTitle_te': '雇佣职位目标编码',
    'postCode_te': '邮编目标编码',
    'regionCode_te': '地区目标编码',
    'isna_n_main': '主要字段缺失标记',
    'isna_n4n10': 'n4-n10缺失标记',
    'isna_n11': 'n11缺失标记',
    'isna_emp_len': '就业年限缺失标记',
}


def biz(col):
    return SHAP_BIZ_NAME.get(col, col)


# ---------------------------------------------------------------------------
def shap_load(tag):
    out = []
    for part in ('train', 'valid', 'test'):
        p = os.path.join(PROC_DIR, 'credit_fe_%s_%s.parquet' % (part, tag))
        df = pd.read_parquet(p)
        out.append(df)
    tr, va, te = out
    y_tr, y_va, y_te = (d.pop(TARGET).values for d in (tr, va, te))
    cols = list(tr.columns)
    imp = SimpleImputer(strategy='median')
    X_tr = pd.DataFrame(imp.fit_transform(tr), columns=cols)
    X_va = pd.DataFrame(imp.transform(va), columns=cols)
    X_te = pd.DataFrame(imp.transform(te), columns=cols)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def shap_fit_model(X_tr, y_tr, X_va, y_va):
    clf = XGBClassifier(**SHAP_XGB_TUNED)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return clf


# ---------------------------------------------------------------------------
def fig_beeswarm(sv, X, tag, label, topk=20):
    """蜂群图：每个特征对违约预测的贡献分布。"""
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    cols = list(imp.head(topk).index)
    sub = sv[:, [X.columns.get_loc(c) for c in cols]]
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sub, X[cols], show=False, max_display=topk, color_bar=True)
    plt.title(
        'SHAP 蜂群图：特征对违约预测的贡献分布（%s）\n'
        '横轴=SHAP值（>0推高违约风险，<0降低），颜色=特征值大小' % label,
        fontsize=11,
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, '09_shap_beeswarm_%s.png' % tag)
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    return out


def fig_bar(imp, tag, label, topk=20):
    """全局重要性条形图。"""
    d = imp.sort_values(ascending=False).head(topk).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#DC2626' if v > 0 else '#2563EB' for v in d.values]
    ax.barh(range(len(d)), d.values, color=colors)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([biz(c) for c in d.index], fontsize=9)
    ax.set_xlabel('平均 |SHAP 值|（对违约预测的平均影响强度）')
    ax.set_title('SHAP 全局特征重要性 TOP%d（%s）' % (topk, label), fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(d.values):
        ax.text(v + max(d.values) * 0.01, i, '%.4f' % v, va='center', fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '09_shap_bar_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    return out


def fig_waterfall(sv, X, y, p, tag, label, k=2):
    """局部瀑布图：违约概率最高/最低各 k 个样本的逐特征拆解。"""
    hi = np.argsort(-p)[:k]
    lo = np.argsort(p)[:k]
    fig, axes = plt.subplots(2, k, figsize=(7.2 * k, 7.5))
    axes = np.atleast_2d(axes)
    for row, (idxs, kind) in enumerate([(hi, '高风险'), (lo, '低风险')]):
        for j, i in enumerate(idxs):
            ax = axes[row][j]
            contrib = pd.Series(sv[i], index=X.columns)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:12])
            order = top.sort_values()
            colors = ['#DC2626' if v > 0 else '#2563EB' for v in order.values]
            base = p[i] - sv[i].sum()
            cum = base + order.values.cumsum()
            ax.barh(range(len(order)), order.values, color=colors, alpha=0.9)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([biz(c) for c in order.index], fontsize=8)
            ax.set_title(
                '样本#%d 预测违约概率 %.3f（实际%s）\n累计到 %.3f'
                % (i, p[i], '违约' if y[i] == 1 else '正常', cum[-1]),
                fontsize=10,
            )
            ax.grid(axis='x', alpha=0.3)
    fig.suptitle('SHAP 局部解释：%s 个体风险拆解（%s）' % (label, tag), fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '09_shap_waterfall_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    return out


def fig_dependence(sv, X, tag, label, feats, tagname='random'):
    """依赖图：SHAP 值随特征值的变化趋势（风险方向可视化）。"""
    n = len(feats)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        c = X.columns.get_loc(f)
        sc = ax.scatter(X[f], sv[:, c], c=X[f], cmap='coolwarm', s=6, alpha=0.6)
        ax.axhline(0, color='#64748B', ls='--', lw=0.8)
        ax.set_xlabel(biz(f), fontsize=9)
        ax.set_ylabel('SHAP 值', fontsize=9)
        ax.set_title('%s 对违约风险的影响' % biz(f), fontsize=10)
        plt.colorbar(sc, ax=ax, fraction=0.045)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('SHAP 依赖关系：关键特征的风险方向（%s）' % label, fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '09_shap_dependence_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    return out


# ---------------------------------------------------------------------------
def business_rules(sv, X, y, p, imp, topn=12):
    """把 SHAP 贡献翻译成可读的高/低风险人群画像。"""
    rules = []
    cols = list(imp.sort_values(ascending=False).head(topn).index)
    # 高风险：预测概率 top20%
    hi_thr = np.quantile(p, 0.80)
    hi_mask = p >= hi_thr
    lo_mask = p <= np.quantile(p, 0.20)
    for c in cols:
        ci = X.columns.get_loc(c)
        v = X[c].values
        for name, mask, direction in [
            ('高风险组', hi_mask, '风险推高'),
            ('低风险组', lo_mask, '风险压低'),
        ]:
            a, b = v[mask], v[~mask]
            # 该组在此特征上的 SHAP 均值
            s = sv[mask, ci].mean()
            rules.append(
                dict(
                    feature=c,
                    feature_cn=biz(c),
                    group=name,
                    shap_mean=round(float(s), 5),
                    group_median=round(float(np.median(a)), 4),
                    other_median=round(float(np.median(b)), 4),
                    effect=(
                        direction if s > 0 else ('风险压低' if name == '高风险组' else '风险推高')
                    ),
                )
            )
    df = pd.DataFrame(rules)
    df['abs_shap'] = df['shap_mean'].abs()
    return df.sort_values('abs_shap', ascending=False)


def fig_rules(rule_df, tag, label, topn=10):
    """高/低风险组的关键特征画像对比。"""
    hi = rule_df[rule_df['group'] == '高风险组'].head(topn)
    lo = rule_df[rule_df['group'] == '低风险组'].head(topn)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, d, name, color in [
        (axes[0], hi, '高风险人群（预测违约概率最高20%）', '#DC2626'),
        (axes[1], lo, '低风险人群（预测违约概率最低20%）', '#2563EB'),
    ]:
        d = d.sort_values('shap_mean')
        ax.barh(range(len(d)), d['shap_mean'], color=color, alpha=0.85)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([biz(c) for c in d['feature']], fontsize=9)
        ax.axvline(0, color='#475569', lw=0.8)
        ax.set_xlabel('该特征在此人群中的平均 SHAP 值（>0=推高风险）')
        ax.set_title(name, fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        for i, (s, g, o) in enumerate(zip(d['shap_mean'], d['group_median'], d['other_median'])):
            ax.text(
                s + (0.005 if s >= 0 else -0.005),
                i,
                '本组中位%.2f/其他%.2f' % (g, o),
                va='center',
                ha='left' if s >= 0 else 'right',
                fontsize=7,
            )
    fig.suptitle('SHAP 业务画像：违约风险由谁驱动（%s）' % label, fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '09_shap_rules_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    return out


# ---------------------------------------------------------------------------
def run_shap(tag, label):
    print('\n' + '=' * 70)
    print('【%s】%s' % (tag, label))
    print('=' * 70)
    X_tr, X_va, X_te, y_tr, y_va, y_te = shap_load(tag)
    clf = shap_fit_model(X_tr, y_tr, X_va, y_va)
    p_test = clf.predict_proba(X_te)[:, 1]
    from sklearn.metrics import roc_auc_score

    print('模型 AUC(test) = %.4f' % roc_auc_score(y_te, p_test))

    # 采样解释：test 集 12 万行，TreeExplainer 用 3 万行足够稳定且快
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_te), size=min(30000, len(X_te)), replace=False)
    Xs = X_te.iloc[idx]
    ys = y_te[idx]
    ps = p_test[idx]

    print('计算 SHAP 值（%d 行）…' % len(Xs))
    t0 = time.time()
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[-1]
    print('  耗时 %.0fs' % (time.time() - t0))

    # 全局重要性
    imp = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False)
    imp_df = pd.DataFrame(
        dict(
            rank=range(1, len(imp) + 1),
            feature=imp.index,
            feature_cn=[biz(c) for c in imp.index],
            mean_abs_shap=imp.values,
        )
    )
    imp_df.to_csv(
        os.path.join(TAB_DIR, '09_shap_global_%s.csv' % tag), index=False, encoding='utf-8-sig'
    )
    print('全局重要性 TOP10：')
    for _, r in imp_df.head(10).iterrows():
        print('  %2d. %-22s %.4f' % (r['rank'], r['feature_cn'], r['mean_abs_shap']))

        # 局部解释表
    order = np.argsort(-ps)[:20]
    loc = []
    base = float(expl.expected_value)
    for i in order:
        contrib = pd.Series(sv[i], index=Xs.columns).sort_values(ascending=False)
        loc.append(
            dict(
                sample_no=int(i),
                prob=round(float(ps[i]), 4),
                actual=int(ys[i]),
                base_value=round(base, 4),
                top1_feature=biz(contrib.index[0]),
                top1_shap=round(float(contrib.iloc[0]), 4),
                top1_value=round(float(Xs.iloc[i][contrib.index[0]]), 3),
                top2_feature=biz(contrib.index[1]),
                top2_shap=round(float(contrib.iloc[1]), 4),
                top3_feature=biz(contrib.index[2]),
                top3_shap=round(float(contrib.iloc[2]), 4),
            )
        )
    loc_df = pd.DataFrame(loc)
    loc_df.to_csv(
        os.path.join(TAB_DIR, '09_shap_local_top20_%s.csv' % tag), index=False, encoding='utf-8-sig'
    )

    # 业务规则
    rule_df = business_rules(sv, Xs, ys, ps, imp)
    rule_df.to_csv(
        os.path.join(TAB_DIR, '09_shap_rules_%s.csv' % tag), index=False, encoding='utf-8-sig'
    )

    # 图
    figs = [
        fig_beeswarm(sv, Xs, tag, label),
        fig_bar(imp, tag, label),
        fig_waterfall(sv, Xs, ys, ps, tag, label, k=2),
    ]
    # 依赖图：取重要性前 6 的连续型特征
    cont = [c for c in imp.index if Xs[c].dtype != object][:6]
    figs.append(fig_dependence(sv, Xs, tag, label, cont))
    figs.append(fig_rules(rule_df, tag, label))

    print('图表：')
    for f in figs:
        print('  figures/%s' % os.path.basename(f))
    return imp_df, rule_df


# ---------------------------------------------------------------------------
# 数据与模型
# ---------------------------------------------------------------------------
def load_features(tag):
    """读取某策略下的三份特征矩阵，返回 (X_train, X_valid, X_test, y_train, y_valid, y_test)。"""
    out = []
    for part in ('train', 'valid', 'test'):
        p = os.path.join(PROC_DIR, 'credit_fe_%s_%s.parquet' % (part, tag))
        if not os.path.exists(p):
            raise FileNotFoundError('缺少特征文件 %s，请先运行 feature_engineering.py' % p)
        out.append(pd.read_parquet(p))
    train, valid, test = out
    y_train, y_valid, y_test = (d.pop(TARGET).values for d in (train, valid, test))
    return train, valid, test, y_train, y_valid, y_test


def impute(X_train, X_valid, X_test):
    """中位数填补缺失值，统计量仅在训练集 fit。"""
    n_before = int(X_train.isna().sum().sum())
    imp = SimpleImputer(strategy='median')
    cols = list(X_train.columns)
    X_train = pd.DataFrame(imp.fit_transform(X_train), columns=cols)
    X_valid = pd.DataFrame(imp.transform(X_valid), columns=cols)
    X_test = pd.DataFrame(imp.transform(X_test), columns=cols)
    print(
        '缺失单元（训练集）：填补前 %d -> 填补后 %d' % (n_before, int(X_train.isna().sum().sum()))
    )
    return X_train, X_valid, X_test


def build_models():
    """三个基础模型：逻辑回归（含标准化管道）、随机森林、XGBoost。"""
    return {
        # 逻辑回归 + StandardScaler 管道
        '逻辑回归': Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'clf',
                    LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, random_state=RANDOM_STATE),
                ),
            ]
        ),
        # 限制树深与叶节点最小样本数
        '随机森林': RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=40,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        # 行/列双采样
        'XGBoost': XGBClassifier(**XGB_BASE_PARAMS),
    }


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------
def ks_statistic(y_true, y_score):
    """KS 统计量 = max(累计违约占比 - 累计正常占比)，返回 (ks, cum_pos, cum_neg)。"""
    y_true = np.asarray(y_true)
    y = y_true[np.argsort(-np.asarray(y_score))]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    cum_pos = np.cumsum(y) / n_pos
    cum_neg = np.cumsum(1 - y) / n_neg
    return float(np.max(cum_pos - cum_neg)), cum_pos, cum_neg


def evaluate(y_true, y_prob, threshold=THRESHOLD):
    """阈值相关指标（准确率/精确率/召回率/F1）+ 阈值无关排序指标（AUC/PR-AUC/KS）。"""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    ks, _, _ = ks_statistic(y_true, y_prob)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'ks': ks,
    }


def extra_metrics(y_true, y_prob, threshold=THRESHOLD):
    """概率质量与混淆结构：Gini、LogLoss、Brier、特异度、混淆矩阵四格。"""
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    auc = roc_auc_score(y, p)
    p_clip = np.clip(p, 1e-7, 1 - 1e-7)  # log_loss 不接受 0/1 概率
    ll = log_loss(y, np.column_stack([1 - p_clip, p_clip]))
    brier = brier_score_loss(y, p_clip)
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    return {
        'gini': 2 * auc - 1,
        'log_loss': ll,
        'brier': brier,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def decile_lift(y_true, y_prob):
    """十等分 Lift / 累计增益：按预测违约概率降序分 10 组。"""
    y = np.asarray(y_true)
    y_sorted = y[np.argsort(-np.asarray(y_prob))]
    overall = y.mean()
    rows, cum = [], 0.0
    for i, idx in enumerate(np.array_split(np.arange(len(y)), 10)):
        seg = y_sorted[idx]
        bad = float(seg.sum())
        cum += bad
        rows.append(
            {
                'decile': i + 1,
                'n': len(seg),
                'default_rate': bad / len(seg),
                'lift': (bad / len(seg)) / overall if overall > 0 else 0.0,
                'cumulative_gain': cum / y.sum(),
            }
        )
    return pd.DataFrame(rows)


def psi(expected_p, actual_p, bins=10):
    """群体稳定性指数：两批预测分数分布的偏移。<0.1 稳定，0.1~0.25 需关注，>0.25 不稳定。"""
    expected_p, actual_p = np.asarray(expected_p), np.asarray(actual_p)
    qs = np.quantile(expected_p, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-6
    qs[-1] += 1e-6
    exp = np.clip(np.histogram(expected_p, bins=qs)[0] / len(expected_p), 1e-4, None)
    act = np.clip(np.histogram(actual_p, bins=qs)[0] / len(actual_p), 1e-4, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def fig_roc_pr(records, tag, label):
    """ROC 与 PR 曲线（验证集、测试集各一栏）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, ds in enumerate(('valid', 'test')):
        ax = axes[0][col]
        for m, y, p in records[ds]:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(
                fpr,
                tpr,
                color=MODEL_COLORS[m],
                lw=1.8,
                label='%s (AUC=%.4f)' % (m, roc_auc_score(y, p)),
            )
        ax.plot([0, 1], [0, 1], '--', color='#9CA3AF', lw=1)
        ax.set_title('ROC 曲线（%s）' % ('验证集' if ds == 'valid' else '测试集'), fontsize=11)
        ax.set_xlabel('假正例率 FPR')
        ax.set_ylabel('真正例率 TPR')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.25)

        ax = axes[1][col]
        for m, y, p in records[ds]:
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(
                rec,
                prec,
                color=MODEL_COLORS[m],
                lw=1.6,
                label='%s (PR-AUC=%.4f)' % (m, average_precision_score(y, p)),
            )
        base = float(np.mean(records[ds][0][1]))
        ax.axhline(base, ls='--', color='#9CA3AF', lw=1, label='违约率基线 %.2f%%' % (base * 100))
        ax.set_title('PR 曲线（%s）' % ('验证集' if ds == 'valid' else '测试集'), fontsize=11)
        ax.set_xlabel('召回率 Recall')
        ax.set_ylabel('精确率 Precision')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle('模型判别能力：%s' % label, fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '05_roc_pr_curves_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_metrics_compare(metrics_df, tag, label):
    """三模型 × 五项指标分组柱状图（验证集）。"""
    sub = metrics_df[metrics_df['dataset'] == 'valid'].set_index('model')
    items = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    names = ['准确率', '精确率', '召回率', 'F1', 'AUC-ROC']
    x = np.arange(len(items))
    models = list(sub.index)
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(models):
        vals = [sub.loc[m, k] for k in items]
        pos = x + i * w - 0.4 + w / 2
        ax.bar(pos, vals, w, label=m, color=MODEL_COLORS[m])
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.012, '%.3f' % v, ha='center', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, ls=':', color='#9CA3AF', lw=1)
    ax.set_ylabel('指标值')
    ax.set_title('三模型性能对比（验证集，阈值 %.2f）：%s' % (THRESHOLD, label), fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '05_metrics_compare_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_ks(records, tag, label):
    """KS 曲线（验证集）：累计违约/正常占比之差的最大值即 KS。"""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m, y, p in records['valid']:
        ks, cum_pos, cum_neg = ks_statistic(y, p)
        grid = np.linspace(0, 1, len(cum_pos))
        ax.plot(
            grid,
            cum_pos,
            color=MODEL_COLORS[m],
            lw=1.6,
            label='%s 累计违约占比 (KS=%.4f)' % (m, ks),
        )
        ax.plot(grid, cum_neg, color=MODEL_COLORS[m], lw=1.2, ls='--', alpha=0.65)
    ax.set_xlabel('按预测违约概率从高到低排序的样本占比')
    ax.set_ylabel('累计占比')
    ax.set_title('KS 曲线（验证集）：%s' % label, fontsize=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '05_ks_curve_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_confusion(records, tag, label, best_model):
    """最优模型在验证集/测试集上的混淆矩阵。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, ds in zip(axes, ('valid', 'test')):
        _, y, p = [r for r in records[ds] if r[0] == best_model][0]
        cm = confusion_matrix(y, (p >= THRESHOLD).astype(int))
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    '%d' % cm[i, j],
                    ha='center',
                    va='center',
                    fontsize=11,
                    color='white' if cm[i, j] > cm.max() / 2 else '#111827',
                )
        ax.set_xticks([0, 1], ['预测正常', '预测违约'])
        ax.set_yticks([0, 1], ['实际正常', '实际违约'])
        ax.set_title(
            '%s（%s）' % (best_model, '验证集' if ds == 'valid' else '测试集'), fontsize=11
        )
    fig.suptitle('混淆矩阵（阈值 %.2f）：%s' % (THRESHOLD, label), fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '05_confusion_matrix_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_threshold(records, tag, label, best_model):
    """阈值扫描：精确率/召回率/F1 随阈值的变化，标注 F1 最优点。"""
    _, y, p = [r for r in records['valid'] if r[0] == best_model][0]
    ths = np.arange(0.05, 0.96, 0.01)
    prec, rec, f1s = [], [], []
    for t in ths:
        pred = (p >= t).astype(int)
        prec.append(precision_score(y, pred, zero_division=0))
        rec.append(recall_score(y, pred, zero_division=0))
        f1s.append(f1_score(y, pred, zero_division=0))
    best_i = int(np.argmax(f1s))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ths, prec, color='#2563EB', lw=1.7, label='精确率 Precision')
    ax.plot(ths, rec, color='#16A34A', lw=1.7, label='召回率 Recall')
    ax.plot(ths, f1s, color='#DC2626', lw=1.9, label='F1')
    ax.axvline(THRESHOLD, ls='--', color='#9CA3AF', lw=1, label='默认阈值 %.2f' % THRESHOLD)
    ax.scatter([ths[best_i]], [f1s[best_i]], color='#DC2626', zorder=5)
    ax.annotate(
        'F1 最大 %.4f @ %.2f' % (f1s[best_i], ths[best_i]),
        (ths[best_i], f1s[best_i]),
        textcoords='offset points',
        xytext=(8, 8),
        fontsize=8.5,
    )
    ax.set_xlabel('判定阈值')
    ax.set_ylabel('指标值')
    ax.set_title('阈值对指标的影响（%s，验证集）：%s' % (best_model, label), fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '05_threshold_sweep_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_xgb_importance(imp, tag, label, topk=20):
    """XGBoost 特征重要性 TOP k。"""
    top = imp.head(topk).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top['feature'], top['importance'], color='#DC2626', alpha=0.85)
    ax.set_xlabel('重要性（分裂增益）')
    ax.set_title('XGBoost 特征重要性 TOP%d：%s' % (topk, label), fontsize=12)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '05_xgb_importance_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_decile_lift(lift_df, tag, label):
    """十等分 Lift 曲线（验证集、测试集各一栏）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, ds in zip(axes, ('valid', 'test')):
        sub = lift_df[lift_df['dataset'] == ds]
        for m in MODEL_COLORS:
            ms = sub[sub['model'] == m]
            ax.plot(
                ms['decile'], ms['lift'], marker='o', ms=4, color=MODEL_COLORS[m], lw=1.6, label=m
            )
        ax.axhline(1.0, ls='--', color='#9CA3AF', lw=1, label='基准 1.0')
        ax.set_xlabel('十等分（按预测违约概率降序，1=最高危）')
        ax.set_ylabel('Lift（该组违约率 / 整体违约率）')
        ax.set_title(
            '十等分 Lift（%s）：%s' % ('验证集' if ds == 'valid' else '测试集', label), fontsize=11
        )
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.25)
    fig.suptitle('模型区分能力：十等分 Lift', fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '05_decile_lift_%s.png' % tag)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def fig_strategy_compare(all_metrics):
    """双策略 × 三模型 AUC 对比：同一模型在两种切分上的落差即时间漂移。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    models = list(MODEL_COLORS.keys())
    tags = [t for t, _ in STRATEGIES]
    names = [n for _, n in STRATEGIES]
    x = np.arange(len(models))
    w = 0.36
    for ax, (ds, dsname) in zip(axes, (('valid', '验证集'), ('test', '测试集'))):
        for i, (tag, name) in enumerate(zip(tags, names)):
            sub = all_metrics[(all_metrics['strategy'] == tag) & (all_metrics['dataset'] == ds)]
            vals = [float(sub[sub['model'] == m]['auc_roc'].iloc[0]) for m in models]
            bars = ax.bar(
                x + (i - 0.5) * w, vals, w, label=name, color=['#2563EB', '#F59E0B'][i], alpha=0.9
            )
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2, v + 0.004, '%.4f' % v, ha='center', fontsize=8
                )
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        lo = float(all_metrics[all_metrics['dataset'] == ds]['auc_roc'].min()) - 0.05
        ax.set_ylim(max(0, lo), 0.88)
        ax.set_ylabel('AUC-ROC')
        ax.set_title('AUC-ROC 对比（%s）' % dsname, fontsize=12)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('两种数据集划分策略的模型表现对比（随机切分 vs 时间切分 OOT）', fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '05_strategy_compare.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def feature_groups(cols):
    """按特征来源把特征列分组（序数/独热/目标编码、log、分箱、多项式、缺失指示等）。"""
    g = defaultdict(list)
    for c in cols:
        if c.startswith('poly_'):
            g['poly'].append(c)
        elif c.startswith('isna_'):
            g['isna'].append(c)
        elif c.endswith('_te'):
            g['te'].append(c)
        elif c.endswith('_log'):
            g['log'].append(c)
        elif c.endswith('_bin'):
            g['bin'].append(c)
        elif c in ('grade_ord', 'subGrade_ord'):
            g['grade'].append(c)
        elif re.match(r'^(homeOwnership|verificationStatus|purpose)_\d+$', c):
            g['onehot'].append(c)
        elif c in ABL_RAW_COLS:
            g['raw'].append(c)
        else:
            g['derived'].append(c)
    return dict(g)


def lift_at_10(y_true, y_prob):
    """十分位 Lift：预测概率最高的 10% 样本的违约率 / 整体违约率。"""
    y = np.asarray(y_true)
    idx = np.argsort(-np.asarray(y_prob))[: len(y) // 10]
    return float(y[idx].mean() / np.mean(y))


def ablation_metrics_row(name, ds, n_feat, y, p, threshold=THRESHOLD):
    """单个特征组合在单个数据集上的完整指标行。"""
    yp = (np.asarray(p) >= threshold).astype(int)
    auc = roc_auc_score(y, p)
    return {
        'model': name,
        'dataset': ds,
        'n_features': n_feat,
        'def_rate': round(float(np.mean(y)) * 100, 4),
        'auc_roc': auc,
        'ks': ks_statistic(y, p)[0],
        'gini': 2 * auc - 1,
        'pr_auc': average_precision_score(y, p),
        'log_loss': log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        'brier': brier_score_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        'accuracy': accuracy_score(y, yp),
        'precision': precision_score(y, yp, zero_division=0),
        'recall': recall_score(y, yp, zero_division=0),
        'f1': f1_score(y, yp, zero_division=0),
        'lift@10%': lift_at_10(y, p),
    }


def run_ablation(tag='random'):
    """以 XGBoost_base 重跑各特征组合，量化各组特征贡献，并核对全特征基线的 test AUC。"""
    print('\n' + '=' * 78)
    print('特征贡献分析：特征组合消融（XGBoost_base，%s 切分）' % tag)
    print('=' * 78)
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_features(tag)
    X_train, X_valid, X_test = impute(X_train, X_valid, X_test)
    cols = list(X_train.columns)
    datasets = [('train', X_train, y_train), ('valid', X_valid, y_valid), ('test', X_test, y_test)]
    print('  特征 %d 列' % len(cols))

    g = feature_groups(cols)

    # TOP40 用基线模型的重要性排序
    print('训练 XGBoost_base 取全特征重要性…')
    clf = XGBClassifier(**XGB_BASE_PARAMS)
    clf.fit(X_train, y_train)
    imp = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
    top40 = list(imp.head(40).index)
    imp.to_csv(os.path.join(TAB_DIR, '08b_importance_base_%s.csv' % tag), encoding='utf-8-sig')

    subsets = [
        ('全特征基线', cols),
        ('去掉信用评级', [c for c in cols if c not in g['grade']]),
        ('仅信用评级', list(g['grade'])),
        ('仅原始字段', sorted(g['raw'] + g['isna'])),
        ('去目标编码', [c for c in cols if c not in g['te']]),
        ('去多项式交互', [c for c in cols if c not in g['poly']]),
        ('去独热编码', [c for c in cols if c not in g['onehot']]),
        ('去时间字段', [c for c in cols if c not in ('issue_year', 'issue_month')]),
        ('紧凑TOP40', top40),
    ]

    rows = []
    print('\n%-12s %5s %8s %8s %10s' % ('组合', '特征数', 'testAUC', 'KS', '相对基线'))
    base_auc = None
    for name, cs in subsets:
        t0 = time.time()
        m = XGBClassifier(**XGB_BASE_PARAMS)
        m.fit(X_train[cs], y_train)
        for ds, X, y in datasets:
            p = m.predict_proba(X[cs])[:, 1]
            rows.append(ablation_metrics_row(name, ds, len(cs), y, p))
        te = rows[-1]
        if base_auc is None:
            base_auc = te['auc_roc']
        print(
            '%-12s %5d %8.4f %8.4f %+10.4f  (%.0fs)'
            % (name, len(cs), te['auc_roc'], te['ks'], te['auc_roc'] - base_auc, time.time() - t0)
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(TAB_DIR, '08b_feature_subsets_base.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    # 全特征基线 test AUC（用于与 ABL_EXPECT_BASE_AUC 对齐）
    got = float(df[(df['model'] == '全特征基线') & (df['dataset'] == 'test')]['auc_roc'].iloc[0])
    print(
        '\n核对：全特征基线 test AUC = %.4f（表 3.1 的 XGBoost = %.4f）'
        % (got, ABL_EXPECT_BASE_AUC)
    )
    assert abs(got - ABL_EXPECT_BASE_AUC) < 5e-4, '与表 3.1 不一致：%.4f' % got
    print('  一致')

    # 消融柱状图
    test = df[df['dataset'] == 'test'].copy()
    test['delta'] = test['auc_roc'] - got
    order = test.sort_values('auc_roc', ascending=False)
    colors = ['#DC2626' if d >= 0 else '#9CA3AF' for d in order['delta']]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.bar(order['model'], order['auc_roc'], color=colors, alpha=0.9)
    for b, (_, r) in zip(bars, order.iterrows()):
        ax.text(
            b.get_x() + b.get_width() / 2,
            r['auc_roc'] + 0.0012,
            '%.4f\n%+0.4f' % (r['auc_roc'], r['delta']),
            ha='center',
            fontsize=8.2,
        )
    ax.axhline(got, color='#DC2626', ls='--', lw=1.2, label='全特征基线 %.4f' % got)
    ax.set_ylim(min(order['auc_roc']) - 0.012, max(order['auc_roc']) + 0.010)
    ax.set_ylabel('测试集 AUC-ROC')
    ax.set_title(
        '特征组合消融结果（XGBoost_base，%s 切分测试集）\n'
        '红色为不低于全特征基线的组合' % ('随机' if tag == 'random' else tag),
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, '08b_feature_subsets_base.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)

    print('\n产物:\n  %s\n  %s' % (out_csv, out_png))
    print('\n测试集汇总（相对全特征基线 %.4f）:' % got)
    for _, r in order.iterrows():
        print(
            '  %-12s n=%-3d AUC=%.4f KS=%.4f  %+0.4f'
            % (r['model'], r['n_features'], r['auc_roc'], r['ks'], r['delta'])
        )
    return df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_strategy(tag, label):
    """训练三模型并完成全部评估，返回 (基础指标表, 扩展指标表, Lift 表, PSI 表, 概率字典)。"""
    print('=' * 78)
    print('策略：%s（tag=%s）' % (label, tag))
    print('=' * 78)
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_features(tag)
    X_train, X_valid, X_test = impute(X_train, X_valid, X_test)
    print('特征数：%d' % X_train.shape[1])
    for nm, y in (('train', y_train), ('valid', y_valid), ('test', y_test)):
        print('  %-5s %7d 行 | 违约率 %.4f%%' % (nm, len(y), y.mean() * 100))

    datasets = {'train': (X_train, y_train), 'valid': (X_valid, y_valid), 'test': (X_test, y_test)}

    models = build_models()
    base_rows, ext_rows, lift_rows, psi_rows = [], [], [], []
    records = {'valid': [], 'test': []}
    proba = {}  # proba[(model, ds)] = 1d 概率数组
    imp_table = None
    # 按验证集 AUC 排序选定最终算法
    best_model, best_auc = None, -1.0

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_s = time.time() - t0
        print('\n[%s] 训练完成 %.1f 秒' % (name, fit_s))

        for ds, (X, y) in datasets.items():
            p = model.predict_proba(X)[:, 1]
            proba[(name, ds)] = p
            m = evaluate(y, p)
            e = extra_metrics(y, p)
            base_rows.append(
                {
                    'strategy': tag,
                    'strategy_name': label,
                    'model': name,
                    'dataset': ds,
                    'n': len(y),
                    'def_rate': round(float(y.mean()) * 100, 4),
                    'accuracy': round(m['accuracy'], 4),
                    'precision': round(m['precision'], 4),
                    'recall': round(m['recall'], 4),
                    'f1': round(m['f1'], 4),
                    'auc_roc': round(m['auc_roc'], 4),
                    'pr_auc': round(m['pr_auc'], 4),
                    'ks': round(m['ks'], 4),
                    'fit_seconds': round(fit_s, 2),
                }
            )
            ext_rows.append(
                {
                    'strategy': tag,
                    'strategy_name': label,
                    'model': name,
                    'dataset': ds,
                    'gini': round(e['gini'], 4),
                    'log_loss': round(e['log_loss'], 4),
                    'brier': round(e['brier'], 5),
                    'specificity': round(e['specificity'], 4),
                    'tp': e['tp'],
                    'fp': e['fp'],
                    'tn': e['tn'],
                    'fn': e['fn'],
                }
            )
            print(
                '  %-5s acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f ks=%.4f '
                'gini=%.4f brier=%.4f'
                % (
                    ds,
                    m['accuracy'],
                    m['precision'],
                    m['recall'],
                    m['f1'],
                    m['auc_roc'],
                    m['ks'],
                    e['gini'],
                    e['brier'],
                )
            )
            if ds in records:
                records[ds].append((name, y, p))
                df = decile_lift(y, p)
                df.insert(0, 'model', name)
                df.insert(0, 'dataset', ds)
                df.insert(0, 'strategy', tag)
                lift_rows.append(df.round(4))
                if ds == 'test':
                    psi_rows.append(
                        {
                            'strategy': tag,
                            'strategy_name': label,
                            'model': name,
                            'comparison': 'valid_vs_test',
                            'psi': round(psi(proba[(name, 'valid')], p), 4),
                        }
                    )
            if ds == 'valid' and m['auc_roc'] > best_auc:
                best_auc, best_model = m['auc_roc'], name

        if name == 'XGBoost':
            imp_table = (
                pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
                .sort_values('importance', ascending=False)
                .reset_index(drop=True)
            )

    metrics_df = pd.DataFrame(base_rows)
    ext_df = pd.DataFrame(ext_rows)
    lift_df = pd.concat(lift_rows, ignore_index=True)
    psi_df = pd.DataFrame(psi_rows)

    metrics_df.to_csv(
        os.path.join(TAB_DIR, '05_model_metrics_%s.csv' % tag), index=False, encoding='utf-8-sig'
    )
    ext_df.to_csv(
        os.path.join(TAB_DIR, '05_extended_metrics_%s.csv' % tag), index=False, encoding='utf-8-sig'
    )
    lift_df.to_csv(
        os.path.join(TAB_DIR, '05_decile_lift_%s.csv' % tag), index=False, encoding='utf-8-sig'
    )

    if imp_table is not None:
        imp_table.to_csv(
            os.path.join(TAB_DIR, '05_xgb_importance_%s.csv' % tag),
            index=False,
            encoding='utf-8-sig',
        )
        fig_xgb_importance(imp_table, tag, label)

    fig_roc_pr(records, tag, label)
    fig_metrics_compare(metrics_df, tag, label)
    fig_ks(records, tag, label)
    fig_confusion(records, tag, label, best_model)
    fig_threshold(records, tag, label, best_model)
    fig_decile_lift(lift_df, tag, label)
    print('\n最优模型（验证集 AUC）：%s（%.4f）' % (best_model, best_auc))
    return metrics_df, ext_df, lift_df, psi_df, proba


def _training_main(tags=None):
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    todo = [(t, n) for t, n in STRATEGIES if tags is None or t in tags]

    frames, ext_frames, lift_frames, psi_frames, stores = [], [], [], [], {}
    for tag, label in todo:
        mdf, edf, ldf, pdf, store = run_strategy(tag, label)
        frames.append(mdf)
        ext_frames.append(edf)
        lift_frames.append(ldf)
        psi_frames.append(pdf)
        stores[tag] = store

    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(
        os.path.join(TAB_DIR, '05_model_metrics_all.csv'), index=False, encoding='utf-8-sig'
    )
    pd.concat(ext_frames, ignore_index=True).to_csv(
        os.path.join(TAB_DIR, '05_extended_metrics_all.csv'), index=False, encoding='utf-8-sig'
    )
    pd.concat(lift_frames, ignore_index=True).to_csv(
        os.path.join(TAB_DIR, '05_decile_lift_all.csv'), index=False, encoding='utf-8-sig'
    )

    # 跨策略 PSI：随机切分测试集 vs 时间切分测试集
    cross_rows = []
    if 'random' in stores and 'oot_y2013' in stores:
        for name in MODEL_COLORS:
            cross_rows.append(
                {
                    'strategy': 'cross',
                    'strategy_name': '随机测试 vs OOT测试',
                    'model': name,
                    'comparison': 'random_test_vs_oot_test',
                    'psi': round(
                        psi(stores['random'][(name, 'test')], stores['oot_y2013'][(name, 'test')]),
                        4,
                    ),
                }
            )
    psi_all = pd.concat(psi_frames + [pd.DataFrame(cross_rows)], ignore_index=True)
    psi_all.to_csv(os.path.join(TAB_DIR, '05_psi.csv'), index=False, encoding='utf-8-sig')
    print('\nPSI（<0.1 稳定 / 0.1~0.25 关注 / >0.25 不稳定）：')
    print(psi_all.to_string(index=False))

    # 漂移对比：同一模型在随机切分与时间切分上的 AUC 落差
    if all_metrics['strategy'].nunique() > 1:
        drift = []
        for model in MODEL_COLORS:
            r = all_metrics[all_metrics['model'] == model]
            row = {'model': model}
            for ds in ('valid', 'test'):
                a = r[(r['strategy'] == 'random') & (r['dataset'] == ds)]
                b = r[(r['strategy'] == 'oot_y2013') & (r['dataset'] == ds)]
                if a.empty or b.empty:
                    continue
                row['random_%s_auc' % ds] = round(float(a['auc_roc'].iloc[0]), 4)
                row['oot_%s_auc' % ds] = round(float(b['auc_roc'].iloc[0]), 4)
                row['%s_auc_drop' % ds] = round(
                    float(a['auc_roc'].iloc[0]) - float(b['auc_roc'].iloc[0]), 4
                )
                row['random_%s_def' % ds] = round(float(a['def_rate'].iloc[0]), 4)
                row['oot_%s_def' % ds] = round(float(b['def_rate'].iloc[0]), 4)
            drift.append(row)
        drift_df = pd.DataFrame(drift)
        drift_df.to_csv(
            os.path.join(TAB_DIR, '05_drift_comparison.csv'), index=False, encoding='utf-8-sig'
        )
        print('\n' + '=' * 78)
        print('两策略指标落差（随机切分 AUC − 时间切分 AUC）')
        print('=' * 78)
        print(
            drift_df[
                [
                    'model',
                    'random_test_auc',
                    'oot_test_auc',
                    'test_auc_drop',
                    'random_test_def',
                    'oot_test_def',
                ]
            ].to_string(index=False)
        )
        fig_strategy_compare(all_metrics)

        # 特征贡献分析（报告第 5 章）：随机切分 + XGBoost_base 口径的特征组合消融
    if any(t == 'random' for t, _ in todo):
        run_ablation('random')

    print('\n产物：')
    for f in sorted(os.listdir(TAB_DIR)):
        if f.startswith('05_'):
            print('  tables/%s' % f)
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith('05_'):
            print('  figures/%s' % f)


def run_training(todo):
    _training_main(todo)


def main(argv=None):
    argv = list(sys.argv[1:]) if argv is None else list(sys.argv[1:])
    modes = [a for a in argv if a in ('random', 'oot_y2013', 'shap')]
    if not modes:
        _training_main(None)
        for tag, _ in STRATEGIES:
            run_shap(tag)
        return
    tags = [m for m in modes if m in ('random', 'oot_y2013')]
    if tags:
        _training_main(tags)
    if 'shap' in modes or not modes:
        for tag, _ in STRATEGIES:
            if not tags or tag in tags:
                run_shap(tag)


if __name__ == '__main__':
    main()
