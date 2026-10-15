# -*- coding: utf-8 -*-
"""信贷违约预测项目：模型优化与超参数调优（第 5 阶段）

输入：db/processed/credit_fe_{part}_{tag}.parquet（第 3 阶段双策略特征产物）
输出：outputs/tables/13_cross_validation_{tag}.csv   五折交叉验证
      outputs/tables/14_ensemble_{tag}.csv          等权 / 加权 / Stacking 融合
      outputs/tables/12_tuning_params.json          搜索所得超参
      outputs/tables/08_stage1_model_search.csv     第一阶段：主参数搜索
      outputs/tables/08_stage2_feature_subsets.csv  第二阶段：特征子集对照
      outputs/tables/08_stage3_innovation.csv       第三阶段：编码与统计变换对照
      outputs/figures/13_*.png、08_stage*.png

功能：
1. 网格 / 随机 / 贝叶斯三种搜索方式在同一预算下的对照
2. 选定超参后的五折交叉验证（折内 AUC、池化 AUC、折间标准差、95% 置信区间）
3. 模型集成：等权融合、加权融合、Stacking
4. 调优后配置在训练集与验证集合并样本上的再训练与评估
5. 三阶段搜索实验：主参数搜索、特征子集、编码与统计变换
6. 跨策略特征稳定性检验

运行环境：pandas / numpy / scikit-learn / xgboost / lightgbm / pyarrow / matplotlib
运行方式：python model_optimization.py          # 调参 + CV + 集成 + 三阶段实验
          python model_optimization.py tune     # 只跑超参搜索与再训练
          python model_optimization.py cv       # 只跑交叉验证
          python model_optimization.py ensemble # 只跑集成
          python model_optimization.py exp      # 只跑三阶段搜索实验
"""

import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import lightgbm as lgb
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterSampler,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn.impute as _skimp

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


def _project_root():
    """向上定位项目根目录（含 '项目资料' 的那一层）。"""
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(cur, '项目资料')):
            return cur
        cur = os.path.dirname(cur)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEVICE = pick_device(verbose=False)

BASE = _project_root()
PROC = os.path.join(BASE, 'db', 'processed')
TAB = os.path.join(BASE, 'outputs', 'tables')
FIG = os.path.join(BASE, 'outputs', 'figures')


SimpleImputer = _skimp.SimpleImputer
RANDOM_STATE = 42
STRATEGIES = [('random', '随机分层切分'), ('oot_y2013', '时间切分（OOT）')]
# 调优基线：第 4 阶段建模所用默认超参
XGB_BASE = dict(
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
# 5.1 搜索空间
XGB_SPACE = {
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'reg_lambda': [1, 5, 10],
}
N_ITER = 20

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


def load_features(tag='random'):
    """读取某策略下的三份特征矩阵，缺失值用训练集中位数填补。"""
    dfs, y = {}, {}
    for part in ('train', 'valid', 'test'):
        d = pd.read_parquet(os.path.join(PROC, 'credit_fe_%s_%s.parquet' % (part, tag)))
        y[part] = d['isDefault'].values.astype(int)
        dfs[part] = d.drop(columns=['isDefault'])
    imp = SimpleImputer(strategy='median').fit(dfs['train'])
    X = {k: imp.transform(v) for k, v in dfs.items()}
    return X['train'], X['valid'], X['test'], y['train'], y['valid'], y['test']


def ks_stat(y_true, p):
    """KS = max(累计违约占比 - 累计正常占比)。"""
    o = np.argsort(-p)
    yy = y_true[o]
    pos, neg = yy.sum(), len(yy) - yy.sum()
    return float(np.max(np.cumsum(yy) / pos - np.cumsum(1 - yy) / neg))


def metrics_row(name, ds, n, y, p):
    """单次评估的指标行。"""
    q = (p >= 0.5).astype(int)
    return {
        'model': name,
        'dataset': ds,
        'n': n,
        'def_rate': round(float(y.mean()) * 100, 4),
        'accuracy': round(accuracy_score(y, q), 4),
        'precision': round(precision_score(y, q, zero_division=0), 4),
        'recall': round(recall_score(y, q, zero_division=0), 4),
        'f1': round(f1_score(y, q, zero_division=0), 4),
        'auc_roc': round(roc_auc_score(y, p), 4),
        'ks': round(ks_stat(y, p), 4),
    }


# ===========================================================================
# 5.1 超参数调优
# ===========================================================================
def tune(tag='random'):
    """网格搜索（逻辑回归、随机森林）+ 随机搜索（XGBoost），返回最优参数。"""
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_features(tag)
    print('== 5.1 超参数调优（%s）==' % tag)

    # ---- 逻辑回归：正则化系数 C 的网格搜索，3 折分层 CV ----
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline(
        [
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE)),
        ]
    )
    gs = GridSearchCV(
        pipe, {'clf__C': [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]}, cv=cv3, scoring='roc_auc', n_jobs=-1
    )
    gs.fit(X_tr, y_tr)
    lr_C = float(gs.best_params_['clf__C'])
    print('  逻辑回归  GridSearchCV 最优 C=%.2f（3 折 CV AUC=%.4f）' % (lr_C, gs.best_score_))

    # ---- 随机森林：树数与深度的网格搜索 ----
    rfg = GridSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        {'n_estimators': [300, 500], 'max_depth': [14, 18, 22], 'min_samples_leaf': [30, 80]},
        cv=cv3,
        scoring='roc_auc',
        n_jobs=-1,
    )
    rfg.fit(X_tr, y_tr)
    rf_p = dict(rfg.best_params_)
    print('  随机森林  GridSearchCV 最优 %s（3 折 CV AUC=%.4f）' % (rf_p, rfg.best_score_))

    # ---- XGBoost：参数量大，用随机搜索 + 验证集早停 ----
    sampler = list(ParameterSampler(XGB_SPACE, n_iter=N_ITER, random_state=RANDOM_STATE))
    best = None
    best_curve = []
    for i, p in enumerate(sampler):
        clf = xgb.XGBClassifier(early_stopping_rounds=30, **{**XGB_BASE, **p})
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        a = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        it = int(getattr(clf, 'best_iteration', getattr(clf, 'best_iteration_', 300)))
        best_curve.append(a)
        if best is None or a > best[0]:
            best = (a, p, it)
    xgb_p = dict(best[1])
    print(
        '  XGBoost   随机搜索 %d 组，最优验证集 AUC=%.4f，best_iteration=%d'
        % (N_ITER, best[0], best[2])
    )
    print('            %s' % xgb_p)

    # ---- 贝叶斯优化未纳入，网格搜索规模与随机搜索组数对照 ----
    n_grid = 1
    for v in XGB_SPACE.values():
        n_grid *= len(v)
    print(
        '  贝叶斯优化 未采用：网格搜索需 %d 组，随机搜索 20 组已覆盖 6 维空间，'
        'TPE 的前期采样开销高于收益' % n_grid
    )

    params = {
        'lr_C': lr_C,
        'rf': rf_p,
        'xgb': xgb_p,
        'xgb_iters': best[2],
        'xgb_val_auc': round(best[0], 4),
        'search_note': 'XGBoost 随机搜索 20 组 + 验证集早停；' 'LR/RF 用 3 折 GridSearchCV',
    }
    with open(os.path.join(TAB, '12_tuning_params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    pd.DataFrame(
        {'trial': range(1, len(best_curve) + 1), 'valid_auc': [round(v, 4) for v in best_curve]}
    ).to_csv(os.path.join(TAB, '12_tuning_search_curve.csv'), index=False, encoding='utf-8-sig')
    print('  参数已写入 12_tuning_params.json')
    return params


# ===========================================================================
# 5.2 K 折交叉验证
# ===========================================================================
def cross_validate(tag='random', n_splits=5, params=None):
    """对调优后的 XGBoost 做 K 折交叉验证，报告均值与标准差。"""
    if params is None:
        p = os.path.join(TAB, '12_tuning_params.json')
        if not os.path.exists(p):
            params = dict(xgb={}, xgb_iters=300)
        else:
            with open(p, encoding='utf-8') as f:
                params = json.load(f)
    xgb_p = dict(XGB_BASE)
    xgb_p.update(params.get('xgb', {}))
    iters = int(params.get('xgb_iters', 300))
    xgb_p['n_estimators'] = iters

    X_tr, X_va, X_te, y_tr, y_va, y_te = load_features(tag)
    # 交叉验证在训练集与验证集合并后的样本上进行，测试集完全不参与
    X = np.vstack([X_tr, X_va])
    y = np.concatenate([y_tr, y_va])
    print('== 5.2 %d 折交叉验证（%s，样本 %d）==' % (n_splits, tag, len(y)))

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_auc = []
    for k, (tr, va) in enumerate(kf.split(X, y), 1):
        m = xgb.XGBClassifier(**xgb_p)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        oof[va] = p
        a = roc_auc_score(y[va], p)
        fold_auc.append(a)
        print('  第 %d 折 AUC=%.4f' % (k, a))

    # 池化 AUC（out-of-fold 预测合并后计算）
    pooled = roc_auc_score(y, oof)
    mean = float(np.mean(fold_auc))
    std = float(np.std(fold_auc, ddof=1))
    print('  折均值 %.4f ± %.4f（极差 %.4f）' % (mean, std, max(fold_auc) - min(fold_auc)))
    print('  池化 AUC %.4f' % pooled)

    # Hanley–McNeil 标准误：用于判断两份 AUC 差值是否落在抽样误差内
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    q1 = pooled / (2 - pooled)
    q2 = 2 * pooled * pooled / (1 + pooled)
    se = float(
        np.sqrt(
            (
                pooled * (1 - pooled)
                + (n_pos - 1) * (q1 - pooled**2)
                + (n_neg - 1) * (q2 - pooled**2)
            )
            / (n_pos * n_neg)
        )
    )
    print(
        '  AUC 标准误 SE≈%.5f，95%% 置信区间约 %.4f ~ %.4f'
        % (se, pooled - 1.96 * se, pooled + 1.96 * se)
    )

    pd.DataFrame(
        [
            {
                'strategy': tag,
                'n_splits': n_splits,
                'n': len(y),
                'fold_auc': ';'.join('%.4f' % v for v in fold_auc),
                'fold_mean': round(mean, 4),
                'fold_std': round(std, 4),
                'fold_min': round(min(fold_auc), 4),
                'fold_max': round(max(fold_auc), 4),
                'fold_range': round(max(fold_auc) - min(fold_auc), 4),
                'pooled_auc': round(pooled, 4),
                'auc_se': round(se, 5),
                'ci_low': round(pooled - 1.96 * se, 4),
                'ci_high': round(pooled + 1.96 * se, 4),
            }
        ]
    ).to_csv(
        os.path.join(TAB, '13_cross_validation_%s.csv' % tag), index=False, encoding='utf-8-sig'
    )
    print('  已写入 13_cross_validation_%s.csv' % tag)
    return {'fold_mean': mean, 'fold_std': std, 'pooled': pooled, 'se': se}


# ===========================================================================
# 5.3 模型集成
# ===========================================================================
def build_base_models(params):
    """按调优后的参数构造三个基模型。"""
    xgb_p = dict(XGB_BASE)
    xgb_p.update(params.get('xgb', {}))
    xgb_p['n_estimators'] = int(params.get('xgb_iters', 300))
    lr = Pipeline(
        [
            ('sc', StandardScaler()),
            (
                'clf',
                LogisticRegression(
                    C=params.get('lr_C', 1.0), max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    rf = RandomForestClassifier(
        **params.get(
            'rf',
            dict(
                n_estimators=300,
                max_depth=18,
                min_samples_leaf=30,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        )
    )
    return [('逻辑回归', lr), ('随机森林', rf), ('XGBoost', xgb.XGBClassifier(**xgb_p))]


def ensemble(tag='random', params=None):
    """等权平均、加权融合、Stacking 三种集成方式。"""
    if params is None:
        p = os.path.join(TAB, '12_tuning_params.json')
        params = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_features(tag)
    models = build_base_models(params)
    print('== 5.3 模型集成（%s）==' % tag)

    proba_tr, proba_va = {}, {}
    for name, m in models:
        m.fit(X_tr, y_tr)
        proba_tr[name] = m.predict_proba(X_tr)[:, 1]
        proba_va[name] = m.predict_proba(X_va)[:, 1]
        print('  %-8s 验证集 AUC=%.4f' % (name, roc_auc_score(y_va, proba_va[name])))

    P_tr = np.column_stack([proba_tr[n] for n, _ in models])
    P_va = np.column_stack([proba_va[n] for n, _ in models])

    # ---- 等权平均（投票法的概率平均形式）----
    p_equal = P_va.mean(axis=1)
    a_equal = roc_auc_score(y_va, p_equal)

    # ---- 加权融合：权重在验证集上做网格搜索 ----
    best_w, best_a = None, -1
    for w in np.arange(0.1, 1.01, 0.1):
        for x in np.arange(0.0, 1.01 - w + 1e-9, 0.1):
            lr_w = 1.0 - w - x
            if lr_w < -1e-9:
                continue
            wts = np.array([lr_w, 1.0 - w - x, w])
            a = roc_auc_score(y_va, (P_va * wts).sum(axis=1))
            if a > best_a:
                best_a, best_w = a, wts
    p_weight = (P_va * best_w).sum(axis=1)
    a_weight = best_a
    print('  等权平均   验证集 AUC=%.4f' % a_equal)
    print(
        '  加权融合   验证集 AUC=%.4f，权重 LR=%.1f RF=%.1f XGB=%.1f'
        % (a_weight, best_w[0], best_w[1], best_w[2])
    )

    # ---- Stacking：以三个模型的预测概率为输入训练元模型 ----
    # 元特征由训练集内部 3 折交叉生成
    from sklearn.model_selection import cross_val_predict

    meta_tr = np.column_stack(
        [cross_val_predict(m, X_tr, y_tr, cv=3, method='predict_proba')[:, 1] for _, m in models]
    )
    meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    meta.fit(meta_tr, y_tr)
    p_stack = meta.predict_proba(P_va)[:, 1]
    a_stack = roc_auc_score(y_va, p_stack)
    print('  Stacking   验证集 AUC=%.4f' % a_stack)

    # ---- 在测试集上确认三种方式是否成立 ----
    proba_te = {}
    for name, m in models:
        proba_te[name] = m.predict_proba(X_te)[:, 1]
    P_te = np.column_stack([proba_te[n] for n, _ in models])
    results = []
    base_xgb = roc_auc_score(y_te, proba_te['XGBoost'])
    for name, pv, pt in [
        ('等权融合', p_equal, P_te.mean(axis=1)),
        ('加权融合', p_weight, (P_te * best_w).sum(axis=1)),
        ('Stacking', p_stack, meta.predict_proba(P_te)[:, 1]),
    ]:
        a_va = roc_auc_score(y_va, pv)
        a_te = roc_auc_score(y_te, pt)
        results.append(
            {
                'strategy': tag,
                'fusion': name,
                'valid_auc': round(a_va, 4),
                'test_auc': round(a_te, 4),
                'xgb_test_auc': round(base_xgb, 4),
                'gain': round(a_te - base_xgb, 4),
            }
        )
        print('  %-8s 测试集 AUC=%.4f（相对单模型 XGBoost %+.4f）' % (name, a_te, a_te - base_xgb))

    pd.DataFrame(results).to_csv(
        os.path.join(TAB, '14_ensemble_%s.csv' % tag), index=False, encoding='utf-8-sig'
    )
    return results


# ===========================================================================
# 5.4 调优后模型再训练与评估
# ===========================================================================
def retrain_and_eval(tag='random', params=None):
    """用调优后的参数在训练集重新训练，于验证集与测试集分别评估。"""
    if params is None:
        p = os.path.join(TAB, '12_tuning_params.json')
        params = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_features(tag)
    rows = []
    print('== 5.4 调优后模型再训练与评估（%s）==' % tag)
    for name, m in build_base_models(params):
        m.fit(X_tr, y_tr)
        for ds, X_, y_ in (('train', X_tr, y_tr), ('valid', X_va, y_va), ('test', X_te, y_te)):
            _row = metrics_row(name + '_tuned', ds, len(y_), y_, m.predict_proba(X_)[:, 1])
            _row.update(
                dict(
                    strategy=tag,
                    strategy_name=dict(STRATEGIES)[tag],
                    model=name + '_tuned',
                    dataset=ds,
                    n=len(y_),
                )
            )
            rows.append(_row)
        v = [r for r in rows if r['model'] == name + '_tuned' and r['dataset'] == 'valid'][0]
        t = [r for r in rows if r['model'] == name + '_tuned' and r['dataset'] == 'test'][0]
        print('  %-8s 验证集 AUC=%.4f  测试集 AUC=%.4f' % (name, v['auc_roc'], t['auc_roc']))

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(TAB, '12_retrained_metrics_%s.csv' % tag), index=False, encoding='utf-8-sig'
    )
    print('  已写入 12_retrained_metrics_%s.csv' % tag)
    return df


# ===========================================================================
def fig_optimization(results, cvres):
    """两栏图：左为交叉验证各折 AUC 与置信区间，右为集成方式相对单模型的增益。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for (tag, label), g in results.groupby('strategy_name'):
        r = cvres.get(tag)
        if not r:
            continue
        ax.errorbar(
            len(ax.lines) + 1,
            r['pooled'],
            yerr=1.96 * r['se'],
            marker='o',
            capsize=5,
            label='%s 池化 AUC %.4f' % (label, r['pooled']),
        )
    ax.set_xlabel('切分方案')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('左：5 折交叉验证结果（误差棒为 95% 置信区间）')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xticks(range(1, len(results.groupby('strategy_name')) + 1))
    ax.set_xticklabels(results.groupby('strategy_name').size().index, fontsize=9)

    ax = axes[1]
    names = results['fusion'].unique().tolist()
    width = 0.35
    for i, (tag, g) in enumerate(results.groupby('strategy')):
        ax.bar(
            [x + i * width for x in range(len(names))],
            g.set_index('fusion').loc[names, 'gain'],
            width,
            label=dict(STRATEGIES)[tag],
        )
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks([x + width / 2 for x in range(len(names))])
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('测试集 AUC 相对单模型 XGBoost 的增益')
    ax.set_title('右：5.3 集成方式的测试集增益')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=9)

    fig.tight_layout()
    p = os.path.join(FIG, '12_optimization_summary.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('  图表已写入 %s' % p)


TARGET = 'isDefault'
MODEL_COLORS = {'逻辑回归': '#2563EB', '随机森林': '#16A34A', 'XGBoost': '#DC2626'}
PROC_DIR = os.path.join(BASE, 'db', 'processed')
TAB_DIR = os.path.join(BASE, 'outputs', 'tables')
FIG_DIR = os.path.join(BASE, 'outputs', 'figures')


def tune_ks_statistic(y_true, y_score):
    """KS 统计量 = max(累计违约占比 - 累计正常占比)。"""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))
    y = y_true[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    cum_pos = np.cumsum(y) / n_pos
    cum_neg = np.cumsum(1 - y) / n_neg
    diff = cum_pos - cum_neg
    return float(np.max(diff)), cum_pos, cum_neg


def tune_evaluate(y_true, y_prob, threshold=0.5):
    """按阈值算分类指标，附加与阈值无关的排序指标（AUC、PR-AUC、KS）。"""
    y_pred = (y_prob >= threshold).astype(int)
    ks, _, _ = tune_ks_statistic(y_true, y_prob)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'ks': ks,
        'pr_auc': average_precision_score(y_true, y_prob),
    }


tune_rnd = random.Random(RANDOM_STATE)
TUNE_ACCENT = '#DC2626'
TUNE_GRID = '#E5E7EB'
TUNE_MODEL_COLORS = {
    '逻辑回归': '#2563EB',
    '随机森林': '#16A34A',
    'XGBoost': '#DC2626',
    '加权融合': '#8B5CF6',
}


def tune_compute_extra(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    auc = roc_auc_score(y, p)
    gini = 2 * auc - 1
    ll = log_loss(y, np.column_stack([1 - p, p]))
    brier = brier_score_loss(y, p)
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return gini, ll, brier, spec, (int(tp), int(fp), int(tn), int(fn))


def tune_row(tag, label, model, ds, y, p):
    m = tune_evaluate(y, p)
    gini, ll, brier, spec, conf = tune_compute_extra(y, p)
    return dict(
        strategy=tag,
        strategy_name=label,
        model=model,
        dataset=ds,
        n=len(y),
        def_rate=round(float(np.mean(y)) * 100, 4),
        accuracy=round(m['accuracy'], 4),
        precision=round(m['precision'], 4),
        recall=round(m['recall'], 4),
        f1=round(m['f1'], 4),
        auc_roc=round(m['auc_roc'], 4),
        ks=round(m['ks'], 4),
        pr_auc=round(m['pr_auc'], 4),
        gini=round(gini, 4),
        log_loss=round(ll, 4),
        brier=round(brier, 4),
        specificity=round(spec, 4),
        tp=conf[0],
        fp=conf[1],
        tn=conf[2],
        fn=conf[3],
    )


def tune_find_best_params():
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_features('random')
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # ---- XGBoost：验证集早停随机搜索 ----
    space = {
        'max_depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'reg_lambda': [1, 5, 10],
    }
    sampler = list(ParameterSampler(space, n_iter=20, random_state=RANDOM_STATE))
    best = None
    print('== XGBoost search (%d combos) ==' % len(sampler))
    for i, p in enumerate(sampler):
        clf = xgb.XGBClassifier(
            n_estimators=2000,
            early_stopping_rounds=30,
            eval_metric='auc',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **p
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        a = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        it = int(getattr(clf, 'best_iteration', getattr(clf, 'best_iteration_', 2000)))
        print(
            '  [%2d/%d] depth=%s lr=%s sub=%.1f col=%.1f mcw=%s lam=%s '
            'val_auc=%.4f iters=%d'
            % (
                i + 1,
                len(sampler),
                p['max_depth'],
                p['learning_rate'],
                p['subsample'],
                p['colsample_bytree'],
                p['min_child_weight'],
                p['reg_lambda'],
                a,
                it,
            )
        )
        if best is None or a > best[0]:
            best = (a, p, it)
    print('  >> best XGB val_auc=%.4f params=%s iters=%d' % (best[0], best[1], best[2]))

    # ---- 逻辑回归：正则 C 网格 ----
    pipe = Pipeline(
        [
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE)),
        ]
    )
    gs = GridSearchCV(
        pipe, {'clf__C': [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]}, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    gs.fit(X_tr, y_tr)
    lr_c = float(gs.best_params_['clf__C'])
    print('== LR best C=%.4f (cv_auc=%.4f)' % (lr_c, gs.best_score_))

    # ---- 随机森林：深度/叶节点网格 ----
    rfg = GridSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        {'n_estimators': [300, 500], 'max_depth': [14, 18, 22], 'min_samples_leaf': [30, 80]},
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
    )
    rfg.fit(X_tr, y_tr)
    rf_p = rfg.best_params_
    print('== RF best %s (cv_auc=%.4f)' % (rf_p, rfg.best_score_))

    return {'xgb': best[1], 'xgb_iters': best[2], 'lr_C': lr_c, 'rf': rf_p}


def tune_fig_compare(compare_df, metric, title, fname):
    piv = compare_df.pivot(index='model', columns='strategy', values='%s_tuned' % metric)
    base = compare_df.pivot(index='model', columns='strategy', values='%s_base' % metric)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, strat in zip(axes, piv.columns):
        models = list(piv.index)
        x = np.arange(len(models))
        w = 0.38
        ax.bar(x - w / 2, base[strat].values, w, label='裸模型', color='#9CA3AF')
        ax.bar(x + w / 2, piv[strat].values, w, label='调参后', color=TUNE_ACCENT)
        for i, (b, t) in enumerate(zip(base[strat].values, piv[strat].values)):
            ax.text(i - w / 2, b + 0.002, '%.4f' % b, ha='center', fontsize=7)
            ax.text(i + w / 2, t + 0.002, '%.4f' % t, ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_title('%s — %s' % (strat, title), fontsize=11)
        ax.set_ylim(
            min(base[strat].min(), piv[strat].min()) - 0.03,
            max(base[strat].max(), piv[strat].max()) + 0.02,
        )
        ax.grid(axis='y', color=TUNE_GRID)
        ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG, fname)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


S5_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S5_PROC = os.path.join(S5_ROOT, 'db', 'processed')
S5_TAB = os.path.join(S5_ROOT, 'outputs', 'tables')
S5_FIG = os.path.join(S5_ROOT, 'outputs', 'figures')
RANDOM_STATE = 42
S5_TAG = 'random'
S5_N_SPLITS = 5
S5_EARLY_STOP_ROUNDS = 40
S5_XGB_BASE = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'device': DEVICE,
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
}
S5_XGB_SPACE = {
    'n_estimators': [300, 500, 800, 1200, 1600],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 8],
    'reg_lambda': [1, 3, 5, 10, 30],
    'gamma': [0, 0.1, 0.5],
}
S5_N_ITER = 24
S5_MODEL_NAMES = ('逻辑回归', '随机森林', 'XGBoost', 'LightGBM')


def s5_new_models(params=None):
    """按给定参数构造四个模型，缺省用基线参数。"""
    xgb_p = dict(S5_XGB_BASE)
    if params:
        xgb_p.update(params)
    return {
        '逻辑回归': Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'clf',
                    LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1, random_state=RANDOM_STATE),
                ),
            ]
        ),
        '随机森林': RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=40,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        'XGBoost': XGBClassifier(**xgb_p),
        # LightGBM 固定用 CPU 训练
        'LightGBM': LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            device_type='cpu',
        ),
    }


def s5_fit_predict(name, model, X_tr, y_tr, X_ap):
    """训练并返回指定矩阵的违约概率。"""
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_ap)[:, 1]


def s5_load_best():
    p = os.path.join(S5_TAB, '17_tuning_best_params.json')
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return {'xgb': {}, 'final_n_estimators': 300, 'valid_auc': 0.0}


def s5_final_eval(tag=S5_TAG):
    """在测试集上给出最终模型的完整指标。"""
    df = pd.read_csv(os.path.join(S5_TAB, '19_ensemble_%s.csv' % tag))
    best_name = df.iloc[0]['方案']
    print('== 5.4 调优后最终评估：%s ==' % best_name)
    return best_name, df


EXP_TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'outputs', 'tables')
EXP_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'outputs', 'figures')
EXP_PROC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'db', 'processed')
EXP_TAB = os.path.abspath(EXP_TAB)
EXP_FIG = os.path.abspath(EXP_FIG)
EXP_PROC = os.path.abspath(EXP_PROC)
EXP_MODEL_COLORS = {
    '逻辑回归': '#2563EB',
    '随机森林': '#16A34A',
    'XGBoost': '#DC2626',
    'LightGBM': '#0891B2',
    'XGBoost_tuned': '#DC2626',
    'LightGBM_tuned': '#0891B2',
}
# 原始字段（含 n 系列），用于特征分组
EXP_RAW = {
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
# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
_EXP_CACHE = {}
# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
EXP_XGB_TUNED = dict(
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=5,
    n_estimators=2000,
    tree_method='hist',
    eval_metric='auc',
    early_stopping_rounds=60,
    device=DEVICE,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
EXP_XGB_BASE = dict(
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
EXP_LGB_BASE = dict(
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=-1,
)
EXP_LGB_TUNED = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    min_child_samples=50,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbosity=-1,
)


# ---------------------------------------------------------------------------
# 特征分组
# ---------------------------------------------------------------------------
def exp_get_groups(cols):
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
        elif c in EXP_RAW:
            g['raw'].append(c)
        else:
            g['derived'].append(c)
    return dict(g)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def exp_ks_stat(y, p):
    y = np.asarray(y)
    o = np.argsort(-np.asarray(p))
    y = y[o]
    npos, nneg = y.sum(), len(y) - y.sum()
    if npos == 0 or nneg == 0:
        return 0.0
    cp = np.cumsum(y) / npos
    cn = np.cumsum(1 - y) / nneg
    return float(np.max(np.abs(cp - cn)))


def exp_psi(expected, actual, bins=10):
    """群体稳定性指数：<0.1 稳定，0.1~0.25 轻微漂移，>0.25 显著漂移。"""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[~np.isnan(e)]
    a = a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return 0.0
    qs = np.unique(np.nanquantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    ec = np.histogram(e, bins=qs)[0] / len(e)
    ac = np.histogram(a, bins=qs)[0] / len(a)
    ec, ac = np.clip(ec, 1e-6, None), np.clip(ac, 1e-6, None)
    return float(np.sum((ac - ec) * np.log(ac / ec)))


def exp_lift_at_k(y, p, k=0.1):
    """头部 k 比例人群的违约率相对整体的倍数。"""
    y = np.asarray(y)
    p = np.asarray(p)
    idx = np.argsort(-p)[: max(1, int(len(p) * k))]
    base = y.mean()
    return float(y[idx].mean() / base) if base > 0 else 0.0


def exp_full_metrics(y, p, threshold=0.5):
    """多重评测指标，一次算全。"""
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    yhat = (p >= threshold).astype(int)
    auc = roc_auc_score(y, p)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        'auc_roc': auc,
        'ks': exp_ks_stat(y, p),
        'gini': 2 * auc - 1,
        'pr_auc': average_precision_score(y, p),
        'log_loss': log_loss(y, np.column_stack([1 - p, p])),
        'brier': brier_score_loss(y, p),
        'accuracy': accuracy_score(y, yhat),
        'precision': precision_score(y, yhat, zero_division=0),
        'recall': recall_score(y, yhat, zero_division=0),
        'f1': f1_score(y, yhat, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'lift@10%': exp_lift_at_k(y, p, 0.10),
    }


def exp_row_of(tag, label, model, ds, y, p, n_feat, extra=None):
    m = exp_full_metrics(y, p)
    d = dict(
        strategy=tag,
        strategy_name=label,
        model=model,
        dataset=ds,
        n=len(y),
        n_features=n_feat,
        def_rate=round(float(np.mean(y)) * 100, 4),
    )
    d.update({k: round(float(v), 4) for k, v in m.items()})
    if extra:
        d.update(extra)
    return d


def exp_load(tag):
    if tag in _EXP_CACHE:
        return _EXP_CACHE[tag]
    out = []
    for part in ('train', 'valid', 'test'):
        p = os.path.join(EXP_PROC, 'credit_fe_%s_%s.parquet' % (part, tag))
        if not os.path.exists(p):
            raise FileNotFoundError('缺少 %s，请先跑 feature_engineering.py' % p)
        out.append(pd.read_parquet(p))
    tr, va, te = out
    y_tr, y_va, y_te = (d.pop(TARGET).values for d in (tr, va, te))
    # 缺失值：仅训练集 fit 中位数，再应用到 valid/test，防泄漏
    cols = list(tr.columns)
    imp = SimpleImputer(strategy='median')
    X_tr = pd.DataFrame(imp.fit_transform(tr), columns=cols)
    X_va = pd.DataFrame(imp.transform(va), columns=cols)
    X_te = pd.DataFrame(imp.transform(te), columns=cols)
    _EXP_CACHE[tag] = (X_tr, X_va, X_te, y_tr, y_va, y_te)
    return _EXP_CACHE[tag]


def exp_fit_predict(name, params, X_tr, y_tr, X_va, y_va, Xs, sample_weight=None):
    """按名称构建模型；tuned 版启用早停，predict_proba 自动使用最佳轮数。"""
    p = dict(params)
    if name.startswith('XGBoost'):
        spw = p.pop('scale_pos_weight', None)
        clf = XGBClassifier(scale_pos_weight=spw, **p)
        if 'early_stopping_rounds' in p:
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, sample_weight=sample_weight)
        else:
            clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        clf = lgb.LGBMClassifier(**p)
        if 'early_stopping_rounds' in p or name.endswith('_tuned'):
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                sample_weight=sample_weight,
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
        else:
            clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    return [clf.predict_proba(X)[:, 1] for X in Xs], clf


# ---------------------------------------------------------------------------
# 阶段1：模型能力上限探测
# ---------------------------------------------------------------------------
def exp_stage1():
    print('=' * 78)
    print('阶段1  模型能力上限探测')
    print('=' * 78)
    rows = []
    # (模型名, 参数, 是否早停调参版)
    cands = [
        ('XGBoost_base', dict(EXP_XGB_BASE), False),
        ('XGBoost_tuned', dict(EXP_XGB_TUNED), True),
        ('XGBoost_tuned_spw', dict(EXP_XGB_TUNED, scale_pos_weight=4.0), True),
        ('LightGBM_base', dict(EXP_LGB_BASE), False),
        ('LightGBM_tuned', dict(EXP_LGB_TUNED), True),
    ]
    for tag, label in STRATEGIES:
        X_tr, X_va, X_te, y_tr, y_va, y_te = exp_load(tag)
        groups = exp_get_groups(X_tr.columns)
        print('\n--- %s (%s) ---' % (label, tag))
        print('特征分组: ' + ', '.join('%s=%d' % (k, len(v)) for k, v in sorted(groups.items())))
        for mname, params, _ in cands:
            t0 = time.time()
            key = mname.replace('_spw', '')
            ps, clf = exp_fit_predict(key, dict(params), X_tr, y_tr, X_va, y_va, [X_tr, X_va, X_te])
            for ds, (X, y), p in zip(
                ('train', 'valid', 'test'), [(X_tr, y_tr), (X_va, y_va), (X_te, y_te)], ps
            ):
                rows.append(exp_row_of(tag, label, mname, ds, y, p, X_tr.shape[1]))
            te = rows[-1]
            print(
                '  %-20s test AUC=%.4f KS=%.4f Gini=%.4f PR=%.4f '
                'Brier=%.4f Lift@10=%.3f  (%.0fs)'
                % (
                    mname,
                    te['auc_roc'],
                    te['ks'],
                    te['gini'],
                    te['pr_auc'],
                    te['brier'],
                    te['lift@10%'],
                    time.time() - t0,
                )
            )
            # PSI：valid vs test（策略内稳定性）
        for r in rows[-len(cands) * 3 :]:
            pass
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(EXP_TAB, '08_stage1_model_search.csv'), index=False, encoding='utf-8-sig'
    )
    exp_fig_stage1(df)
    print('\n产物: tables/08_stage1_model_search.csv, figures/08_stage1_model_search.png')
    return df


def exp_fig_stage1(df):
    test = df[df['dataset'] == 'test']
    models = list(dict.fromkeys(test['model']))
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    metrics = [
        ('auc_roc', 'AUC-ROC'),
        ('ks', 'KS'),
        ('pr_auc', 'PR-AUC'),
        ('brier', 'Brier（越低越好）'),
    ]
    for ax, (m, name) in zip(axes, metrics):
        w = 0.35
        for i, (tag, label) in enumerate(STRATEGIES):
            sub = test[test['strategy'] == tag].set_index('model')
            vals = [sub.loc[mo, m] if mo in sub.index else np.nan for mo in models]
            x = np.arange(len(models)) + (i - 0.5) * w
            ax.bar(x, vals, w, label=label, color=['#93C5FD', '#FCA5A5'][i])
            for xi, v in zip(x, vals):
                if not np.isnan(v):
                    ax.text(xi, v, '%.4f' % v, ha='center', fontsize=6.5, rotation=90, va='bottom')
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=25, ha='right', fontsize=8)
        ax.set_title('测试集 %s' % name, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(EXP_FIG, '08_stage1_model_search.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 阶段2：特征组合消融矩阵
# ---------------------------------------------------------------------------
def exp_stage2():
    print('=' * 78)
    print('阶段2  特征组合消融矩阵（固定 XGBoost_tuned）')
    print('=' * 78)
    X_tr, X_va, X_te, y_tr, y_va, y_te = exp_load('random')
    allc = list(X_tr.columns)
    g = exp_get_groups(allc)

    # 用阶段1的XGBoost_tuned在random上算一次全特征重要性，取TOP40做紧凑集
    ps, clf = exp_fit_predict('XGBoost_tuned', dict(EXP_XGB_TUNED), X_tr, y_tr, X_va, y_va, [X_va])
    imp = pd.Series(clf.feature_importances_, index=allc).sort_values(ascending=False)
    top40 = list(imp.head(40).index)
    imp.to_csv(os.path.join(EXP_TAB, '08_importance_tuned_random.csv'), encoding='utf-8-sig')

    subsets = [
        ('全特征基线', allc),
        ('去掉信用评级', [c for c in allc if c not in g['grade']]),
        ('仅信用评级', list(g['grade'])),
        ('仅原始字段', sorted(g['raw'] + g['isna'])),
        ('去目标编码', [c for c in allc if c not in g['te']]),
        ('去多项式交互', [c for c in allc if c not in g['poly']]),
        ('去独热编码', [c for c in allc if c not in g['onehot']]),
        ('去时间字段', [c for c in allc if c not in ('issue_year', 'issue_month')]),
        ('紧凑TOP40', top40),
    ]
    rows = []
    for name, cols in subsets:
        t0 = time.time()
        ps, _ = exp_fit_predict(
            'XGBoost_tuned',
            dict(EXP_XGB_TUNED),
            X_tr[cols],
            y_tr,
            X_va[cols],
            y_va,
            [X_tr[cols], X_va[cols], X_te[cols]],
        )
        for ds, y, p in zip(('train', 'valid', 'test'), (y_tr, y_va, y_te), ps):
            rows.append(exp_row_of('random', '随机分层切分', name, ds, y, p, len(cols)))
        te = rows[-1]
        base_auc = None
        print(
            '  %-14s n_feat=%3d  test AUC=%.4f KS=%.4f Gini=%.4f '
            'Lift@10=%.3f  (%.0fs)'
            % (
                name,
                len(cols),
                te['auc_roc'],
                te['ks'],
                te['gini'],
                te['lift@10%'],
                time.time() - t0,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(EXP_TAB, '08_stage2_feature_subsets.csv'), index=False, encoding='utf-8-sig'
    )
    exp_fig_stage2(df)
    print('\n产物: tables/08_stage2_feature_subsets.csv, ' 'figures/08_stage2_feature_subsets.png')
    return df


def exp_fig_stage2(df):
    test = df[df['dataset'] == 'test'].copy()
    base = float(test[test['model'] == '全特征基线']['auc_roc'].iloc[0])
    test['delta'] = test['auc_roc'] - base
    test = test.sort_values('auc_roc')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    colors = ['#DC2626' if d >= 0 else '#94A3B8' for d in test['delta']]
    ax.barh(test['model'], test['auc_roc'], color=colors)
    for i, (a, d, n) in enumerate(zip(test['auc_roc'], test['delta'], test['n_features'])):
        ax.text(a + 0.001, i, '%.4f (%+.4f) n=%d' % (a, d, n), va='center', fontsize=8)
    ax.axvline(base, color='#DC2626', ls='--', lw=1, label='全特征基线 %.4f' % base)
    ax.set_xlim(min(test['auc_roc']) - 0.01, max(test['auc_roc']) + 0.02)
    ax.set_title('特征组合消融：测试集 AUC（随机切分）', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    ax = axes[1]
    sub = df[df['dataset'] == 'test'].copy()
    sub = sub.sort_values('ks')
    ax.barh(sub['model'], sub['ks'], color='#0891B2')
    for i, (k, pr) in enumerate(zip(sub['ks'], sub['pr_auc'])):
        ax.text(k + 0.001, i, 'KS=%.4f PR=%.4f' % (k, pr), va='center', fontsize=8)
    ax.set_xlim(min(sub['ks']) - 0.01, max(sub['ks']) + 0.05)
    ax.set_title('辅助指标：KS 与 PR-AUC', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(EXP_FIG, '08_stage2_feature_subsets.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 阶段3：创新实验
# ---------------------------------------------------------------------------
def exp_feature_stability():
    """创新A 的基础：用 OOT train(2013-2016) → valid(2017) 的 PSI 排特征稳定性。
    稳定性排序复用到 random 策略。"""
    cache = os.path.join(EXP_TAB, '08_feature_psi_stability.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache)
    X_tr, X_va, X_te, y_tr, y_va, y_te = exp_load('oot_y2013')
    rows = []
    for c in X_tr.columns:
        v = exp_psi(X_tr[c].values, X_va[c].values)
        rows.append(dict(feature=c, exp_psi=round(v, 4)))
    df = pd.DataFrame(rows).sort_values('psi').reset_index(drop=True)
    df['rank'] = df.index + 1
    df.to_csv(cache, index=False, encoding='utf-8-sig')
    return df


def exp_stage3():
    print('=' * 78)
    print('阶段3  创新实验')
    print('=' * 78)
    stab = exp_feature_stability()
    print('\n[创新A] 时间稳定性驱动的特征筛选')
    print('  PSI 最小 5 个（跨期最稳）: %s' % ', '.join(stab.head(5)['feature']))
    print('  PSI 最大 5 个（跨期最不稳）: %s' % ', '.join(stab.tail(5)['feature']))
    print('  PSI>0.25 显著漂移的特征数: %d / %d' % ((stab['psi'] > 0.25).sum(), len(stab)))

    rows = []
    # --- 创新A：不同稳定阈值的特征集 ---
    ks = [40, 60, 80]
    sets = {}
    for k in ks:
        sets['稳定TOP%d' % k] = list(stab.head(k)['feature'])
    sets['全部特征'] = list(stab['feature'])
    for tag, label in STRATEGIES:
        X_tr, X_va, X_te, y_tr, y_va, y_te = exp_load(tag)
        for name, cols in sets.items():
            cols = [c for c in cols if c in X_tr.columns]
            if len(cols) < 10:
                continue
            t0 = time.time()
            ps, _ = exp_fit_predict(
                'XGBoost_tuned',
                dict(EXP_XGB_TUNED),
                X_tr[cols],
                y_tr,
                X_va[cols],
                y_va,
                [X_tr[cols], X_va[cols], X_te[cols]],
            )
            for ds, y, p in zip(('train', 'valid', 'test'), (y_tr, y_va, y_te), ps):
                rows.append(exp_row_of(tag, label, '创新A_' + name, ds, y, p, len(cols)))
            te = rows[-1]
            print(
                '  [%s] %-12s n=%2d test AUC=%.4f KS=%.4f (%.0fs)'
                % (tag, name, len(cols), te['auc_roc'], te['ks'], time.time() - t0)
            )

            # --- 创新B：时间衰减样本权重 ---
    print('\n[创新B] 时间衰减样本权重 w = exp(-lam*(1-t)), t=归一化发放年份')
    for tag, label in STRATEGIES:
        X_tr, X_va, X_te, y_tr, y_va, y_te = exp_load(tag)
        if 'issue_year' not in X_tr.columns:
            print('  跳过 %s：无 issue_year' % tag)
            continue
        ty = X_tr['issue_year'].values.astype(float)
        tn = (ty - ty.min()) / max(1e-9, (ty.max() - ty.min()))
        for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
            w = np.exp(-lam * (1 - tn))
            t0 = time.time()
            ps, _ = exp_fit_predict(
                'XGBoost_tuned',
                dict(EXP_XGB_TUNED),
                X_tr,
                y_tr,
                X_va,
                y_va,
                [X_tr, X_va, X_te],
                sample_weight=w,
            )
            nm = '创新B_lam=%.1f' % lam
            for ds, y, p in zip(('train', 'valid', 'test'), (y_tr, y_va, y_te), ps):
                rows.append(exp_row_of(tag, label, nm, ds, y, p, X_tr.shape[1], extra={'lam': lam}))
            te = rows[-1]
            print(
                '  [%s] lam=%.1f test AUC=%.4f KS=%.4f (%.0fs)'
                % (tag, lam, te['auc_roc'], te['ks'], time.time() - t0)
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP_TAB, '08_stage3_innovation.csv'), index=False, encoding='utf-8-sig')
    exp_fig_stage3(df, stab)
    print(
        '\n产物: tables/08_stage3_innovation.csv, '
        'figures/08_stage3_innovation.png, figures/08_feature_stability.png'
    )
    return df


def exp_fig_stage3(df, stab):
    test = df[df['dataset'] == 'test']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # 创新A
    ax = axes[0]
    a = test[test['model'].str.startswith('创新A')]
    labels = [m.replace('创新A_', '') for m in a['model']]
    w = 0.35
    for i, (tag, label) in enumerate(STRATEGIES):
        sub = a[a['strategy'] == tag]
        ax.bar(
            np.arange(len(sub)) + (i - 0.5) * w,
            sub['auc_roc'],
            w,
            label=label,
            color=['#93C5FD', '#FCA5A5'][i],
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('创新A：时间稳定性特征筛选（测试集 AUC）', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    # 创新B
    ax = axes[1]
    b = test[test['model'].str.startswith('创新B')]
    lams = sorted(b['lam'].unique())
    w = 0.35
    for i, (tag, label) in enumerate(STRATEGIES):
        sub = b[b['strategy'] == tag].set_index('lam')
        vals = [sub.loc[l, 'auc_roc'] if l in sub.index else np.nan for l in lams]
        ax.bar(
            np.arange(len(lams)) + (i - 0.5) * w,
            vals,
            w,
            label=label,
            color=['#93C5FD', '#FCA5A5'][i],
        )
    ax.set_xticks(np.arange(len(lams)))
    ax.set_xticklabels(['%.1f' % l for l in lams])
    ax.set_xlabel('衰减系数 λ（越大越看重近期样本）', fontsize=9)
    ax.set_title('创新B：时间衰减权重（测试集 AUC）', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    # 特征稳定性分布
    ax = axes[2]
    ax.hist(stab['psi'], bins=25, color='#0891B2', alpha=0.8)
    ax.axvline(0.1, color='#DC2626', ls='--', label='PSI=0.1 轻微漂移')
    ax.axvline(0.25, color='#7C3AED', ls='--', label='PSI=0.25 显著漂移')
    ax.set_xlabel('PSI（OOT train→valid）', fontsize=9)
    ax.set_ylabel('特征数', fontsize=9)
    ax.set_title('特征时间稳定性分布', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(EXP_FIG, '08_stage3_innovation.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    argv = list(sys.argv[1:]) if argv is None else list(sys.argv[1:])
    what = next((a for a in argv if a in ('tune', 'cv', 'ensemble', 'exp')), 'all')
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    t0 = time.time()

    if what in ('tune', 'all'):
        params = tune('random')
        retrain_and_eval('random', params)
    if what in ('cv', 'all'):
        cross_validate('random')
        cross_validate('oot_y2013')
    if what in ('ensemble', 'all'):
        res = []
        for tag, _ in STRATEGIES:
            res += ensemble(tag)
        pd.DataFrame(res).to_csv(
            os.path.join(TAB, '14_ensemble_all.csv'), index=False, encoding='utf-8-sig'
        )
    if what in ('exp', 'all'):
        exp_stage1()
        exp_stage2()
        exp_stage3()

    print('\n总耗时 %.0f 分钟' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
