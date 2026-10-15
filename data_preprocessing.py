# -*- coding: utf-8 -*-
"""信贷违约预测项目：数据预处理（第 2 阶段）

输入：项目资料/credit_data(1).csv（800,000 行 x 47 列）
输出：db/processed/credit_clean.parquet（800,000 行 x 49 列）

处理内容：
1. 数据加载与副本建立
2. 缺失值检查（缺失规模、缺失与违约的关系、成批缺失的时间分布）
3. 异常值检查（IQR 围栏、3σ 原则、业务规则）
4. 重复与一致性检查（重复行、重复列、字段逻辑一致性、常数列与类别基数）
5. 清洗执行（修正错误值、删除无效列、增设缺失指示列、字段变换、缺失填充、日期解析）
6. 清洗结果校验
7. 结果保存与回读验证
8. 偏度检查与对数变换效果验证

运行环境：credit（Python 3.10 / pandas / numpy / scipy / pyarrow）
运行方式：python data_preprocessing.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

pd.set_option('display.max_columns', 60)


# ---------------------------------------------------------------------------
# 路径定位
# ---------------------------------------------------------------------------
# 自动向上定位项目根目录（以 项目资料/credit_data(1).csv 为锚点）
def find_base():
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(cur, '项目资料', 'credit_data(1).csv')):
            return cur
        cur = os.path.dirname(cur)
    raise FileNotFoundError('未找到项目根目录')


BASE = find_base()
DATA_PATH = os.path.join(BASE, '项目资料', 'credit_data(1).csv')
OUT_DIR = os.path.join(BASE, 'db', 'processed')


# ---------------------------------------------------------------------------
# 2. 缺失值检查
# ---------------------------------------------------------------------------
def check_missing(data):
    """输出缺失规模、缺失组与完整组的违约率差异（卡方检验）、成批缺失的年份分布。"""
    miss = data.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print('有缺失的列：%d / %d' % (len(miss), data.shape[1]))
    print(miss.to_string())

    print('\n缺失组与完整组的违约率差异（卡方检验）：')
    for name, mask in [
        ('成批 n 列(11列)', data['n0'].isna()),
        ('n4 / n10', data['n4'].isna()),
        ('n11', data['n11'].isna()),
        ('employmentLength', data['employmentLength'].isna()),
        ('revolUtil', data['revolUtil'].isna()),
        ('pubRecBankruptcies', data['pubRecBankruptcies'].isna()),
        ('dti', data['dti'].isna()),
    ]:
        rate_missing = data.loc[mask, 'isDefault'].mean() * 100
        rate_full = data.loc[~mask, 'isDefault'].mean() * 100
        p = chi2_contingency(pd.crosstab(mask, data['isDefault']))[1]
        verdict = '差异显著' if p < 0.05 else '差异不显著'
        print('%-18s 缺失 %7d 行 | 缺失组 %5.2f%%  完整组 %5.2f%%  差 %+5.2fpp | p=%.2e  %s'
              % (name, int(mask.sum()), rate_missing, rate_full,
                 rate_missing - rate_full, p, verdict))

    year = data['issueDate'].str[:4].astype(int)
    tab = data.groupby(year).agg(笔数=('id', 'size'), n列缺失数=('n0', lambda s: s.isna().sum()))
    tab['缺失占比%'] = (tab['n列缺失数'] / tab['笔数'] * 100).round(2)
    print('\n成批缺失的年份分布：')
    print(tab.to_string())

    # 成批缺失组与完整组的关键字段均值对比
    m = data['n0'].isna()
    pairs = [('ficoRangeLow', 'FICO 分数下限'), ('loanAmnt', '贷款金额'),
             ('annualIncome', '年收入'), ('dti', '债务收入比'), ('interestRate', '利率')]
    print('\n成批缺失组与完整组的关键字段均值：')
    print('%-18s %12s %12s %12s' % ('字段', '完整组', '缺失组', '差值'))
    for c, label in pairs:
        full, miss_ = data.loc[~m, c].mean(), data.loc[m, c].mean()
        print('%-18s %12.2f %12.2f %+12.2f' % (label, full, miss_, miss_ - full))


# ---------------------------------------------------------------------------
# 3. 异常值检查
# ---------------------------------------------------------------------------
def check_outliers_iqr(data, cols):
    """IQR 围栏法：输出各列上围栏、超出行数、占比与该组违约率。"""
    print('%-14s %10s %10s %10s %10s %10s %9s %7s %10s'
          % ('字段', 'Q1', '中位数', 'Q3', 'IQR', '上围栏', '超出行数', '占比', '该组违约率'))
    for c in cols:
        s = data[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        hi = q3 + 1.5 * (q3 - q1)
        m = s > hi
        print('%-14s %10.2f %10.2f %10.2f %10.2f %10.2f %9d %6.2f%% %9.2f%%'
              % (c, q1, s.median(), q3, q3 - q1, hi, int(m.sum()), m.mean() * 100,
                 data.loc[s[m].index, 'isDefault'].mean() * 100))


def check_outliers_3sigma(data, cols):
    """3σ 原则：输出各列均值、标准差、3σ 区间、超出条数、占比与该组违约率。"""
    print('%-14s %10s %10s %10s %10s %9s %7s %10s'
          % ('字段', '均值', '标准差', '3σ 下限', '3σ 上限', '超出条数', '占比', '该组违约率'))
    for c in cols:
        s = data[c].dropna()
        lo, hi = s.mean() - 3 * s.std(), s.mean() + 3 * s.std()
        m = (s < lo) | (s > hi)
        if m.sum():
            rate_str = '%9.2f%%' % (data.loc[s[m].index, 'isDefault'].mean() * 100)
        else:
            rate_str = '%9s' % '-'
        print('%-14s %10.2f %10.2f %10.2f %10.2f %9d %6.2f%% %s'
              % (c, s.mean(), s.std(), lo, hi, int(m.sum()), m.mean() * 100, rate_str))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------------
    # 1. 数据加载与副本建立
    # -----------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    print('原始数据：%s' % (df.shape,))

    # 原始数据保持不动，清洗在副本上进行
    df_clean = df.copy()

    # -----------------------------------------------------------------------
    # 2. 缺失值检查
    # -----------------------------------------------------------------------
    check_missing(df_clean)

    # -----------------------------------------------------------------------
    # 3. 异常值检查
    # -----------------------------------------------------------------------
    print('\nIQR 围栏检查：')
    check_outliers_iqr(df_clean, ['dti', 'annualIncome', 'revolUtil', 'loanAmnt'])

    print('\n3σ 原则检查：')
    check_outliers_3sigma(df_clean, ['loanAmnt', 'annualIncome', 'dti', 'revolBal',
                                     'revolUtil', 'interestRate', 'openAcc', 'installment'])

    print('\n业务规则检查：')
    print('dti 为负数：%d 行' % int((df_clean['dti'] < 0).sum()))
    print('dti = 999：%d 行' % int((df_clean['dti'] == 999).sum()))
    print('dti 取值范围：%.1f ~ %.1f' % (df_clean['dti'].min(), df_clean['dti'].max()))
    print('\n极端值的违约率对比（整体基准 %.2f%%）：' % (df_clean['isDefault'].mean() * 100))
    rules = [
        ('dti > 50', df_clean['dti'] > 50),
        ('revolUtil > 100', df_clean['revolUtil'] > 100),
        ('revolUtil > 150', df_clean['revolUtil'] > 150),
        ('annualIncome = 0', df_clean['annualIncome'] == 0),
        ('annualIncome > 100 万', df_clean['annualIncome'] > 1_000_000),
        ('annualIncome > p99', df_clean['annualIncome'] > df_clean['annualIncome'].quantile(0.99)),
    ]
    for name, m in rules:
        print('%-24s 命中 %9d 行 %7.3f%% | 该组违约率 %6.2f%%  其余 %6.2f%%'
              % (name, int(m.sum()), m.mean() * 100,
                 df_clean.loc[m, 'isDefault'].mean() * 100,
                 df_clean.loc[~m, 'isDefault'].mean() * 100))

    # -----------------------------------------------------------------------
    # 4. 重复与一致性检查
    # -----------------------------------------------------------------------
    print('\n完全重复的行数：%d' % df_clean.duplicated().sum())
    print('id 唯一值个数：%d | 总行数：%d' % (df_clean['id'].nunique(), len(df_clean)))

    same = (df_clean['n2'] == df_clean['n3']) | (df_clean['n2'].isna() & df_clean['n3'].isna())
    print('n2 与 n3 逐行相同的比例：%.2f%%' % (same.mean() * 100))

    print('FICO 下限 <= 上限：%s' % bool((df_clean['ficoRangeLow'] <= df_clean['ficoRangeHigh']).all()))
    both = df_clean[['pubRec', 'pubRecBankruptcies']].dropna()
    print('破产数 <= 公共记录总数：违反 %d 行' % int((both['pubRec'] < both['pubRecBankruptcies']).sum()))
    d_issue = pd.to_datetime(df_clean['issueDate'], format='%Y-%m-%d')
    d_credit = pd.to_datetime(df_clean['earliesCreditLine'], format='%b-%Y')
    print('最早信用线早于放款日：%s' % bool((d_credit < d_issue).all()))

    print('\npolicyCode 取值：%s（恒为 1，无区分度）' % df_clean['policyCode'].unique())
    print('高基数类别列：')
    for c in ['title', 'employmentTitle', 'postCode', 'regionCode']:
        print('  %-18s %8d 个取值' % (c, df_clean[c].nunique()))
    print('低基数类别列：')
    for c in ['grade', 'subGrade', 'term', 'homeOwnership', 'verificationStatus',
              'purpose', 'applicationType', 'initialListStatus']:
        print('  %-18s %8d 个取值' % (c, df_clean[c].nunique()))

    # -----------------------------------------------------------------------
    # 5. 清洗执行
    # -----------------------------------------------------------------------
    # ① 修正逻辑错误：dti 的 999 占位值与负值置为缺失，不删行
    df_clean.loc[df_clean['dti'].isin([999]) | (df_clean['dti'] < 0), 'dti'] = np.nan

    # ② 删除无信息列：n3（与 n2 完全重复）、policyCode（恒为 1）
    df_clean = df_clean.drop(columns=['n3', 'policyCode'])

    # ③ 按缺失批次增设 4 个缺失指示列
    df_clean['isna_n_main'] = df_clean['n0'].isna().astype('int8')       # 11 个 n 列成批缺失
    df_clean['isna_n4n10'] = df_clean['n4'].isna().astype('int8')        # n4/n10 成对缺失
    df_clean['isna_n11'] = df_clean['n11'].isna().astype('int8')
    df_clean['isna_emp_len'] = df_clean['employmentLength'].isna().astype('int8')

    # ④ employmentLength 文本转数值序数（0~10），缺失保留为 NaN
    EMP_MAP = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
               '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
               '10+ years': 10}
    df_clean['employmentLength'] = df_clean['employmentLength'].map(EMP_MAP)

    # ⑤ 零星缺失填充
    df_clean['pubRecBankruptcies'] = df_clean['pubRecBankruptcies'].fillna(0)
    for c in ['dti', 'revolUtil', 'title', 'employmentTitle']:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())
    df_clean['postCode'] = df_clean['postCode'].fillna(df_clean['postCode'].mode()[0])

    # ⑥ 日期文本转 datetime
    df_clean['issueDate'] = pd.to_datetime(df_clean['issueDate'], format='%Y-%m-%d')
    df_clean['earliesCreditLine'] = pd.to_datetime(df_clean['earliesCreditLine'], format='%b-%Y')

    # -----------------------------------------------------------------------
    # 6. 清洗结果校验
    # -----------------------------------------------------------------------
    print('\n清洗结果校验：')
    print('行数：%d' % len(df_clean))
    print('列数：%d（原 47 - 删 2 + 加 4 = 49）' % df_clean.shape[1])
    print('违约率：%.4f%%（与原始数据一致）' % (df_clean['isDefault'].mean() * 100))
    print('id 唯一：%s' % (df_clean['id'].nunique() == len(df_clean)))
    remain = df_clean.columns[df_clean.isna().any()].tolist()
    print('剩余有缺失的列数：%d：%s' % (len(remain), remain))
    print('缺失指示列均值（缺失率%）：')
    print((df_clean[['isna_n_main', 'isna_n4n10', 'isna_n11', 'isna_emp_len']].mean() * 100)
          .round(2).to_string())
    print('employmentLength 序数化结果：')
    print(df_clean['employmentLength'].describe().round(2).to_string())

    # -----------------------------------------------------------------------
    # 7. 结果保存与回读验证
    # -----------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'credit_clean.parquet')
    df_clean.to_parquet(out_path, index=False)
    print('\n已保存：%s（%.1f MB）' % (out_path, os.path.getsize(out_path) / 1024 / 1024))
    check = pd.read_parquet(out_path)
    print('回读验证：形状 %s' % (check.shape,))

    # -----------------------------------------------------------------------
    # 8. 偏度检查与对数变换效果验证
    # -----------------------------------------------------------------------
    skew_cols = ['loanAmnt', 'annualIncome', 'revolBal', 'dti', 'revolUtil', 'interestRate',
                 'delinquency_2years', 'openAcc']
    print('\n各数值列的偏度：')
    print(df_clean[skew_cols].skew().sort_values(ascending=False).round(2).to_string())

    print('\n对数变换（log1p）前后偏度对比：')
    print('%-22s %10s %10s' % ('字段', '变换前偏度', '变换后偏度'))
    for c in ['annualIncome', 'revolBal', 'loanAmnt', 'delinquency_2years']:
        before = df_clean[c].skew()
        after = np.log1p(df_clean[c].clip(lower=0)).skew()
        print('%-22s %10.2f %10.2f' % (c, before, after))
    print('正式的对数变换在特征工程阶段执行，且只使用训练集的统计量。')


if __name__ == '__main__':
    main()
