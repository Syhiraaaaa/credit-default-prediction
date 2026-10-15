# -*- coding: utf-8 -*-
"""信贷违约预测项目：数据加载与初步探索（第 1 阶段）

输入：项目资料/credit_data(1).csv（800,000 行 x 47 列）
输出：
    outputs/tables/01_column_overview.csv       每列 dtype / 非空数 / 缺失数 / 缺失率 / 唯一值数
    outputs/tables/01_target_distribution.csv   目标列 isDefault 分布
    outputs/tables/01_numeric_describe.csv      数值列描述统计
    outputs/logs/01_data_loading.log            运行日志

运行环境：Python 3.11（conda 环境 credit），依赖见项目根目录 requirements.txt

检查内容：
1. 全量加载（耗时、形状、内存占用）
2. 数据类型分布
3. 缺失值统计（缺失规模、缺失模式分组）
4. 每列总览
5. 目标变量分布
6. 数值列描述统计
7. 非数值列概览
8. 重复值与主键检查
9. 数据样例
"""

import os
import sys
import time

import numpy as np
import pandas as pd


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
TABLE_DIR = os.path.join(BASE, 'outputs', 'tables')
LOG_DIR = os.path.join(BASE, 'outputs', 'logs')

TARGET = 'isDefault'


class Tee:
    """控制台与日志文件双向输出。"""

    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(LOG_DIR, '01_data_loading.log'))

    t0 = time.time()
    print('=' * 70)
    print('信贷违约预测 · 第 1 阶段 数据加载与初步探索')
    print(pd.Timestamp.now().strftime('运行时间：%Y-%m-%d %H:%M:%S'))
    print('=' * 70)

    # 1. 全量加载
    df = pd.read_csv(DATA_PATH)
    print('\n【1】全量加载')
    print('  加载耗时：%.1f 秒' % (time.time() - t0))
    print('  数据形状：%d 行 × %d 列' % (df.shape[0], df.shape[1]))
    print('  内存占用：%.1f MB' % (df.memory_usage(deep=True).sum() / 1024 ** 2))

    # 2. 数据类型分布
    print('\n【2】数据类型分布')
    for dtype, cnt in df.dtypes.astype(str).value_counts().items():
        print('  %-10s %3d 列' % (dtype, cnt))

    # 3. 缺失值统计
    missing = df.isna().sum()
    missing_pct = missing / len(df) * 100
    missing_df = (
        pd.DataFrame({'缺失数': missing, '缺失率(%)': missing_pct.round(2)})
        .query('缺失数 > 0')
        .sort_values('缺失数', ascending=False)
    )
    print('\n【3】缺失值统计')
    print('  有缺失的列：%d / %d' % (len(missing_df), df.shape[1]))
    if len(missing_df):
        print(missing_df.to_string())
        print('\n  缺失数相同的列分组：')
        for n, grp in missing_df.groupby('缺失数'):
            print('    缺 %7d 行（%5.2f%%）：%s'
                  % (n, n / len(df) * 100, ', '.join(grp.index.tolist())))

    # 4. 每列总览
    overview = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        '非空数': df.notna().sum(),
        '缺失数': missing,
        '缺失率(%)': missing_pct.round(2),
        '唯一值数': df.nunique(dropna=True),
    })
    overview.index.name = '列名'
    overview.to_csv(os.path.join(TABLE_DIR, '01_column_overview.csv'), encoding='utf-8-sig')

    # 5. 目标变量分布
    print('\n【4】目标变量分布')
    vc = df[TARGET].value_counts().sort_index()
    target_df = pd.DataFrame({'样本数': vc, '占比(%)': (vc / len(df) * 100).round(2)})
    target_df.index.name = TARGET
    print(target_df.to_string())
    print('  违约率：%.2f%%（约 1:%.1f）'
          % (df[TARGET].mean() * 100, (1 - df[TARGET].mean()) / df[TARGET].mean()))
    target_df.to_csv(os.path.join(TABLE_DIR, '01_target_distribution.csv'), encoding='utf-8-sig')

    # 6. 数值列描述统计
    print('\n【5】数值列描述统计')
    num_df = df.select_dtypes(include=[np.number])
    describe = num_df.describe().T
    describe['缺失数'] = missing[num_df.columns]
    describe = describe.round(3)
    print(describe.to_string())
    describe.to_csv(os.path.join(TABLE_DIR, '01_numeric_describe.csv'), encoding='utf-8-sig')

    # 7. 非数值列概览
    print('\n【6】非数值列概览')
    obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print('  共 %d 列' % len(obj_cols))
    for col in obj_cols:
        top = df[col].value_counts(dropna=True).head(3)
        print('  %s：唯一值 %d，Top3 = %s' % (col, df[col].nunique(), dict(top)))

    # 8. 重复值与主键检查
    print('\n【7】重复值与主键检查')
    print('  完全重复行：%d' % df.duplicated().sum())
    print('  重复 id：%d，id 范围 %d ~ %d'
          % (df['id'].duplicated().sum(), df['id'].min(), df['id'].max()))

    # 9. 数据样例
    print('\n【8】数据样例（前 3 行，转置）')
    print(df.head(3).T.to_string())

    print('\n' + '=' * 70)
    print('完成。总耗时 %.1f 秒' % (time.time() - t0))
    print('=' * 70)


if __name__ == '__main__':
    main()
