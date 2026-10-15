# -*- coding: utf-8 -*-
"""信贷违约预测业务仪表板。

Streamlit + Plotly 交互式看板，读取 outputs/tables 下已产出的结果表，
不重新训练模型。最终模型为最优配置（深6 / lr 0.02 / sub 0.8 / col 0.7 /
λ 30 / mcw 24 / γ 0，迭代上限 3000，seed 42 峰值轮 2920）。
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _find_tables():
    """定位结果表目录：优先 outputs/tables，其次随脚本同级的 outputs/tables。"""
    for base in (ROOT, Path(__file__).resolve().parent):
        d = base / 'outputs' / 'tables'
        if d.is_dir():
            return d
    d = Path(__file__).resolve().parent / 'outputs' / 'tables'
    d.mkdir(parents=True, exist_ok=True)
    return d


TAB = _find_tables()

MAIN = '#4C72B0'      # 蓝：主对象
ALT = '#DD8452'       # 橙：对比/突出
GREY = '#8C8C8C'      # 灰：次要
LIGHT = '#CFCFCF'     # 浅灰：基线
LINE = '#555555'      # 深灰：参考线
INK = '#1F2430'       # 正文深色
FONT = 'Microsoft YaHei, PingFang SC, SimSun, sans-serif'

st.set_page_config(page_title='信贷违约预测业务仪表板', page_icon='📊',
                   layout='wide', initial_sidebar_state='expanded')

CSS = """
<style>
  .stApp { background: #F5F6F8; }
  .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
  h1, h2, h3, h4 { font-family: @FONT@ !important; color: @INK@ !important; }
  /* 页标题（markdown ##）：Streamlit 默认偏小，这里放大并留足上下间距 */
  h2 { font-size: 30px !important; font-weight: 700 !important;
       margin: 36px 0 16px !important; letter-spacing: -.4px; }
  h2:first-of-type { margin-top: 6px !important; }
  h3 { font-size: 21px !important; font-weight: 650 !important;
       margin: 24px 0 12px !important; }
  /* 正文：Streamlit 默认 14px 偏小，整体放大到 17px 并放宽行距 */
  .stApp p, .stApp li { font-size: 17px; line-height: 1.95; }
  .stApp [data-testid="stMarkdownContainer"] p { line-height: 1.95; }
  .kpi {
      background: #FFFFFF; border-radius: 10px; padding: 16px 18px;
      box-shadow: 0 1px 3px rgba(31,36,48,.08), 0 1px 2px rgba(31,36,48,.04);
      border-left: 4px solid @MAIN@; height: 100%;
  }
  /* KPI 卡的标签与副标题原来 12/11px，比正文小太多，一并放大 */
  .kpi .lab { font-size: 15px; color: #5F6779; letter-spacing: .3px;
              font-weight: 600; line-height: 1.5;
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  /* 值可能较长（如 "6 棵树深 / 0.02 步长"），缩到能完整显示，不靠省略号截断 */
  .kpi .val { font-size: clamp(20px, 1.7vw, 27px); font-weight: 700; color: @INK@;
              margin: 6px 0 2px; font-family: @FONT@; letter-spacing: -.2px;
              white-space: nowrap; }
  .kpi .sub { font-size: 13.5px; color: #7C8494; line-height: 1.6;
              white-space: normal; }
  .kpi.alt { border-left-color: @ALT@; }
  .panel {
      background: #FFFFFF; border-radius: 10px; padding: 18px 20px;
      box-shadow: 0 1px 3px rgba(31,36,48,.08); margin-bottom: 14px;
  }
  .src { font-size: 12px; color: #9AA0AC; margin-top: 8px; }
  /* 图/表的小标题：标题名与数据来源标记同一行，来源悬停显示 */
  .fig-head { display: flex; align-items: center; gap: 7px;
              margin: 6px 0 8px; line-height: 1.35; }
  .fig-head b { font-size: 17.5px; font-weight: 600; color: #262D3D;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .src-tag { display: inline-block; font-size: 11px; color: #AEB4C0;
             cursor: help; border: 1px solid #DDE1E8; border-radius: 50%;
             width: 15px; height: 15px; line-height: 14px; text-align: center;
             flex: none; margin: 0; }
  .src-tag:hover { color: @MAIN@; border-color: @MAIN@; }
  /* note 是图下方的解读文字，字号放大、行距放宽，长段落也能读得下去 */
  .note { font-size: 17px; color: #3E4658; background: #EDF1F7;
          border-radius: 10px; padding: 18px 22px; margin-top: 12px;
          line-height: 2.05; }
  /* 关键结论：淡黄底 + 左侧色条，用于页面里最该被看到的那一句 */
  .note.key { background: #FFF4DC; border-left: 5px solid @ALT@;
             color: #4A3A1E; box-shadow: 0 2px 12px rgba(190,140,50,.18); }
  /* 关键数字高亮：底色 + 略微加深，用于结论里的核心数值 */
  .note mark, .panel mark { background: #FFD97A; color: #6B3F00;
                            padding: 1px 6px; border-radius: 5px;
                            font-weight: 700; box-shadow: inset 0 -1px 0 #E8B84B; }
  /* 长段落之间要留呼吸的空隙：<br> 只是硬换行，视觉上还是一整块，
     读者分不清哪里是一个意思说完了。把 note 里的 br 变成块级并给上间距，
     全站 58 个解读块一次生效，不用逐个改文案。 */
  .note br, .panel-txt br {
      display: block; content: ""; margin-top: 14px; }
  /* 紧跟在开头标签后的第一个 br 不留上间距，否则块顶部会多出一截空 */
  .note > br:first-child, .panel-txt > br:first-child { margin-top: 0; }
  /* 口径 / 前提这类"读数字前必须知道"的信息：淡蓝底 + 蓝边，与结论块区分 */
  .note.tip { background: #E9F1FB; border-left: 5px solid @MAIN@;
              color: #33415A; }
  /* panel 内的说明文字：原来内联 13px 太小，改走 class 统一字号 */
  .panel-txt { font-size: 16.5px; line-height: 1.95; color: #3A4150; }
  /* 表格：Streamlit 默认 14px 偏小；表格有放大空间，字号跟着正文走。
     行高与内边距同步放大，否则放大字号后文字会被上一行压住。 */
  .stApp [data-testid="stDataFrame"] { font-size: 16px; }
  .stApp [data-testid="stDataFrame"] * { font-size: inherit; }
  .stApp [data-testid="stDataFrame"] div[role="columnheader"],
  .stApp [data-testid="stDataFrame"] div[role="gridcell"] {
      font-size: 16px !important; line-height: 1.85;
      padding-top: 12px !important; padding-bottom: 12px !important;
      white-space: normal !important; height: auto !important; }
  /* 表格默认固定行高会截断换行后的文字，这里放开 */
  .stApp [data-testid="stDataFrame"] div[role="row"] { height: auto !important; }
  .stApp [data-testid="stDataFrameStickyColumn"],
  .stApp [data-testid="stDataFrame"] canvas { height: auto !important; }
  /* 开标签 <div class="panel"> 被 Streamlit 单独渲染成块，内部无内容
     → 显示为空白色圆角盒子。隐藏之，由其后的图表/表格承载视觉。 */
  .stApp div.panel:empty,
  .stApp div.panel:not(:has(> *)):not(:has(> span)) { display: none; }
  /* 参数页的分页（st.tabs）：7 个标签横排挤在一起，读者分不清这是一组切换控件。
     这套选择器必须覆盖 baseweb 各种可能的挂载位置——历史上用
     [data-baseweb="tab"] 改过一轮，截图验证背景完全没生效（前端里没这个属性）。
     现在多路并挂，只要命中任意一路就能上样式。 */
  .stApp [data-testid="stTabs"],
  .stApp [data-testid="stVerticalBlockBorderWrapper"],
  .stApp [data-testid="stTabs"] > div {
      background: #F1F4F9; border: 1px solid #DFE4EC;
      border-radius: 10px; padding: 12px 16px 6px; }
  .stApp [data-testid="stTabs"] button,
  .stApp [data-testid="stTabs"] [role="tab"],
  .stApp [role="tablist"] button {
      background: #FFFFFF !important; border: 1px solid #D5DCE8 !important;
      border-radius: 8px !important; padding: 7px 16px !important;
      margin-right: 8px !important; margin-bottom: 10px !important;
      font-size: 16px !important; font-weight: 600 !important;
      color: #4A5265 !important; box-shadow: 0 1px 2px rgba(31,36,48,.07);
      min-width: auto !important; height: auto !important; }
  .stApp [data-testid="stTabs"] button:hover,
  .stApp [role="tablist"] button:hover {
      background: #E6EDF8 !important; border-color: #B9C8DE !important; }
  /* 选中态：主色实心。baseweb 可能把 active 类挂在 button 上，
     也可能落在内层 span，两处都压。 */
  .stApp [data-testid="stTabs"] button[aria-selected="true"],
  .stApp [data-testid="stTabs"] button[aria-selected="true"] *,
  .stApp [role="tablist"] button[aria-selected="true"],
  .stApp [role="tablist"] button[aria-selected="true"] *,
  .stApp [data-testid="stTabs"] button[data-active],
  .stApp [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
      background: @MAIN@ !important; border-color: @MAIN@ !important;
      color: #FFFFFF !important; }
  /* baseweb 默认给选中标签画了一条下划线，和我们的实心底冲突，去掉 */
  .stApp [data-testid="stTabs"] button::after,
  .stApp [role="tablist"] button::after { display: none !important; }
  [data-testid="stSidebar"] { background: #FFFFFF; padding-top: 0.9rem; }
  /* 侧栏宽度：默认 336px 对纯导航太宽（右侧一片空白），收到 270px */
  [data-testid="stSidebar"] { width: 270px !important; min-width: 270px !important; }
  [data-testid="stSidebarCollapsed"] { width: 0 !important; }
  /* 侧栏内容不需要滚动条：宽度收窄到刚好放下导航文字 + 顶部空隙压小 */
  [data-testid="stSidebar"] > div:first-child { overflow-y: visible; }
  [data-testid="stSidebarUserContent"] { overflow-y: visible; }
  [data-testid="stSidebar"] [role="radiogroup"] { gap: 0px; }
  /* 导航项行距收到 6px / 1.4，8 个页面名要在一屏内全露出，不出现侧栏滚动 */
  [data-testid="stSidebar"] [role="radiogroup"] label {
      padding: 6px 10px; border-radius: 8px; margin-bottom: 0px;
      font-size: 16px; font-weight: 600; color: #3A4252;
      line-height: 1.4; transition: background .12s, color .12s; }
  [data-testid="stSidebar"] [role="radiogroup"] label:hover {
      background: #F2F4F8; color: @MAIN@; }
  [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
  [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
      background: #EAF0FA; color: @MAIN@; font-weight: 700; }
  [data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
      width: 17px; height: 17px; margin-right: 10px; }
  [data-testid="stSidebar"] hr { margin: 12px 0 10px; border-color: #E6E9EF; }
  [data-testid="stMetricValue"] { font-family: @FONT@; color: @INK@; }
</style>
"""
for _k, _v in [('@FONT@', FONT), ('@INK@', INK), ('@MAIN@', MAIN), ('@ALT@', ALT)]:
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)


def style(fig, h=340):
    # 图内不放标题——标题统一由图上方的 markdown 小标题承担，
    # 避免同一句话出现两遍；结论与特征判断全部放在图下方的解读文字里。
    fig.update_layout(
        template='plotly_white', height=h,
        font=dict(family=FONT, size=13, color=INK),
        margin=dict(l=64, r=24, t=52, b=48),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        legend=dict(orientation='h', y=1.0, yanchor='bottom',
                    x=1, xanchor='right', bgcolor='rgba(0,0,0,0)',
                    font=dict(size=11.5)),
        hoverlabel=dict(font_family=FONT, font_size=13),
    )
    fig.update_xaxes(gridcolor='#ECEEF2', zerolinecolor='#ECEEF2',
                     linecolor='#D8DCE3', ticks='outside', tickcolor='#D8DCE3')
    fig.update_yaxes(gridcolor='#ECEEF2', zerolinecolor='#ECEEF2',
                     linecolor='#D8DCE3', ticks='outside', tickcolor='#D8DCE3')
    return fig


def kpi(col, label, value, sub='', alt=False):
    col.markdown(
        '<div class="kpi%s"><div class="lab">%s</div>'
        '<div class="val">%s</div><div class="sub">%s</div></div>'
        % (' alt' if alt else '', label, value, sub), unsafe_allow_html=True)


SRC_LOG = []


def head(col, title, source=''):
    """图表小标题：标题名与数据来源标记排同一行，来源悬停显示。

    数据来源挂在标题右侧而不是图下方，避免一个孤立的小圆点占一行。
    """
    if source:
        SRC_LOG.append(source)
        tag = '<span class="src-tag" title="数据来源：%s">ⓘ</span>' % source
    else:
        tag = ''
    col.markdown('<div class="fig-head"><b>%s</b>%s</div>' % (title, tag),
                 unsafe_allow_html=True)


def takeaway(text):
    """每页最该被记住的那一句：淡黄底 + 左侧色条 + 轻微阴影。"""
    st.markdown('<div class="note key">%s</div>' % text,
                unsafe_allow_html=True)


@st.cache_data
def load(name):
    return pd.read_csv(TAB / name, encoding='utf-8-sig')


# ---------------------------------------------------------------- 数据
CURVES = load('50_curves_after_opt.csv')
THRESH = load('51_threshold_after_opt.csv')
CALIB = load('43_calibration_after_opt.csv')
CALIB_FIG = load('45_calibration_figure_after.csv')
PSI = load('44_psi_after_opt.csv')
LIFT = load('68_lift_decile.csv')
SHAP = load('09_shap_global_random.csv')
TARGET = load('01_target_distribution.csv')
TREND = load('03_monthly_trend.csv')
SPLIT = load('04_split_stats.csv')
TSCAN = load('76_threshold_scan.csv')
COST = load('77_cost_ratio_threshold.csv')
PEAK = load('97_peak_round_choice.csv')
CV_R = load('99_cv_final_c_random.csv')
CV_O = load('99_cv_final_c_oot_y2013.csv')
FPSI = load('08_feature_psi_stability.csv')
PROFILE = load('01b_combination_profile.csv')
CAT = load('03_categorical_default_rate.csv')
NUMDESC = load('01_numeric_describe.csv')
PSUM = load('101_param_sensitivity_summary.csv')      # 各超参取值的 AUC 分布
PREG = load('102_regularization_ablation.csv')        # 正则参数逐项消融
EGRP = load('103_experiment_groups.csv')              # 实验组汇总（调优总组数）
COVER = load('01_column_overview.csv')                # 47 个原始字段的类型与缺失
MISS = load('02_missing_vs_default.csv')              # 缺失组 vs 完整组违约率对比
ENC = load('04_encoding_summary_random.csv')         # 编码方式与新增列数

# 特征业务化：把字段名 / 独热编码列名换成业务可读名称，并给出业务含义
BIZ_NAME = {
    'subGrade_ord': '信用评级细分档位',
    'grade_ord': '信用评级主档位',
    'term': '贷款期限',
    'issue_year': '放款年份',
    'issue_month': '放款月份',
    'employmentTitle_te': '在职职位（按违约率编码）',
    'employmentLength': '在职年限',
    'loan_to_income': '贷款额 / 年收入',
    'revolBal_to_income': '循环授信余额 / 年收入',
    'revolUtil': '循环额度使用率',
    'revolUtil_sq2': '循环额度使用率（平方项）',
    'installment_to_income': '月供 / 年收入',
    'int_rate_sq2': '利率（平方项）',
    'dti': '负债收入比',
    'fico_mean': 'FICO 评分均值',
    'fico_range': 'FICO 评分跨度',
    'totalAcc': '历史账户总数',
    'openAcc': '当前未结账户数',
    'revolTotal': '循环授信总额',
    'revolUsed': '循环授信已用额',
    'pubRec': '公开违约记录数',
    'pubRecBankruptcies': '破产记录数',
    'delinq_2y': '近 2 年逾期次数',
    'delinq_2y_sq2': '近 2 年逾期（平方项）',
    'earliest_cr_line_yr': '最早信用记录年份',
    'homeOwnership': '住房所有权',
    'homeOwnership_1': '住房为自有（是）',
    'homeOwnership_2': '住房为按揭（是）',
    'homeOwnership_3': '住房为租赁（是）',
    'homeOwnership_4': '住房情况其他（是）',
    'homeOwnership_5': '住房信息缺失（是）',
    'addr_ext': '地址信息完整度',
    'postCode_te': '邮编（按违约率编码）',
    'regionCode_te': '地区（按违约率编码）',
    'title_te': '职位大类（按违约率编码）',
    'verificationStatus_te': '收入核验方式（按违约率编码）',
    'initial_list_status_te': '初始上市状态（按违约率编码）',
    'purpose_te': '贷款用途（按违约率编码）',
    'policy_te': '还款计划（按违约率编码）',
    'risk_segment': '风险分层',
    'loanAmnt': '放款金额',
    'funded_amnt': '到账金额',
    'interestRate': '年化利率',
    'installment': '月还款额',
    'annual_inc': '年收入',
    'annual_inc_joint': '联合年收入',
    'emp_length_num': '在职年限（数值化）',
    'emp_title_num': '职位层级（数值化）',
    'revol_bal_ratio': '循环授信使用比例',
    'delinq_ratio': '历史逾期占比',
    'n2': '原始匿名字段 n2',
    'n3': '原始匿名字段 n3',
    'n11': '原始匿名字段 n11',
    'n12': '原始匿名字段 n12',
    'n13': '原始匿名字段 n13',
    'n14': '原始匿名字段 n14',
    'poly_interestRate_x_dti': '利率 × 负债收入比（交叉项）',
    'poly_interestRate_x_term': '利率 × 期限（交叉项）',
    'poly_revolUtil_x_dti': '额度使用率 × 负债收入比（交叉项）',
    'poly_interestRate_sq2_x_term': '利率平方 × 期限（交叉项）',
    'poly_revolUtil_sq2_x_loan_to_income': '额度使用率平方 × 贷款收入比（交叉项）',
    'interestRate_bin': '利率分档',
    'dti_bin': '负债收入比分档',
    'annual_inc_bin': '收入分档',
    'fico_bin': 'FICO 分档',
    'revolBal': '循环授信余额',
    'annualIncome': '年收入',
    'credit_hist_days': '信用记录天数（征信历史长度）',
    'delinquency_2years': '近 2 年逾期天数',
    'verificationStatus': '收入核验方式',
    'homeOwnership_0': '住房信息完全缺失',
    'poly_dti_x_revolUtil': '负债率 × 额度使用率（交叉项）',
    'poly_interestRate_x_revolUtil': '利率 × 额度使用率（交叉项）',
    'poly_interestRate_x_fico_mean': '利率 × FICO 评分（交叉项）',
    'poly_revolUtil_x_fico_mean': '额度使用率 × FICO 评分（交叉项）',
    'isna_n11': '字段 n11 缺失标记',
    'n5': '原始匿名字段 n5',
    'n6': '原始匿名字段 n6',
    'n7': '原始匿名字段 n7',
    'n8': '原始匿名字段 n8',
}
BIZ_MEANING = {
    'subGrade_ord': '外部评级机构给出的信用档位，越差违约率越高，是最主要的判断依据',
    'grade_ord': '评级主档位（A / B / C / D / E），越靠后风险越高',
    'term': '还款期数。36 / 60 期为主，期限越长借款人压力越大',
    'issue_year': '放款年份。用于吸收 2008 金融危机等宏观年份的影响',
    'issue_month': '放款月份。用于捕捉季节性放款节奏',
    'employmentTitle_te': '在职职位，按历史违约率做目标编码。职位稳定性影响还款能力',
    'employmentLength': '在职年限，越长说明收入越稳定',
    'loan_to_income': '贷款额相对年收入的比例，该拿一年收入还多高额贷款风险更大',
    'revolBal_to_income': '循环授信余额占收入比重，反映已有负债压力',
    'revolUtil': '已用授信额度占获批额度的比例，越高说明信用被消耗越多',
    'revolUtil_sq2': '额度使用率的平方项，放大高使用率的影响',
    'installment_to_income': '月供占年收入比例，直接反映月度还款压力',
    'int_rate_sq2': '利率的平方项，放大高利率的影响',
    'dti': '负债收入比，即每月还款额占月收入比例',
    'fico_mean': 'FICO 评分均值，行业通用的信用分',
    'fico_range': 'FICO 评分波动幅度，波动大说明财务状况不稳定',
    'totalAcc': '历史开立的信用账户总数，越多说明信用记录越丰富',
    'openAcc': '当前仍在使用的账户数',
    'revolTotal': '获批的循环授信总额度',
    'revolUsed': '循环授信已使用额度',
    'pubRec': '公开的违约 / 破产记录次数，有记录属于严重负面',
    'pubRecBankruptcies': '破产公示记录次数',
    'delinq_2y': '近两年逾期 30 天以上的次数，最直接的近期风险信号',
    'delinq_2y_sq2': '逾期次数的平方项，放大多次逾期的负面影响',
    'earliest_cr_line_yr': '最早一笔信用记录的年份，越早说明征信积累越久',
    'homeOwnership': '住房所有权状况，自有住房通常更稳定',
    'homeOwnership_1': '住房为自有，作为稳定性的正向信号',
    'homeOwnership_2': '住房为按揭，负债属性更明显',
    'homeOwnership_3': '住房为租赁，稳定性相对弱',
    'homeOwnership_4': '住房情况为其他类别',
    'homeOwnership_5': '住房信息缺失，信息不完整本身是风险',
    'addr_ext': '地址信息完整度分档',
    'postCode_te': '邮编按历史违约率编码，代表地区信用环境',
    'regionCode_te': '地区按历史违约率编码，代表区域经济与信用水平',
    'title_te': '职位大类按违约率编码',
    'verificationStatus_te': '收入核验方式，核验越充分信息可信度越高',
    'initial_list_status_te': '初始上市状态（新增 / 转续）',
    'purpose_te': '贷款用途，不同用途的违约表现差异明显',
    'policy_te': '还款计划类型',
    'risk_segment': '按放款额与利率组合划分的风险分层',
    'loanAmnt': '单笔放款金额，金额越大风险敞口越高',
    'fundedAmnt': '实际到账金额',
    'funded_amnt': '实际到账金额',
    'interestRate': '年化利率，利率越高借款人压力越大',
    'installment': '月还款额',
    'annual_inc': '申请人年收入，还款能力的基础',
    'annual_inc_joint': '共同申请人的年收入',
    'emp_length_num': '在职年限（数值化），越长越稳定',
    'emp_title_num': '职位层级（数值化），越高收入通常越高',
    'revol_bal_ratio': '循环授信使用比例',
    'delinq_ratio': '历史逾期次数占比',
    'n2': '原始数据集中无语义命名的匿名字段，仅按取值分布参与建模',
    'n3': '原始数据集中无语义命名的匿名字段',
    'n11': '原始数据集中无语义命名的匿名字段',
    'n12': '原始数据集中无语义命名的匿名字段',
    'n13': '原始数据集中无语义命名的匿名字段',
    'n14': '原始数据集中无语义命名的匿名字段',
    'poly_interestRate_x_dti': '利率与负债率的交互项，同时偏高时风险叠加放大',
    'poly_interestRate_x_term': '利率与期限的交互项，长期高息还款压力大',
    'poly_revolUtil_x_dti': '额度使用率与负债率的交互项',
    'poly_interestRate_sq2_x_term': '高利率叠加长期限的放大效应',
    'poly_revolUtil_sq2_x_loan_to_income': '高额度使用叠加高贷款收入比的放大效应',
    'interestRate_bin': '利率分档后的编码',
    'dti_bin': '负债收入比分档后的编码',
    'annual_inc_bin': '收入分档后的编码',
    'fico_bin': 'FICO 分档后的编码',
}
BIZ_MEANING.update({
    'revolBal': '循环授信已用余额，直接体现当前负债水平',
    'annualIncome': '申请人年收入，还款能力的基础',
    'credit_hist_days': '首笔信用记录到放款日的间隔天数，越长说明征信历史越久',
    'delinquency_2years': '近两年累计逾期天数，逾期天数越多风险越高',
    'verificationStatus': '收入核验方式，核验越充分信息可信度越高',
    'homeOwnership_0': '住房信息完全缺失，信息缺失本身是风险信号',
    'poly_dti_x_revolUtil': '负债率与额度使用率同时偏高时，风险叠加放大',
    'poly_interestRate_x_revolUtil': '高利率叠加高额度使用率，还款压力叠加',
    'poly_interestRate_x_fico_mean': '高利率叠加低信用分，定价与资质不匹配',
    'poly_revolUtil_x_fico_mean': '高额度使用率叠加低信用分',
    'isna_n11': '该字段缺失时为 1，用于捕捉信息缺失模式',
    'n5': '原始数据集中无语义命名的匿名字段',
    'n6': '原始数据集中无语义命名的匿名字段',
    'n7': '原始数据集中无语义命名的匿名字段',
    'n8': '原始数据集中无语义命名的匿名字段',
})

# 独热编码列（字段_数字）统一给出业务名称与含义
_CAT_CN = {
    'homeOwnership': '住房所有权', 'purpose': '贷款用途',
    'verificationStatus': '收入核验方式', 'initial_list_status': '初始上市状态',
    'policy': '还款计划', 'risk_segment': '风险分层',
    'application_type': '申请类型', 'debt_settlement_flag': '债务重组标记',
}
for _f, _cn in _CAT_CN.items():
    for _i in range(0, 20):
        BIZ_NAME.setdefault('%s_%d' % (_f, _i), '%s · 类别 %d' % (_cn, _i))
        BIZ_MEANING.setdefault('%s_%d' % (_f, _i),
                               '%s 独热编码列，值为 1 表示属于类别 %d' % (_cn, _i))

DESC = NUMDESC.set_index(NUMDESC.columns[0])
AVG_LOAN = float(DESC.loc['loanAmnt', 'mean'])          # 单笔平均放款额
AVG_RATE = float(DESC.loc['interestRate', 'mean'])      # 平均年利率(%)
AVG_INST = float(DESC.loc['installment', 'mean'])       # 平均月供
N_TEST = int(SPLIT.loc[SPLIT['tag'] == 'random', 'test_n'].iloc[0])
DEF_TEST = float(SPLIT.loc[SPLIT['tag'] == 'random', 'test_def'].iloc[0]) / 100
OVERALL_DEF = float(TARGET.loc[TARGET['isDefault'] == 1, '占比(%)'].iloc[0]) / 100
N_FEATURE = int(SPLIT.loc[SPLIT['tag'] == 'random', 'n_features'].iloc[0])

# 显示名不叫「配置 A/B/C」——那是脚本内部的代号，业务侧看不懂；
# 改成按"正则强度"这一实际差异命名，基线与最终方案直接点名。
CFG_MAP = {'基线 第4阶段默认超参': '未调优基线',
           '甲 深7 λ10 mcw8（现配置）': '候选 1 · 弱正则',
           '乙 深6 λ20 mcw16': '候选 2 · 中正则',
           '丙 深6 λ30 mcw24': '最优配置 ★'}

FINAL = PEAK[(PEAK['切分'] == 'random') & (PEAK['特征集'] == '全特征')
             & (PEAK['配置'].str.startswith('丙')) & (PEAK['随机种子'] == 42)].iloc[0]
BASE = PEAK[(PEAK['切分'] == 'random') & (PEAK['特征集'] == '全特征')
            & (PEAK['配置'].str.startswith('基线')) & (PEAK['随机种子'] == 42)].iloc[0]
# 四组候选的平均测试 AUC（跨页共用：选型页与稳定性页都引用）
_PEAK_SEL = PEAK[(PEAK['切分'] == 'random') & (PEAK['特征集'] == '全特征')].copy()
_PEAK_SEL['显示名'] = _PEAK_SEL['配置'].map(CFG_MAP).fillna(_PEAK_SEL['配置'])
BEST_AUC = float(_PEAK_SEL.groupby('显示名')['测试 auc'].mean().max())
# 最优配置跨随机种子的成绩极差（选型页与表现页共用，说明结果稳不稳）
BEST_SEED_SPREAD = float(
    _PEAK_SEL[_PEAK_SEL['显示名'] == '最优配置 ★']['测试 auc'].max()
    - _PEAK_SEL[_PEAK_SEL['显示名'] == '最优配置 ★']['测试 auc'].min())
V4 = '第4阶段默认超参'
V5 = '第5阶段调优后'
# 前 3 档（各占 10%）累计拦住的违约客户占比 —— 风险分层页与结论页共用
BASE_RATE = float(TARGET.loc[TARGET['isDefault'] == 1, '占比(%)'].iloc[0]) / 100
_L5 = LIFT[LIFT['模型'] == V5].sort_values('十分位')['Lift'].values
CAP30 = float(np.cumsum(_L5)[2] / _L5.sum() * 100)

# 超参的业务含义（选型页的"最终参数一览"与"换参数会怎样"共用）
_MEAN = {
    '树深': '树最多往下分几层',
    '学习率': '每轮调整的步长',
    '样本采样比': '每棵树随机抽多少样本',
    '列采样比': '每次分裂随机抽多少特征',
    'L2正则': '对权重平方的惩罚强度',
    '最小子样本权重': '一个叶子的最少样本数',
    '分裂损失阈值': '分裂所需的最小收益',
    '迭代上限': '最多训练多少棵树',
}


def curve_tag(model, curve):
    return float(CURVES[(CURVES['模型'] == model) &
                        (CURVES['曲线'] == curve)]['指标值'].iloc[0])


KS4 = curve_tag(V4, 'KS_违约')
KS5 = curve_tag(V5, 'KS_违约')

# 单参数敏感度汇总（选型页画图用，结论页引用强度上界）
# 「迭代上限」在 101 表里没有「是否为最终取值」标记（该值由 17 号 JSON 记录），
# 为保证每个参数页都能算出"当前取值排名"，只纳入带标记的 7 个参数。
_KEEP = ['树深', '学习率', '样本采样比', '列采样比', 'L2正则',
         '最小子样本权重', '分裂损失阈值']
_PS = PSUM[PSUM['参数'].isin(_KEEP)].copy().sort_values(['参数', '取值'])
_infl = []
for _nm, _sub in _PS.groupby('参数', sort=False):
    _fin = _sub[_sub['是否为最终取值'] == '是']
    if not len(_fin):
        continue
    _fv = _fin['取值'].iloc[0]
    _fauc = _fin['均值AUC'].iloc[0]
    _top = _sub.loc[_sub['均值AUC'].idxmax()]
    _infl.append({'参数': _nm, '最终取值': _fv, '取值数': len(_sub),
                  '均值极差': float(_sub['均值AUC'].max() - _sub['均值AUC'].min()),
                  '最终取值排名': int((_sub['均值AUC'] > _fauc).sum() + 1),
                  '最优取值': _top['取值'],
                  '与最优差距': float(_fauc - _top['均值AUC'])})
INFL = pd.DataFrame(_infl).sort_values('均值极差', ascending=False)
N_GROUPS = int(EGRP['组数'].sum())

# ---------------------------------------------------------------- 导航
st.sidebar.markdown('<div style="font-size:18px;font-weight:700;color:%s;'
                    'margin-bottom:2px;line-height:1.35">信贷风险决策看板</div>'
                    '<div style="font-size:13.5px;color:#7A8090;line-height:1.55;'
                    'margin-bottom:10px">回答三个问题：风险有多大、用什么模型管、'
                    '依据是哪些特征</div>' % INK, unsafe_allow_html=True)
# 导航项：每个页面配一个 emoji 图标，字体放大到 16.5px（见 CSS）
_NAV_ICON = {'业务概览': '📊', '特征解释': '🔍', '模型选型': '⚙️', '模型表现': '📈',
             '阈值决策': '⚖️', '风险分层': '🧭', '稳定性与校准': '🛡️',
             '结论与局限': '📋'}
_pages = ['业务概览', '特征解释', '模型选型', '模型表现', '阈值决策',
          '风险分层', '稳定性与校准', '结论与局限']
PAGE = st.sidebar.radio('页面', _pages, format_func=lambda p: _NAV_ICON[p] + '  ' + p,
                        label_visibility='collapsed')

# 滚动位置控制：左边大导航切换要回到最上面，页内 tab 切换不能动。
# Streamlit 任何 rerun（含点 tab）都会执行自己的 scroll restoration，
# 所以脚本必须**每次**注入，用标记区分这次该"归零"还是"保位置"：
#  - _RESET='1'（刚换了页面）→ 反复压回顶部
#  - _RESET='0'（只是点了页内 tab）→ 把切走前的位置原样还回去
# 踩了三次才定位到真因，记住这两条：
#  1) st.html 不是 iframe（官方文档明确 not iframed），普通 st.html 不执行 script，
#     必须 unsafe_allow_javascript=True；components.v1.html 已废弃。
#  2) **滚动容器选择器不能带标签名**。线上实测：真正会滚的是
#     [data-testid=stMain]（scrollHeight 5752 / clientHeight 566），
#     它没有标签名，写成 div[data-testid=stMain] 一条都匹配不到 → 前面三次全部空跑。
#     documentElement / window / stAppViewContainer 的 scrollHeight 都等于 clientHeight，不滚。
if st.session_state.get('_last_page') != PAGE:
    # 换大页面：记住要归零。首屏 _last_page 为 None，也算换页。
    st.session_state['_want_top'] = '1'
elif st.session_state.get('_want_top') != '1':
    # 同一页面内的重跑（点 tab、滑块等）：不干预位置
    st.session_state['_want_top'] = '0'
_want_top = st.session_state.get('_want_top')
# 把切换瞬间的滚动位置记下来，页内 tab 切回来时要用
st.html(
    '<script>(function(){'
    'var MAIN="[data-testid=stMain]";'
    'var KEY="__dash_scroll";'
    'function m(){ return document.querySelector(MAIN); }'
    # 每次脚本跑都挂一个滚动监听，把真实位置实时写进 sessionStorage。
    # 监听挂一次就不摘（同一个函数引用，重复挂不会叠加执行）。
    'if(!window.__dashScrollHook){'
    '  window.__dashScrollHook=true;'
    '  window.addEventListener("scroll",function(){'
    '    var e=m();'
    '    if(e){ try{ sessionStorage.setItem(KEY,String(e.scrollTop)); }catch(x){} }'
    '  },true);'
    '}'
    'var WANT=%s;' % ("'1'" if _want_top == '1' else "'0'") +
    'var box=m();'
    'if(WANT==="1"){'
    # 换大页面：先把旧位置清掉，再反复压回顶部，直到 Streamlit 不再改写为止
    '  try{ sessionStorage.setItem(KEY,"0"); }catch(e){}'
    '  function top(){'
    '    var e=m();'
    '    if(e && e.scrollTop!==0){ e.scrollTop=0; e.scrollTo(0,0); }'
    '    if(document.documentElement.scrollTop!==0){'
    '      document.documentElement.scrollTop=0; }'
    '    if(document.body && document.body.scrollTop!==0){'
    '      document.body.scrollTop=0; }'
    '    if(window.scrollY!==0){ window.scrollTo(0,0); }'
    '  }'
    '  top();'
    '  var n=0;'
    '  var iv=setInterval(function(){ top(); if(++n>30){ clearInterval(iv); } },50);'
    '  try{'
    '    var t=document.querySelector("[data-testid=stAppViewContainer]")||document.body;'
    '    if(t){ new MutationObserver(function(){ top(); })'
    '              .observe(t,{childList:true,subtree:true}); }'
    '  }catch(e){}'
    '}else{'
    # 页内切换（点 tab）：把切走前的位置原样还回去。
    # 位置由上面的滚动监听实时写入 sessionStorage，这里读回来即可。
    '  var keep=0;'
    '  try{ keep=parseInt(sessionStorage.getItem(KEY)||"0",10)||0; }catch(e){}'
    '  var tries=0;'
    '  var iv2=setInterval(function(){'
    '    var e=m();'
    '    if(e && e.scrollTop!==keep){ e.scrollTop=keep; }'
    '    if(++tries>24){ clearInterval(iv2); }'
    '  },50);'
    '}'
    '})();</script>', unsafe_allow_javascript=True)
st.session_state['_last_page'] = PAGE
st.sidebar.markdown('---')
st.sidebar.markdown(
    '<div style="font-size:13.5px;color:#5C6474;line-height:1.7">'
    '<b>数据范围</b><br>'
    '80 万笔历史放款（2007~2018 年）<br>'
    '训练 70%% / 验证 20%% / 测试 10%%<br>'
    '测试集全程未参与训练与调参<br>'
    '<br><b>最终模型</b><br>'
    '梯度提升树（XGBoost）<br>'
    '测试 AUC %.4f / KS %.4f'
    # AUC 与 KS 都取 97 表同一行，避免两套口径混在一行里
    '</div>' % (FINAL['测试 auc'], float(FINAL['测试 ks'])),
    unsafe_allow_html=True)
# 拉杆只在阈值决策页出现（第 5 条：不在其他页占位）
if PAGE == '阈值决策':
    # LGD / 机会成本两个拉杆放在页面内「阈值决策」的滑杆区（见下方），
    # 放在侧边栏既突兀又跟本页内容脱节；这里只留默认值。
    LGD = st.session_state.get('_lgd', 0.60)
    OPP = st.session_state.get('_opp', 1900)
else:
    LGD, OPP = 0.60, 1900

if PAGE == '业务概览':
    st.markdown('## 业务概览')
    st.markdown(
        '<div class="panel"><div class="panel-txt">'
        '<b>业务背景</b>　本看板面向<b>消费信贷审批</b>场景：借款人提交申请后，'
        '风控需要在放款前给出<b>通过 / 拒绝</b>决策。数据为 2007—2018 年 '
        '80 万笔放款记录，覆盖借款人信用等级、收入负债、贷款期限与利率等信息，'
        '整体违约率 <b>19.95%</b>。<br>'
        '<b>决策难点</b>　两类错误的代价并不对等：<b>漏拒</b>（违约客户仍通过）'
        '损失的是本金，按 LGD 折算金额大；<b>误拒</b>（正常客户被拒）损失的是'
        '这笔业务的利息收入，金额小但影响规模。故审批策略不是追求准确率最高，'
        '而是在两类错误之间找<b>总成本最低</b>的阈值。<br>'
        '<b>模型定位</b>　模型输出违约概率，替代"按信用等级一刀切"的规则，'
        '把风险排序做得更细，再按阈值落到具体审批动作。'
        '</div></div>', unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3)
    kpi(r1c1, '累计放款样本', '800,000 笔', '2007—2018 年 · 47 个字段')
    kpi(r1c2, '平均单笔放款额', '¥%s' % format(AVG_LOAN, ',.0f'),
        '月供均值 ¥%s' % format(AVG_INST, ',.0f'))
    kpi(r1c3, '平均年利率', '%.2f%%' % AVG_RATE, '期限以 36 / 60 期为主')

    r2c1, r2c2, r2c3 = st.columns(3)
    kpi(r2c1, '整体违约率', '19.95%', '正负样本约 4 : 1')
    kpi(r2c2, '模型区分度 AUC', '%.4f' % FINAL['测试 auc'], '最优配置 · seed 42', alt=True)
    kpi(r2c3, '业务最优阈值', '%.2f' % FINAL['业务最优阈值'], '加权成本最小点', alt=True)

    st.markdown(
        '<div class="note" style="margin:-4px 0 14px">'
        '<b>AUC 是什么：</b>随机抓 100 个会违约的客户和 100 个不会违约的客户，'
        '模型能把会违约的那批排到前面的程度，取 100 次的平均值就是 AUC。'
        '本项目 AUC %.4f 意味着约 <b>%.0f%% 的时候能把坏客户排在好客户前面</b>。<br>'
        '<b>为什么不用准确率：</b>这份数据好客户占 80%%，'
        '模型把所有人都判成"正常"就有 80%% 准确率——看着很高，但一个坏客户也没抓住。'
        '准确率会被这种"多数类"淹没，AUC 衡量的是'
        '<b>好坏两类能不能分开排序</b>，这才是审批场景真正需要的。</div>'
        % (FINAL['测试 auc'], FINAL['测试 auc'] * 100),
        unsafe_allow_html=True)

    st.markdown('')
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        t = TREND.copy()
        t['ym'] = pd.to_datetime(t['ym'])
        m = t[t['放款笔数'] >= 1000]
        fig = go.Figure()
        fig.add_bar(x=m['ym'], y=m['放款笔数'], name='放款笔数',
                    marker_color=LIGHT, marker_line=dict(color=MAIN, width=0.6))
        fig.add_scatter(x=m['ym'], y=m['违约率'] * 100, name='当月违约率',
                        yaxis='y2', mode='lines+markers',
                        line=dict(color=ALT, width=2.2),
                        marker=dict(size=4, color=ALT))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right',
                                      title='违约率 (%)', gridcolor='rgba(0,0,0,0)',
                                      tickfont=dict(color=ALT)),
                          yaxis=dict(title='放款笔数'),
                          xaxis=dict(title='放款月份'))
        style(fig, 330)
        head(st, '放款量的起伏与当月违约率', '03_monthly_trend.csv')
        st.plotly_chart(fig, width='stretch')
        st.markdown('<div class="note">横轴是放款月份、柱子是当月放款笔数、'
                    '橙线是当月违约率。可以看到放款量在 2007—2018 年间'
                    '先升后降（2018 年样本明显变少），而违约率逐月起伏、'
                    '没有单调趋势——<b>业务量的起伏不影响模型的判断逻辑</b>，'
                    '但也意味着不能只看某一个月的数据下结论，'
                    '所以下面要按时间把数据切成两套来验证。</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['正常', '违约'], values=TARGET['样本数'], hole=0.62,
            marker=dict(colors=[MAIN, ALT], line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent', textfont=dict(size=13, color='#FFFFFF')))
        fig.update_layout(showlegend=False, annotations=[
            dict(text='<b>80.05%</b><br><span style="font-size:12.5px;color:#7A8090">'
                 '正常</span>', x=0.5, y=0.5, showarrow=False,
                 font=dict(size=22, color=INK))])
        style(fig, 330)
        head(st, '违约与正常样本的比例', '01_target_distribution.csv')
        st.plotly_chart(fig, width='stretch')
        st.markdown('<div class="note">模型要学的就是这个 19.95%——'
                    '把每一笔申请判成"会违约 / 不会违约"。<br>'
                    '<b>注意正负比例接近 4 : 1，不是 1 : 1。</b>'
                    '所以哪怕模型什么都不做、全判"正常"，准确率也有 80%，'
                    '看起来很高却毫无用处——这也是本项目用 AUC 而不是准确率'
                    '作为主指标的原因。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- 数据集概况 ----------------
    st.markdown('## 数据集概况')
    n_field = len(COVER)
    n_num = int((COVER['dtype'].isin(['float64', 'int64'])).sum())
    n_cat = n_field - n_num
    miss_df = COVER[COVER['缺失数'] > 0]
    n_miss = len(miss_df)
    # 成批缺失 = 缺失量超过 100 笔的字段（零星 1 笔的不算）
    n_batch = int((miss_df['缺失数'] > 100).sum())
    n_sparse = n_miss - n_batch
    # 成批缺失那一组的行数：40270/46799/69752…不同批次，取出现最多的那一档
    _batch = miss_df.loc[miss_df['缺失数'] > 100, '缺失数']
    batch_n = int(_batch.value_counts().idxmax())
    n_clean = n_field - n_miss
    st.markdown(
        '<div class="note">在看模型之前，先说清数据本身的样子。'
        '原始文件共 <b>%d 个字段</b>：<b>%d 个数值型</b>（金额、期限、利率、'
        '收入、信用分等）和 <b>%d 个类别型</b>（信用等级、住房状况、贷款用途等），'
        '每行是一笔已放款的申请，共 80 万行。<br>'
        '<b>缺失集中在两类字段</b>：%d 个字段有缺失，其中 <b>%d 个是成批缺失</b>'
        '（同一批申请一起缺，量级在 %s 笔上下），'
        '另外 %d 个字段只有零星几笔缺失。'
        '成批缺失说明这些字段来自同一次信息采集流程，'
        '本身可以作为"资料是否齐全"的信号使用。'
        '</div>' % (n_field, n_num, n_cat, n_miss, n_batch,
                    format(batch_n, ','), n_sparse),
        unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        head(st, '各字段缺失率排名（Top 12）', '01_column_overview.csv')
        mp = COVER[COVER['缺失数'] > 0].sort_values('缺失率(%)', ascending=False)
        mp = mp.head(12).iloc[::-1]
        lbl = [BIZ_NAME.get(x, x) if x in BIZ_NAME else x for x in mp['列名']]
        fig = go.Figure(go.Bar(x=mp['缺失率(%)'], y=lbl, orientation='h',
                               marker_color=MAIN,
                               text=['%.2f%%' % x for x in mp['缺失率(%)']],
                               textposition='outside',
                               textfont=dict(size=10, color=INK)))
        fig.update_layout(xaxis=dict(title='该字段缺失的申请占比 (%)',
                                     range=[0, float(mp['缺失率(%)'].max()) * 1.28]),
                          yaxis=dict(tickfont=dict(size=10), automargin=True))
        style(fig, 380)
        st.plotly_chart(fig, width='stretch')
        _worst = mp.iloc[-1]
        st.markdown('<div class="note"><b>缺失率最高的是 %s（%.2f%%）</b>，'
                    '属于原始数据里的<b>匿名字段</b>——数据提供方没给业务命名，'
                    '只能按取值分布参与建模。<br>'
                    '有业务含义的字段里，<b>employmentLength（在职年限）'
                    '缺失 %.2f%%</b> 最值得注意，它是信息填不全的典型；'
                    '而放款额、期限、利率、收入这些核心字段基本没有缺失。<br>'
                    '缺失不直接影响建模——预处理阶段按训练集的中位数填补，'
                    '测试集不会泄漏信息。</div>'
                    % (_worst['列名'], float(_worst['缺失率(%)']),
                       float(COVER.loc[COVER['列名'] == 'employmentLength',
                                       '缺失率(%)'].iloc[0])),
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        head(st, '缺失组与完整组的违约率差异', '02_missing_vs_default.csv')
        mv = MISS.copy().sort_values('差异(百分点)', ascending=False).iloc[::-1]
        fig = go.Figure()
        fig.add_bar(x=mv['差异(百分点)'], y=mv['分组'],
                    orientation='h', marker_color=MAIN,
                    text=['%+.2f' % x for x in mv['差异(百分点)']],
                    textposition='outside', textfont=dict(size=10, color=INK))
        fig.add_vline(x=0, line=dict(color=LINE, width=1.2))
        fig.update_layout(xaxis=dict(title='缺失组违约率 − 完整组违约率（百分点）',
                                     range=[float(mv['差异(百分点)'].min()) * 1.3,
                                            float(mv['差异(百分点)'].max()) * 1.3]),
                          yaxis=dict(tickfont=dict(size=9), automargin=True))
        style(fig, 380)
        st.plotly_chart(fig, width='stretch')
        emp = MISS[MISS['分组'] == 'employmentLength缺失'].iloc[0]
        st.markdown(
            '<div class="note">横条为正 = 缺这组的人违约更高，为负 = 缺这组的人'
            '违约反而更低。<b>最有业务含义的是在职年限缺失</b>：'
            '缺失组违约率 %.2f%%，比资料完整的人高 %.2f 个百分点——'
            '填不出工作信息，本身就是还款能力存疑的信号。<br>'
            '其余字段缺失多来自匿名字段，缺失组违约率还低于完整组，'
            '差异在 %.2f ~ %.2f 个百分点之间，'
            '<b>不构成稳定的风险信号</b>。</div>'
            % (float(emp['缺失组违约率(%)']), float(emp['差异(百分点)']),
               float(mv['差异(百分点)'].min()), float(mv['差异(百分点)'].max())),
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    head(st, '原始字段到模型特征的转换过程（%d 个字段 → %d 个特征）'
         % (n_field, N_FEATURE), '04_encoding_summary_random.csv')
    enc = ENC.copy()
    enc['说明'] = enc['方法'].map({
        '序数编码': '等级类字段（grade / subGrade）转成可比较大小的一列',
        '独热编码': '类别字段拆成"是不是某一类"的 0/1 列',
        '目标编码（K 折 OOF）': '把高基数字段按历史违约率折算成一个数值'})
    enc.columns = ['处理方法', '涉及字段', '拆出列数', '说明']
    st.dataframe(enc, width='stretch', hide_index=True)
    st.markdown(
        '<div class="note">原始文件 %d 列里，<b>1 列是样本编号、1 列是标签</b>'
        '（是否违约），真正进入建模的是 %d 个业务字段；'
        '经过缺失处理与编码后共 <b>%d 个特征</b>。'
        '其中高基数类别（职位、邮编、地区这类取值过万的字段）'
        '不直接拆成 0/1 列——那会造出几千列稀疏数据，'
        '而是按历史违约率折算成一个数值；这一步在训练集内部用 K 折交叉的方式算，'
        '避免把标签信息泄漏到特征里。<br>'
        '<b>这 %d 个特征就是后面「特征解释」页面要逐个翻译成业务语言的清单。</b>'
        '</div>' % (n_field, n_field - 2, N_FEATURE, N_FEATURE),
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- 数据怎么分 ----------------
    st.markdown('## 验证方案：两种切分方式')
    st.markdown(
        '<div class="note" style="margin-bottom:10px">'
        '本项目做了两套验证方式，结论互相印证：<br>'
        '<b>① 随机切分</b> —— 把 80 万笔<b>打乱了顺序</b>再按 70/20/10 分成三份。'
        '三份数据来自同一个时期，因此违约率几乎一样'
        '（都约 %.2f%%）。回答的问题是：<b>同一时期内模型能不能泛化</b>。<br>'
        '<b>② 时间外推（OOT）</b> —— 用 <b>2013~2016 年</b>的数据训练，'
        '拿 <b>2017~2018 年</b>的数据做测试，相当于"用过去预测未来"。'
        '这是更严格的做法，回答的问题是：<b>模型放到未来还用不用得上</b>。</div>'
        % float(SPLIT[SPLIT['tag'] == 'random'][['train_def', 'valid_def',
                                                  'test_def']].iloc[0].mean()),
        unsafe_allow_html=True)
    sp = SPLIT[['tag', 'n_features', 'train_n', 'valid_n', 'test_n',
                'train_def', 'valid_def', 'test_def']].copy()
    sp['方案'] = sp['tag'].map({
        'random': '① 随机切分（打乱后 70/20/10，同一时期）',
        'oot_y2013': '② 时间外推（2013~2016 训练 → 2017~2018 测试）'})
    sp['违约率 (%)'] = sp['tag'].map({
        'random': '%.2f / %.2f / %.2f' % tuple(
            SPLIT[SPLIT['tag'] == 'random'][['train_def', 'valid_def',
                                             'test_def']].iloc[0]),
        'oot_y2013': '%.2f / %.2f / %.2f' % tuple(
            SPLIT[SPLIT['tag'] == 'oot_y2013'][['train_def', 'valid_def',
                                                'test_def']].iloc[0])})
    sp = sp[['方案', 'n_features', 'train_n', 'valid_n', 'test_n', '违约率 (%)']].copy()
    sp.columns = ['验证方式', '使用特征数', '训练笔数', '验证笔数', '测试笔数',
                  '三份数据的违约率 (%)']
    head(st, '两种切分方式的具体划分', '04_split_stats.csv')
    st.dataframe(sp.style.format({'使用特征数': '{:,.0f}', '训练笔数': '{:,}',
                                  '验证笔数': '{:,}', '测试笔数': '{:,}'}),
                 width='stretch', hide_index=True)
    _rnd = SPLIT[SPLIT['tag'] == 'random'].iloc[0]
    _oot = SPLIT[SPLIT['tag'] == 'oot_y2013'].iloc[0]
    st.markdown(
        '<div class="note">'
        '<b>为什么两行的训练集违约率不一样（%.4f%% vs %.4f%%）？</b>'
        '不是谁算错了，而是两个训练集覆盖的<b>年份不同</b>。'
        '随机切分从 2007—2018 全时段抽样，违约率就是全样本的 %.2f%%；'
        '时间外推只用 <b>2013—2016 年</b>这四年，而这四年恰好是违约率较低的时段'
        '（从"放款规模与违约率走势"图能看到 2013—2014 年违约率明显低于全期）。'
        '所以训练集违约率低 %.2f 个百分点，'
        '<b>是时间窗带来的，不是处理错误</b>。<br>'
        '<b>最终以哪个为准？</b>以 <b>随机切分</b>（%s）为模型选型、阈值决策、'
        '风险分层的主口径——它三份数据同分布，训练/验证/测试可直接对比，'
        '成绩高低只反映模型好坏，不受年份干扰。'
        '时间外推作为<b>稳健性检验</b>放在「稳定性与校准」页，'
        '只回答"结论放到未来还成不成立"，不参与选型。<br>'
        '需要留意：时间外推那行的<b>验证集违约率 %.2f%% 明显偏高</b>，'
        '原因是验证段正好落在违约率上行区间（2016 年前后），'
        '属于时序切分的固有特征，不影响测试集结论。</div>'
        % (_rnd['train_def'], _oot['train_def'], float(_rnd['train_def']),
           float(_rnd['train_def']) - float(_oot['train_def']),
           '训练 %.0f%% / 验证 %.0f%% / 测试 %.0f%%'
           % (_rnd['train_n'] / 8e5 * 100, _rnd['valid_n'] / 8e5 * 100,
              _rnd['test_n'] / 8e5 * 100),
           float(_oot['valid_def'])),
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- 业务风险画像：从全量样本看真实风险规律 ----
    st.markdown('## 业务侧已知风险规律')
    st.markdown(
        '<div class="note">下面按<b>单个字段</b>和<b>字段组合</b>两种口径，'
        '统计各群体在全量 80 万笔样本上的实际违约率，'
        '得到的是业务经验里已经成立的风险排序。'
        '模型要做的就是在同一批人身上给出<b>更细、更准</b>的排序——'
        '如果模型判定的高风险人群与这里的高风险客群完全对不上，'
        '即使分数再高也不能上线。</div>', unsafe_allow_html=True)

    def profile_theme(theme):
        d = PROFILE[PROFILE['分析主题'] == theme].copy()
        d['违约率'] = d['违约率%']
        return d

    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        head(st, '信号一：定价（利率）', '01b_combination_profile.csv · 利率分档')
        d = profile_theme('利率分档')
        fig = go.Figure(go.Bar(
            x=d['分组'], y=d['违约率'], marker_color=MAIN,
            text=['%.1f%%' % x for x in d['违约率']],
            textposition='outside', textfont=dict(size=10, color=INK)))
        fig.add_hline(y=OVERALL_DEF * 100, line=dict(color=ALT, width=1.4, dash='dash'),
                      annotation_text='整体 %.2f%%' % (OVERALL_DEF * 100),
                      annotation_font=dict(size=10, color=ALT))
        fig.update_layout(xaxis=dict(title='年化利率分档'),
                          yaxis=dict(title='违约率 (%)', range=[0, 50]))
        style(fig, 300)
        st.plotly_chart(fig, width='stretch')
        rt_hi = float(d[d['分组'] == '>20%']['违约率'].iloc[0])
        rt_lo = float(d[d['分组'] == '≤8%']['违约率'].iloc[0])
        st.markdown(
            '<div class="note">利率 ≤8%% 的客户违约率只有 %.1f%%，'
            '利率 &gt;20%% 的客户违约率 %.1f%%，'
            '相差 <b>%.1f 倍</b>。<br>利率是这笔业务<b>自己定的价格</b>，'
            '定得越高、客户还款压力越大，所以它既是收入来源也是风险来源。'
            '这就是"定价风险"——<b>不能只看客户资质，还要看自己给的价格</b>。'
            '实务上这条规律会被用来做两件事：高风险客群收紧利率上限、'
            '低风险客群适当降价换取更大利差。</div>'
            % (rt_lo, rt_hi, rt_hi / rt_lo), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        head(st, '信号二：客户资质（信用等级）', '03_categorical_default_rate.csv · grade')
        g = CAT[CAT['字段'] == 'grade'].sort_values('违约率', ascending=False)
        fig = go.Figure()
        fig.add_bar(x=g['取值'], y=g['样本占比%'], name='资产占比',
                    marker_color=LIGHT, marker_line=dict(color=MAIN, width=0.5))
        fig.add_scatter(x=g['取值'], y=g['违约率%'], name='违约率',
                        yaxis='y2', mode='lines+markers',
                        line=dict(color=ALT, width=2.4),
                        marker=dict(size=6, color=ALT))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right',
                                      title='违约率 (%)', range=[0, 55],
                                      tickfont=dict(color=ALT)),
                          yaxis=dict(title='资产占比 (%)'),
                          xaxis=dict(title='信用等级（由好到差 A→G）'))
        style(fig, 300)
        st.plotly_chart(fig, width='stretch')
        gd = g[g['取值'].isin(['D', 'E', 'F', 'G'])]
        # 违约笔数贡献占比 = 该组违约笔数 / 全量违约笔数（实算，不用占比相乘近似）
        n_def_all = float((g['样本数'] * g['违约率']).sum())
        gd_share = float((gd['样本数'] * gd['违约率']).sum()) / n_def_all * 100
        st.markdown(
            '<div class="note">D~G 四个等级只占资产的 <b>%.1f%%</b>，'
            '却贡献了 <b>%.1f%%</b> 的违约笔数。'
            '换句话说，<b>风险高度集中在少数低资质客户身上</b>——'
            '"按信用等级一刀切"的规则方向是对的，'
            '但它只分了 7 档，粗得没法区分同档内部的差异。'
            '这就是模型存在的价值：在等级之内再细分。</div>'
            % (gd['样本占比%'].sum(), gd_share),
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    head(st, '信号三：三个条件同时踩中的极端客群', '01b_combination_profile.csv · 多字段组合')
    d = profile_theme('多字段组合')
    st.markdown(
        '<div class="note">前两张图是"一个条件"的影响，违约率会被其他因素稀释。'
        '这里把<b>三个条件同时成立</b>的一小群人单独圈出来看——'
        '所谓"<b>极端</b>"，指的是<b>同时命中多个不利条件</b>的客群，'
        '不是单看某一项特别差。'
        '圈这类人有两个用处：一是评估最坏情况下要准备多少损失资金，'
        '二是这类组合条件足够明确，可以直接写成硬规则进审批系统。</div>',
        unsafe_allow_html=True)
    cols = st.columns(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        tag = ' alt' if r['违约率'] > OVERALL_DEF * 100 else ''
        cols[i].markdown(
            '<div class="kpi%s"><div class="lab">%s</div>'
            '<div class="val">%.1f%%</div>'
            '<div class="sub">%s 笔 · 占 %.2f%%</div></div>'
            % (tag, r['分组'], r['违约率'], format(int(r['样本数']), ','),
               r['样本占比%']), unsafe_allow_html=True)
    st.markdown('<div class="note">高风险组合（利率&gt;15%% + 5 年期 + FICO&lt;680）'
                '违约率 %.1f%%，是低风险组合（利率≤11%% + 3 年期 + FICO≥720）'
                '的 <b>%.1f 倍</b>。这类"硬规则"可直接写进审批策略，与模型概率形成双保险。'
                '</div>' % (d.iloc[0]['违约率'], d.iloc[0]['违约率'] / d.iloc[1]['违约率']),
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif PAGE == '模型选型':
    st.markdown('## 模型选型')

    # ---- 四组配置横评（先算，后面文案全部引用这里） ----
    cfg_map = CFG_MAP
    sel = _PEAK_SEL.copy()
    order = ['未调优基线', '候选 1 · 弱正则', '候选 2 · 中正则', '最优配置 ★']
    g = sel.groupby('显示名').agg(
        测试AUC均值=('测试 auc', 'mean'),
        测试AUC最低=('测试 auc', 'min'),
        测试AUC最高=('测试 auc', 'max'),
        KS均值=('测试 ks', 'mean'),
        Brier均值=('测试 brier', 'mean'),
        召回率均值=('测试 recall', 'mean'),
        种子数=('随机种子', 'count')).reset_index()
    g['跨种子波动'] = g['测试AUC最高'] - g['测试AUC最低']
    g = g.set_index('显示名').loc[[o for o in order if o in set(g['显示名'])]].reset_index()
    pick = g[g['显示名'] == '最优配置 ★'].iloc[0]
    base = g[g['显示名'] == '未调优基线'].iloc[0]
    best_auc = float(g['测试AUC均值'].max())
    base_peak = sorted(set(sel[sel['配置'].str.startswith('基线')]['峰值轮'].astype(int)))
    base_peak_txt = ('%d' % base_peak[0]) if len(base_peak) == 1 \
        else ('%d ~ %d' % (base_peak[0], base_peak[-1]))
    tuned = g[~g['显示名'].str.startswith('未调优')]
    n_groups = N_GROUPS
    # 最优配置的实际超参：一律从 101 表「是否为最终取值」行取，不写死
    _fin_ps = PSUM[PSUM['是否为最终取值'] == '是']
    _pv = dict(zip(_fin_ps['参数'], _fin_ps['取值']))

    st.markdown(
        '<div class="note" style="margin-bottom:14px">'
        '<b>为什么最后选了这套参数？</b><br>'
        '调优一共跑了 <b>%d 组</b>实验，'
        '四组候选配置在测试集上的平均 AUC 相差仅 <b>%.4f</b>'
        '（区间 %.4f ~ %.4f），单看准确率几乎分不出高下。<br>'
        '选型依据是两条：<br>'
        '①<b>平均 AUC 最高</b>——3 个随机种子的平均值最高，'
        '不是某一次跑分碰巧高；<br>'
        '②<b>跨种子波动可控</b>——极差 %.4f，换随机数后结果不会大幅漂移。<br>'
        '另外必须提一句：未调优基线早停在 %s 轮、迭代极少，'
        '三个种子跑出<b>完全相同</b>的结果，波动为 %.4f；'
        '但它的平均 AUC 最低（%.4f）。<br>'
        '<b>波动小不等于好，要和准确率一起看。</b></div>'
        % (n_groups, best_auc - float(g['测试AUC均值'].min()),
           float(g['测试AUC均值'].min()), best_auc,
           pick['跨种子波动'], base_peak_txt,
           base['跨种子波动'], base['测试AUC均值']),
        unsafe_allow_html=True)

    takeaway('四组候选配置跑满 %d 组实验，'
             '平均 AUC 只差 %.4f，<mark>真正分出高下的是「稳不稳」而不是「高多少」</mark>，'
             '所以最终取平均 AUC 最高、跨种子极差 %.4f 的那一组。'
             % (n_groups, best_auc - float(g['测试AUC均值'].min()),
                pick['跨种子波动']))

    a1, a2, a3, a4 = st.columns(4)
    kpi(a1, '最终选定配置', '树深 %g · 步长 %g' % (_pv['树深'], _pv['学习率']),
        '正则 λ%g · 采样 %g（完整取值见下表）' % (_pv['L2正则'], _pv['样本采样比']),
        alt=True)
    kpi(a2, '四组中平均 AUC 最高', '%.6f' % best_auc, '3 个种子均值')
    kpi(a3, '相对未调优基线提升', '%+.4f' % (pick['测试AUC均值'] - base['测试AUC均值']),
        '基线 %.4f' % base['测试AUC均值'])
    kpi(a4, '调优实验总组数', '%d 组' % n_groups,
        '含特征、消融、正则等多类实验')

    st.markdown('')
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    head(st, '最终选定参数一览', '101_param_sensitivity_summary.csv · 97_peak_round_choice.csv')
    _order = ['树深', '学习率', '样本采样比', '列采样比', 'L2正则',
              '最小子样本权重', '迭代上限']
    prow = pd.DataFrame({
        '参数': _order,
        '业务含义': [_MEAN[k] for k in _order],
        '最终取值': ['%g' % _pv[k] if k != '迭代上限'
                     else '%d' % int(FINAL['迭代上限']) for k in _order]})
    st.dataframe(prow, width='stretch', hide_index=True)
    st.markdown('<div class="note">这套取值属于<b>中等偏保守</b>的一档：'
                '树浅、步长小、L2 正则偏强。方向上说明这份数据上'
                '<b>模型越激进越容易过拟合</b>——80 万样本、'
                '89 个特征、信噪比有限，复杂的拟合结构没有回报。'
                'seed 42 实际训练到第 %d 轮达峰（上限 %d 轮）。</div>'
                % (int(FINAL['峰值轮']), int(FINAL['迭代上限'])),
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('')
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        head(st, '四组配置逐项对比（3 个随机种子均值）', '97_peak_round_choice.csv（random 切分，全特征，3 seed 均值）')
        fig = go.Figure()
        metrics = [('测试AUC均值', '测试 AUC', MAIN), ('KS均值', 'KS', ALT),
                   ('召回率均值', '召回率', GREY)]
        for col, lab, colr in metrics:
            fig.add_bar(x=g['显示名'], y=g[col], name=lab, marker_color=colr,
                        text=['%.4f' % x for x in g[col]], textposition='outside',
                        textfont=dict(size=10, color=INK))
        fig.update_layout(barmode='group', yaxis=dict(title='指标值', range=[0, 0.85]),
                          xaxis=dict(tickfont=dict(size=10)),
                          legend=dict(orientation='h', y=1.0, yanchor='bottom',
                                      x=1, xanchor='right', font=dict(size=10)))
        style(fig, 360)
        st.plotly_chart(fig, width='stretch')
    with r:
        head(st, '跨随机种子波动（越小越稳）', '97_peak_round_choice.csv（random 切分，全特征，3 seed 极差）')
        fig = go.Figure()
        colrs = [ALT if x == '最优配置 ★' else LIGHT for x in g['显示名']]
        fig.add_bar(x=g['显示名'], y=g['跨种子波动'], marker_color=colrs,
                    text=['%.4f' % x for x in g['跨种子波动']],
                    textposition='outside', textfont=dict(size=10, color=INK))
        fig.update_layout(yaxis=dict(title='AUC 极差（最高−最低）'),
                          xaxis=dict(tickfont=dict(size=10)))
        style(fig, 360)
        st.plotly_chart(fig, width='stretch')
    st.markdown('<div class="note">'
                '左图看"谁的分数高"，右图看"这个分数稳不稳"。'
                '四组配置的平均测试 AUC 落在 %.4f ~ %.4f 之间，'
                '跨度 %.4f，与随机种子本身带来的波动同量级，'
                '<b>所以不能只看某一次跑分</b>。<br>'
                '最优配置的 3 种子平均 AUC 最高（%.6f），'
                '跨种子极差 %.4f，两项都占优；'
                '未调优基线虽然波动为 0（三个种子跑出完全相同的数字），'
                '但那是因为它早停在 %s 轮、迭代太少，'
                '代价是平均 AUC 只有 %.6f，落后最优配置 %.4f。'
                '<b>波动小不等于好，必须和准确率一起看。</b></div>'
                % (float(g['测试AUC均值'].min()), best_auc,
                   best_auc - float(g['测试AUC均值'].min()),
                   pick['测试AUC均值'], pick['跨种子波动'],
                   base_peak_txt, base['测试AUC均值'],
                   pick['测试AUC均值'] - base['测试AUC均值']),
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- 单参数敏感性（逐参数展开） ----
    # 「迭代上限」在 101 表里没有「是否为最终取值」标记（该值由 17 号 JSON 记录），
    # 为保证每个参数页都能算出"当前取值排名"，此处只纳入带标记的 7 个参数，
    # 敏感度排序与结论都限定在这7 个之内，不与迭代上限混算。
    keep = ['树深', '学习率', '样本采样比', '列采样比', 'L2正则',
            '最小子样本权重', '分裂损失阈值']
    ps = PSUM[PSUM['参数'].isin(keep)].copy()
    ps = ps.sort_values(['参数', '取值'])
    # 每个参数的「最佳取值与最终取值的差距」→ 衡量该参数的影响强度
    infl = []
    for nm, sub in ps.groupby('参数', sort=False):
        fin = sub[sub['是否为最终取值'] == '是']
        if not len(fin):
            continue
        fv = fin['取值'].iloc[0]
        fauc = fin['均值AUC'].iloc[0]
        spread = float(sub['均值AUC'].max() - sub['均值AUC'].min())
        top = sub.loc[sub['均值AUC'].idxmax()]
        infl.append({'参数': nm, '最终取值': fv, '取值数': len(sub),
                     '均值极差': spread, '最终取值排名': int(
                         (sub['均值AUC'] > fauc).sum() + 1),
                     '最优取值': top['取值'], '与最优差距': float(fauc - top['均值AUC'])})
    infl = pd.DataFrame(infl).sort_values('均值极差', ascending=False)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('## 换参数会怎样')
    st.markdown(
        '<div class="note" style="margin-bottom:10px">'
        '调优时对下面 %d 个参数逐个试了若干取值。'
        '这里按<b>取值敏感度</b>排序——该参数各个取值之间平均 AUC 的差距越大，'
        '说明它对结果影响越大。<br>'
        '<b>最敏感的是 %s（%.5f），最不敏感的是 %s（%.5f），'
        '两者相差不到 %.1f 倍。</b>'
        '全部 %d 个参数的极差都在万分之一量级，'
        '说明这份数据上<b>没有"某个参数必须怎么调"的关键点</b>。'
        '点下方标签页可以逐个参数看它试过的每个取值。</div>'
        % (len(infl),
           infl['参数'].iloc[0], float(infl['均值极差'].iloc[0]),
           infl['参数'].iloc[-1], float(infl['均值极差'].iloc[-1]),
           float(infl['均值极差'].iloc[0]) / float(infl['均值极差'].iloc[-1]),
           len(infl)),
        unsafe_allow_html=True)

    # ---- 敏感度总览（横条，横向排版） ----
    ov = infl.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=ov['均值极差'], y=ov['参数'], orientation='h',
        marker_color=[ALT if p == infl['参数'].iloc[0] else MAIN
                      for p in ov['参数']],
        text=['%.5f' % x for x in ov['均值极差']],
        textposition='outside', textfont=dict(size=10, color=INK)))
    fig.update_layout(xaxis=dict(title='该参数各取值之间的平均 AUC 极差（越大越敏感）',
                                 range=[0, float(infl['均值极差'].max()) * 1.16]),
                      yaxis=dict(tickfont=dict(size=11), automargin=True))
    style(fig, 320)
    head(st, '各参数的影响强度排序', '101_param_sensitivity_summary.csv')
    st.plotly_chart(fig, width='stretch')
    st.markdown('<div class="note">这条衡量的是"同一个参数换成不同取值，'
                '结果会差多少"。注意它<b>不是</b>单独调这个参数得到的实验结果，'
                '而是把该参数在不同实验组里出现过的取值汇总后横向比较，'
                '所以下面每个参数的结论都要配合"涉及实验组数"一起看。</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- 逐参数展开：每个参数一个标签页，页内自成一个小模块 ----
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    head(st, '单个参数各取值的成绩分布', '101_param_sensitivity_summary.csv')
    st.markdown(
        '<div class="note" style="margin-bottom:8px">'
        '每个标签页是一张"取值 → 平均验证 AUC"的折线图：'
        '蓝线是平均值，灰带是该取值下最好到最差的范围，'
        '橙色虚线标出当前最终取值的位置。'
        '横线越平缓，说明这个参数怎么调都差不多。</div>',
        unsafe_allow_html=True)
    _tab = st.tabs(list(infl['参数']))
    for _t, _i in zip(_tab, range(len(infl))):
        with _t:
            picked = infl.iloc[_i]['参数']
            sub = ps[ps['参数'] == picked].sort_values('取值')
            fin = sub[sub['是否为最终取值'] == '是']
            fv = fin['取值'].iloc[0] if len(fin) else sub['取值'].iloc[0]
            top_row = sub.loc[sub['均值AUC'].idxmax()]
            cur_auc = float(sub[sub['取值'] == fv]['均值AUC'].iloc[0])
            gap = cur_auc - float(top_row['均值AUC'])
            rank = int((sub['均值AUC'] > cur_auc).sum() + 1)
            span = float(sub['均值AUC'].max() - sub['均值AUC'].min())

            # 页内小标题：参数名 + 当前取值 + 影响强度
            p_meaning = _MEAN.get(picked, '')
            st.markdown(
                '<div style="display:flex;align-items:baseline;gap:10px;'
                'flex-wrap:wrap;margin:2px 0 6px">'
                '<span style="font-size:18px;font-weight:700;color:%s">%s</span>'
                '<span style="font-size:14.5px;color:#7A8090">%s</span>'
                '<span style="font-size:14.5px;color:%s">当前取值 <b>%g</b>'
                '（平均 AUC %.5f，%d 个取值中第 %d）</span>'
                '<span style="font-size:14px;color:#7C8494">'
                '取值敏感度 %.5f（第 %d / %d 位）</span></div>'
                % (INK, picked, p_meaning, MAIN, fv, cur_auc, len(sub), rank,
                   span, _i + 1, len(infl)),
                unsafe_allow_html=True)

            fig = go.Figure()
            xs = list(range(len(sub)))
            fig.add_scatter(x=xs, y=sub['最差AUC'], mode='lines',
                            line=dict(color=LIGHT, width=8), showlegend=False,
                            hoverinfo='skip')
            fig.add_scatter(x=xs, y=sub['最优AUC'], mode='lines',
                            line=dict(color=LIGHT, width=8), showlegend=False,
                            hoverinfo='skip')
            fig.add_scatter(x=xs, y=sub['均值AUC'], mode='lines+markers+text',
                            line=dict(color=MAIN, width=2.4),
                            marker=dict(size=8, color=MAIN),
                            text=['%g' % v for v in sub['取值']],
                            textposition='top center',
                            textfont=dict(size=10, color=INK), showlegend=False,
                            customdata=np.stack([sub['出现组数']], axis=-1),
                            hovertemplate='取值 %{text}<br>平均验证 AUC %{y:.5f}'
                                          '<br>涉及 %{customdata[0]} 组实验<extra></extra>')
            fx = list(sub['取值']).index(fv)
            fig.add_vline(x=fx, line=dict(color=ALT, width=1.8, dash='dash'),
                          annotation_text='当前取值 %g' % fv,
                          annotation_position='top',
                          annotation_font=dict(color=ALT, size=11))
            fig.update_xaxes(tickvals=xs, ticktext=['%g' % v for v in sub['取值']],
                             tickfont=dict(size=10), showgrid=False,
                             title_text='%s 的取值' % picked)
            _ymin = float(min(sub['最差AUC'].min(), sub['均值AUC'].min())) - 0.0006
            _ymax = float(max(sub['最优AUC'].max(), sub['均值AUC'].max())) + 0.0006
            fig.update_yaxes(range=[_ymin, _ymax], title='验证集平均 AUC')
            head(st, '%s：各取值的影响' % picked, '101_param_sensitivity_summary.csv')
            style(fig, 330)
            st.plotly_chart(fig, width='stretch')

            if abs(gap) < 1e-9:
                _pick_txt = ('当前取值 <b>%g</b> 正是该参数下的最优选择'
                             '（平均 AUC %.5f，%d 个取值里排第 1），不用调整。'
                             % (fv, cur_auc, len(sub)))
                if _i == 0:
                    _cmp_txt = ('它是<b>所有参数里最值得关注的旋钮</b>，'
                                '但即使换成该参数下的次优取值，'
                                '损失也在万分之一量级，'
                                '换过去没有实质收益，所以保持当前取值。')
                else:
                    _cmp_txt = ('但即使换成该参数下的次优取值，'
                                '损失也在万分之一量级，'
                                '换过去没有实质收益，所以保持当前取值。')
            else:
                _pick_txt = ('当前取值 <b>%g</b> 在 %d 个取值里排第 %d；'
                             '分数最高的是 <b>%g</b>（%.5f），高出 %.5f。'
                             % (fv, len(sub), rank, top_row['取值'],
                                float(top_row['均值AUC']), abs(gap)))
                _cmp_txt = ('这 %.5f 的差距<b>小于换随机种子带来的波动</b>'
                            '（四组配置跨种子极差最多 %.4f），'
                            '换过去换不来实质收益，<b>所以保持当前取值</b>。'
                            % (abs(gap), float(tuned['跨种子波动'].max())))
            st.markdown(
                '<div class="note">'
                '横轴是这个参数试过的 %d 个取值，纵轴是各取值下的'
                '<b>平均验证 AUC</b>。蓝线走平 = 调它影响不大，'
                '蓝线起伏大 = 它是个关键旋钮；灰带是同一取值在不同实验组里'
                '成绩的最好—最差范围，<b>带子越宽说明这个取值本身越不稳</b>；'
                '橙色竖虚线是最终选定的取值。<br>'
                '%s 该参数敏感度在 %d 个参数里排第 %d 位'
                '（极差 %.5f）。%s<br>'
                '上面的横向比较是把各参数在实验组里出现过的取值汇总后得到的，'
                '各取值参与的实验组数不同（%s），'
                '所以不能理解成"改成 %g 一定能提升 %.5f"。</div>'
                % (len(sub), _pick_txt, len(infl), _i + 1, span, _cmp_txt,
                   ' ~ '.join('%g（%d 组）' % (t, n) for t, n in
                              zip(sub['取值'], sub['出现组数'])),
                   top_row['取值'], abs(gap)),
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif PAGE == '模型表现':
    st.markdown('## 模型表现')
    st.markdown(
        '<div class="note" style="margin-bottom:14px">'
        '<b>这个模型的判别能力，用一句业务话讲就是：</b><br>'
        '随机抓 100 个会违约的客户和 100 个不会违约的客户，'
        '模型能把会违约的那批排在<b>后面的 %.0f%%</b>（AUC %.4f）。'
        '换成审批视角——按模型打分从高到低排，最前面那一段人群里的违约率'
        '是整体的 %.1f 倍（KS %.4f 表示好坏两段能拉开这么宽的差距）。<br>'
        '<b>为什么用 AUC 而不是准确率？</b><br>'
        '因为正负样本是 4 : 1，'
        '模型全判"正常"也有 80%% 准确率，看着高但没用。AUC 衡量的是'
        '"能不能把坏的和好的分开排"，这才是审批场景真正需要的。</div>'
        % (FINAL['测试 auc'] * 100, FINAL['测试 auc'],
           float(FINAL['首档 Lift']), KS5),
        unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, '调优后 测试 AUC', '%.4f' % CALIB.loc[1, '测试集 AUC'],
        '最优配置（合并重训口径）', alt=True)
    kpi(c2, '调优前 测试 AUC', '%.4f' % CALIB.loc[0, '测试集 AUC'],
        '未调优默认超参')
    kpi(c3, 'KS（调优前 → 后）', '%.4f → %.4f' % (KS4, KS5),
        '调优后 %.4f（略降）' % KS5 if KS5 < KS4 else '调优后 %.4f' % KS5)
    kpi(c4, 'Brier（调优后）', '%.5f' % CALIB.loc[1, 'Brier'], '越低越好')

    st.markdown('')
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        fig = go.Figure()
        for name, colr, dash in [(V4, LIGHT, 'dot'), (V5, MAIN, 'solid')]:
            g = CURVES[(CURVES['模型'] == name) & (CURVES['曲线'] == 'ROC')].sort_values('x')
            fig.add_scatter(x=g['x'], y=g['y'], name=name, mode='lines',
                            line=dict(color=colr, width=2.4 if name == V5 else 1.8,
                                      dash=dash),
                            fill='tozeroy' if name == V5 else None,
                            fillcolor='rgba(76,114,176,.08)')
        fig.add_scatter(x=[0, 1], y=[0, 1], name='随机', mode='lines',
                        line=dict(color=LINE, width=1.2, dash='dash'))
        fig.update_layout(xaxis=dict(title='假正率 FPR', range=[0, 1]),
                          yaxis=dict(title='真正率 TPR', range=[0, 1]))
        style(fig, 320)
        head(st, 'ROC 曲线', '50_curves_after_opt.csv')
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            '<div class="note">曲线越贴近左上角越好，'
            '线下面积就是 AUC。灰色虚线是"完全没有区分能力"的随机线。'
            '<br>两条线几乎叠在一起（AUC %.4f → %.4f，'
            '差 %+.4f），<b>调优带来的提升非常有限</b>。'
            '原因见上一页——各参数取值之间平均 AUC 差距只有万分之一量级，'
            '这份数据上已经找不到"再调一档就明显更好"的组合了。</div>'
            % (CALIB.loc[0, '测试集 AUC'], CALIB.loc[1, '测试集 AUC'],
               CALIB.loc[1, '测试集 AUC'] - CALIB.loc[0, '测试集 AUC']),
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        head(st, 'PR 曲线', '50_curves_after_opt.csv')
        fig = go.Figure()
        for name, colr, dash in [(V4, LIGHT, 'dot'), (V5, MAIN, 'solid')]:
            g = CURVES[(CURVES['模型'] == name) & (CURVES['曲线'] == 'PR')].sort_values('x')
            fig.add_scatter(x=g['x'], y=g['y'], name=name, mode='lines',
                            line=dict(color=colr, width=2.4 if name == V5 else 1.8,
                                      dash=dash),
                            fill='tozeroy' if name == V5 else None,
                            fillcolor='rgba(76,114,176,.08)')
        fig.update_layout(xaxis=dict(title='召回率 Recall', range=[0, 1]),
                          yaxis=dict(title='精确率 Precision'))
        style(fig, 320)
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            '<div class="note">横轴是"拒掉的客户里实际有多少是坏客户"'
            '召回的召回率，纵轴是"拒掉的人里真违约的占比"精确率。'
            '越靠左上越好。<br>AP %.4f → %.4f，同样几乎没变。'
            'PR 曲线整体比 ROC 低很多，是因为正负样本只有 4 : 1，'
            '基线精度天然只有 20%%——'
            '<b>所以审批策略应该定阈值而不是一刀切</b>，'
            '下一页「阈值决策」就是回答"拒到多少人为止"。</div>'
            % (curve_tag(V4, 'PR'), curve_tag(V5, 'PR')),
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        # 第4阶段用中灰虚线并加宽，避免与第5阶段的实线重合后看不见
        for curve, lab, cl in [('KS_违约', '违约客户累计占比', MAIN),
                               ('KS_正常', '正常客户累计占比', ALT)]:
            for name, colr, dash, wd in [(V4, GREY, 'dash', 2.4),
                                         (V5, cl, 'solid', 2.6)]:
                g = CURVES[(CURVES['模型'] == name) & (CURVES['曲线'] == curve)].sort_values('x')
                fig.add_scatter(x=g['x'], y=g['y'], mode='lines',
                                name='%s · %s' % ('未调优基线' if name == V4
                                                  else '最优配置', lab),
                                line=dict(color=colr, width=wd, dash=dash))
        fig.add_annotation(x=0.5, y=0.02, xref='paper', yref='paper',
                           text='两条线在中部岔得最开，这个最大间距就是 KS = %.4f'
                                % KS5, showarrow=False, xanchor='center',
                           font=dict(size=11, color=INK))
        fig.update_layout(xaxis=dict(title='按模型打分从低到高排列，'
                                          '横轴是这批人从第 0%% 排到第 100%%'),
                          yaxis=dict(title='累计占比'))
        style(fig, 330)
        head(st, 'KS 曲线', '50_curves_after_opt.csv（KS_违约 / KS_正常）')
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            '<div class="note">橙线是违约客户按打分的累计分布，'
            '蓝线是正常客户的。两条线<b>岔得越开</b>，说明模型越能'
            '把坏客户排前面；<b>两条线的最大垂直间距就是 KS 值</b>（%.4f）。'
            '<br>换句话说，把打分最高的那一段划出来，'
            '这段人群里违约率会明显高于整体——'
            '风控可以据此只对这批人加严审批。KS 越低说明越要靠后段，'
            '实际价值有限。</div>' % KS5, unsafe_allow_html=True)
    with r:
        cv = CV_R.iloc[0]
        folds = [float(x) for x in str(cv['fold_auc']).split(';')]
        fig = go.Figure(go.Bar(x=['第1折', '第2折', '第3折', '第4折', '第5折'],
                               y=folds, marker_color=MAIN,
                               text=['%.4f' % v for v in folds],
                               textposition='outside',
                               textfont=dict(size=11, color=INK)))
        fig.add_hline(y=float(cv['pooled_auc']), line=dict(color=ALT, width=1.6, dash='dash'),
                      annotation_text='池化 %.4f' % float(cv['pooled_auc']),
                      annotation_font=dict(color=ALT, size=11))
        fig.update_yaxes(range=[min(folds) - 0.0025, max(folds) + 0.0025],
                         title='折内 AUC')
        head(st, '五折交叉验证的折内成绩',
             '99_cv_final_c_random.csv / 99_cv_final_c_oot_y2013.csv')
        style(fig, 330)
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            '<div class="note"><b>为什么还要做交叉验证？</b>'
            '测试集只有一份成绩，可能是这批数据刚好运气好。'
            '五折交叉验证把训练集再切成五份轮流验证，'
            '<b>相当于用五份数据各考一次</b>。<br>'
            '五折成绩 %.4f ~ %.4f，极差仅 %.4f，'
            '折间标准差 %.4f——<b>说明成绩稳定，不是碰巧</b>。'
            '这也从另一个角度印证了上一页"四组配置差距很小"的判断：'
            '这份数据上换种子、换折、换参数，成绩都在同一水平。</div>'
            % (min(folds), max(folds), max(folds) - min(folds), float(cv['fold_std'])),
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    head(st, '调优前后指标对照（测试集）',
         '43_calibration_after_opt.csv · 44_psi_after_opt.csv · '
         '50_curves_after_opt.csv（KS 取曲线标注值）')
    df = pd.DataFrame({
        '指标': ['AUC', 'KS', 'LogLoss', 'Brier', 'ECE（测试集）', 'MCE（测试集）',
                 'PSI（验证对测试）'],
        '业务含义': ['区分好坏客户的能力，越高越好',
                     '好坏两段能拉开的最大宽度，越高越好',
                     '预测概率的偏离程度，越低越好',
                     '概率数字本身的准确程度，越低越好',
                     '模型说 20% 的一批人实际违约率与 20% 的平均差，越低越好',
                     '最差的那个分箱差多少，越低越好',
                     '新批次人群与训练时长得有多不像，越低越好'],
        '未调优默认超参': ['%.4f' % CALIB.loc[0, '测试集 AUC'], '%.4f' % KS4,
                     '%.5f' % CALIB.loc[0, 'LogLoss'],
                     '%.5f' % CALIB.loc[0, 'Brier'],
                     '%.5f' % CALIB.loc[0, 'ECE_测试集'],
                     '%.5f' % CALIB.loc[0, 'MCE_测试集'],
                     '%.5f' % float(PSI.loc[0, 'PSI_验证集对测试集'])],
        '最优配置': ['%.4f' % CALIB.loc[1, '测试集 AUC'], '%.4f' % KS5,
                  '%.5f' % CALIB.loc[1, 'LogLoss'],
                  '%.5f' % CALIB.loc[1, 'Brier'],
                  '%.5f' % CALIB.loc[1, 'ECE_测试集'],
                  '%.5f' % CALIB.loc[1, 'MCE_测试集'],
                  '%.5f' % float(PSI.loc[1, 'PSI_验证集对测试集'])],
    })
    st.dataframe(df, width='stretch', hide_index=True)
    _d_auc = CALIB.loc[1, '测试集 AUC'] - CALIB.loc[0, '测试集 AUC']
    _d_ks = KS5 - KS4
    _d_ll = CALIB.loc[1, 'LogLoss'] - CALIB.loc[0, 'LogLoss']
    _d_br = CALIB.loc[1, 'Brier'] - CALIB.loc[0, 'Brier']
    _d_ece = CALIB.loc[1, 'ECE_测试集'] - CALIB.loc[0, 'ECE_测试集']
    _d_mce = CALIB.loc[1, 'MCE_测试集'] - CALIB.loc[0, 'MCE_测试集']
    # 逐项判定方向，不靠嘴说"全部改善"：AUC/KS 越高越好，其余越低越好
    _judge = [('AUC', _d_auc, _d_auc > 0),
              ('KS', _d_ks, _d_ks > 0),
              ('LogLoss', _d_ll, _d_ll < 0),
              ('Brier', _d_br, _d_br < 0),
              ('ECE', _d_ece, _d_ece < 0),
              ('MCE', _d_mce, _d_mce < 0)]
    _better = [n for n, d, ok in _judge if ok]
    _worse = [n for n, d, ok in _judge if not ok]
    st.markdown(
        '<div class="note"><b>这些数字够不够上线用？</b><br>'
        '<b>① 主指标 AUC %.4f。</b>随机抓 100 个违约客户和 100 个正常客户，'
        '模型平均能把违约的那批排到 %.0f%% 位置——'
        '意味着每 100 笔申请里，'
        '<b>模型能挑出的高风险那一段里，坏客户密度是随机的 2.5 倍左右</b>'
        '（对应 KS %.4f、首档相对倍数 %.2f）。'
        '换个角度看更直观：<b>人工只审打分最高的 30%% 的人，就能拦住 %.1f%% 的违约客户</b>'
        '（详见「风险分层」页）；'
        '剩下 70%% 的人违约率明显低于整体，可以走自动化。'
        '<b>这是一个"能配合人工审批用"的水平，而不是"能替人审批"的水平</b>——'
        '把它当成排序工具是合适的，当成自动决策依据还差一截。<br>'
        '<b>② 六项指标里 %d 项改善、%d 项变差。</b>改善的是 %s；变差的是 %s。'
        '变差的三项幅度都在千分之一以内'
        '（KS %+.4f、ECE %+.5f、MCE %+.5f），'
        '而同一模型换随机种子重训本身就会带来这个量级的抖动，'
        '<b>所以不能据此说它变差了</b>。<br>'
        '<b>③ 那为什么最后选它？</b>不是因为某一项指标最高，'
        '而是因为<b>它没有短板</b>：'
        '四项概率类指标（LogLoss / Brier / ECE / MCE）全部处在未调优基线附近或更好，'
        '说明这套取值在"分数准不准"上也没有付出代价；'
        '同时跨随机种子波动 %.4f，说明换一批数据重训结果不会漂。'
        '对要长期跑在生产里的风控模型，'
        '<b>这种"各项都稳、无明显退化"的配置，比单项多 0.002 更重要</b>。<br>'
        '<b>④ 这两处数字不能混着比。</b>本页 AUC / LogLoss / Brier / ECE 来自'
        '「训练集 + 验证集合并重训」（43 表），与 97 表「仅训练集训练」口径'
        '（测试 AUC %.4f）不同源，两处数字不能混着比。</div>'
        % (CALIB.loc[1, '测试集 AUC'], CALIB.loc[1, '测试集 AUC'] * 100, KS5,
           float(FINAL['首档 Lift']), CAP30,
           len(_better), len(_worse),
           '、'.join(_better) if _better else '无',
           '、'.join(_worse) if _worse else '无',
           _d_ks, _d_ece, _d_mce, BEST_SEED_SPREAD,
           FINAL['测试 auc']),
        unsafe_allow_html=True)

    takeaway('主指标 <mark>AUC %.4f</mark>，'
             '<mark>人工只审打分最高的 30%% 就能拦住 %.1f%% 的违约客户</mark>。'
             '够配合人工审批用，<b>不够替人审批</b>——'
             '当排序工具用合适，当自动决策依据还差一截。'
             % (CALIB.loc[1, '测试集 AUC'], CAP30))

elif PAGE == '阈值决策':
    st.markdown('## 阈值决策')
    st.markdown(
        '<div class="note" style="margin-bottom:14px">'
        '下面的 <b>违约损失率 LGD</b> 与 <b>误拒机会成本</b> 两个假设'
        '直接决定本页的金额成本与最优阈值：'
        '漏拒损失 = FN 笔 × 平均放款额 ¥%s × LGD；'
        '误拒损失 = FP 笔 × 机会成本。<br>'
        '拖动后本页的<b>金额总成本</b> KPI、<b>成本构成图</b>、'
        '<b>金额总成本曲线</b>与<b>最优阈值</b>标注会同步变化。'
        '默认取值附近成本曲线较平坦，最优阈值变化不明显；'
        '把 LGD 拖到两端（如 0.90 或 0.20）会看到阈值明显移动。</div>'
        % format(AVG_LOAN, ',.0f'), unsafe_allow_html=True)
    th = TSCAN.sort_values('阈值').reset_index(drop=True)
    # 三个拉杆并排放在同一块控制区里：决策阈值 + 两个业务假设
    _sc = st.columns(3)
    v = _sc[0].slider('决策阈值（预测违约概率 ≥ 阈值则拒绝）',
                      float(th['阈值'].min()), float(th['阈值'].max()),
                      float(FINAL['业务最优阈值']), 0.01)
    LGD = _sc[1].slider('违约损失率 LGD', 0.20, 1.00,
                        st.session_state.get('_lgd', 0.60), 0.05,
                        help='业务假设，非本项目实测。漏拒损失 = FN 笔 × '
                             '平均放款额 ¥%s × LGD。调高表示拒贷代价更重，'
                             '最优阈值会下移。' % format(AVG_LOAN, ',.0f'))
    OPP = _sc[2].slider('误拒一笔的机会成本（元）', 300, 6000,
                        st.session_state.get('_opp', 1900), 100,
                        help='业务假设，非本项目实测。误拒损失 = FP 笔 × '
                             '本值。默认按平均放款额 × 平均年利率计一年利息。')
    st.session_state['_lgd'], st.session_state['_opp'] = LGD, OPP
    st.markdown(
        '<div class="note tip" style="font-size:14.5px;padding:12px 15px;'
        'margin-top:4px">'
        'LGD 与机会成本是<b>业务假设</b>，不是本项目实测结果；'
        '拖动后本页金额成本与最优阈值同步变化。'
        '</div>', unsafe_allow_html=True)
    row = th.iloc[(th['阈值'] - v).abs().argmin()]
    rec = THRESH[THRESH['模型'] == V5].iloc[0]

    # 按当前 LGD / 机会成本假设实算金额成本
    fn_loss = th['漏放数 FN'] * AVG_LOAN * LGD
    fp_loss = th['误拒数 FP'] * OPP
    th = th.assign(金额总成本=fn_loss + fp_loss, 漏放损失=fn_loss, 误拒损失=fp_loss)
    row = th.iloc[(th['阈值'] - v).abs().argmin()]
    best_row = th.loc[th['金额总成本'].idxmin()]

    r1c1, r1c2, r1c3 = st.columns(3)
    kpi(r1c1, '精确率', '%.4f' % row['精确率'], '拒绝的人中真违约占比')
    kpi(r1c2, '召回率', '%.4f' % row['召回率'], '真违约被拦下的比例')
    kpi(r1c3, 'F1', '%.4f' % row['F1'], '精确率与召回率调和平均')

    r2c1, r2c2, r2c3 = st.columns(3)
    kpi(r2c1, '拒绝率', '%.2f%%' % (row['拒绝率'] * 100), '被拒绝的申请占比')
    kpi(r2c2, '漏拒 / 误拒', '%d / %d' % (row['漏放数 FN'], row['误拒数 FP']),
        'FN：违约仍通过，FP：正常被拒')
    kpi(r2c3, '金额总成本', '¥%s' % format(row['金额总成本'], ',.0f'),
        'LGD %.2f · 机会成本 ¥%s' % (LGD, format(OPP, ',.0f')), alt=True)

    st.markdown('<div class="note tip" style="margin:10px 0 14px">'
                '上表为当前阈值 %.2f 处的指标；'
                '「金额总成本」为按上面 LGD 与机会成本实算的金额口径，'
                '与下方 5 : 1 笔数口径的「加权成本」不是同一指标，不可直接比较。'
                '</div>' % v, unsafe_allow_html=True)

    st.markdown('')
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for col, lab, colr in [('精确率', '精确率', MAIN), ('召回率', '召回率', ALT),
                               ('F1', 'F1', GREY)]:
            fig.add_scatter(x=th['阈值'], y=th[col], name=lab, mode='lines',
                            line=dict(color=colr, width=2.0))
        fig.add_vline(x=v, line=dict(color=LINE, width=1.4, dash='dash'),
                      annotation_text='当前 %.2f' % v,
                      annotation_font=dict(color=INK, size=11))
        fig.update_layout(xaxis=dict(title='阈值：模型判定的违约概率 ≥ 此值则拒绝'),
                          yaxis=dict(title='指标值'))
        style(fig, 330)
        head(st, '精确率、召回率与 F1 随阈值的变化', '76_threshold_scan.csv（测试集扫描）')
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            '<div class="note">三条线是"阈值定在哪"'
            '对应的三个业务口径——'
            '<b>召回率</b>＝真违约客户里被拒掉的比例（抓得全不全）；'
            '<b>精确率</b>＝被拒的人里真违约的比例（拒得准不准）；'
            'F1 是前两者的调和平均，用来衡量两个口径的平衡。<br>'
            '阈值往右移＝拒得越狠，召回率越高但精确率越低；'
            '往左移则相反，两条线必然交叉——'
            '<b>交叉点附近就是"拒得又准又全"的折中位置</b>。<br>'
            '在当前阈值 %.2f 处，召回率 %.1f%%、精确率 %.1f%%，'
            '意味着每拒 100 个人能拦下约 %.0f 个真违约客户，'
            '代价是放过了 %.1f%% 的好客户。</div>'
            % (v, row['召回率'] * 100, row['精确率'] * 100,
               row['精确率'] * 100, (1 - row['精确率']) * 100),
            unsafe_allow_html=True)
    with b:
        fig = go.Figure()
        fig.add_scatter(x=th['阈值'], y=th['漏放损失'], name='漏拒损失（本金 × LGD）',
                        mode='lines', line=dict(color=MAIN, width=2.2),
                        fill='tozeroy', fillcolor='rgba(76,114,176,.08)')
        fig.add_scatter(x=th['阈值'], y=th['误拒损失'], name='误拒损失（机会成本）',
                        mode='lines', line=dict(color=ALT, width=2.2),
                        fill='tozeroy', fillcolor='rgba(221,132,82,.08)')
        fig.add_vline(x=float(best_row['阈值']), line=dict(color=INK, width=1.6, dash='dash'),
                      annotation_text='最优 %.2f' % float(best_row['阈值']),
                      annotation_font=dict(color=INK, size=11))
        fig.update_layout(xaxis=dict(title='阈值'), yaxis=dict(title='金额（元）'))
        style(fig, 330)
        head(st, '漏拒损失与误拒损失的金额分解', '76_threshold_scan.csv · 51_threshold_after_opt.csv')
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            '<div class="note">这张图把上一步那三条比例曲线'
            '换算成钱。蓝线是<b>漏拒损失</b>——阈值越低拒得越狠，漏拒越少，'
            '这条线往下走；橙线是<b>误拒损失</b>——阈值越低拒得越狠，'
            '错拒好客户越多，这条线往上走。<br>'
            '两条线一升一降，交点附近就是总成本最低处'
            '（当前假设下为 %.2f）。'
            '这也是必须扫全区间、而不是拍一个阈值的原因：'
            '<b>只有两条线交叉的那个点才是省钱的位置</b>。</div>'
            % float(best_row['阈值']), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    l, r = st.columns(2)
    with r:
        i = th.index[th['阈值'].idxmin()]
        _lo_l = float(th.loc[i, '漏放损失'])
        _fp_l = float(th.loc[i, '误拒损失'])
        _tot_l = _lo_l + _fp_l
        fig = go.Figure(go.Pie(
            labels=['漏拒损失（本金 × LGD %.2f）' % LGD,
                    '误拒损失（机会成本 ¥%s）' % format(OPP, ',.0f')],
            values=[_lo_l, _fp_l], hole=0.62,
            marker=dict(colors=[MAIN, ALT], line=dict(color='#FFFFFF', width=2)),
            textinfo='percent', textfont=dict(size=13, color='#FFFFFF'),
            hovertemplate='%{label}<br>¥%{value:,.0f}（占 %{percent}）<extra></extra>'))
        fig.update_layout(showlegend=True,
                          legend=dict(orientation='v', x=0.02, y=0.5,
                                      font=dict(size=11)),
                          annotations=[dict(
                              text='<b>¥%s</b><br><span style="font-size:12.5px;'
                                   'color:#7A8090">最优阈值 %.2f<br>总成本</span>'
                                   % (format(_tot_l, ',.0f'), float(th.loc[i, '阈值'])),
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=17, color=INK))])
        style(fig, 330)
        head(st, '最优阈值处的成本构成', '按 LGD %.2f、机会成本 ¥%s 实算' % (LGD, format(OPP, ',.0f')))
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            '<div class="note">在当前假设下的最优阈值 %.2f 处，'
            '每 8 万笔测试申请会产生的两项成本分别是：'
            '<b>漏拒损失 ¥%s</b>（%d 笔漏拒 × 平均放款额 × LGD %.2f）和'
            '<b>误拒损失 ¥%s</b>（%d 笔误拒 × 机会成本），合计 ¥%s。<br>'
            '当前假设下误拒成本占总成本的 <b>%.1f%%</b>，'
            '漏拒成本占 <b>%.1f%%</b>——'
            '说明"拒得太紧"和"放得太松"都不能接受，'
            '必须落在两者平衡的那个阈值上；'
            '这也是下一页要扫全阈值区间的原因。</div>'
            % (float(th.loc[i, '阈值']), format(_lo_l, ',.0f'),
               int(th.loc[i, '漏放数 FN']), LGD, format(_fp_l, ',.0f'),
               int(th.loc[i, '误拒数 FP']), format(_tot_l, ',.0f'),
               100 * _fp_l / _tot_l, 100 * _lo_l / _tot_l),
            unsafe_allow_html=True)
    with l:
        fig = go.Figure()
        fig.add_scatter(x=th['阈值'], y=th['金额总成本'], name='金额总成本',
                        mode='lines', line=dict(color=MAIN, width=2.4),
                        fill='tozeroy', fillcolor='rgba(76,114,176,.08)')
        fig.add_vline(x=float(best_row['阈值']), line=dict(color=ALT, width=1.8, dash='dash'),
                      annotation_text='最优 %.2f' % float(best_row['阈值']),
                      annotation_font=dict(color=ALT, size=11))
        fig.add_vline(x=v, line=dict(color=LINE, width=1.4, dash='dash'),
                      annotation_text='当前 %.2f' % v,
                      annotation_font=dict(color=INK, size=11))
        fig.update_layout(xaxis=dict(title='阈值：模型判定的违约概率 ≥ 此值则拒绝'),
                          yaxis=dict(title='金额总成本（元）'))
        style(fig, 330)
        head(st, '总成本曲线', '76_threshold_scan.csv（按当前 LGD / 机会成本实算）')
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            '<div class="note">横轴是决策阈值，'
            '纵轴是按当前假设实算的总成本；<b>曲线最低点就是成本最小的阈值</b>'
            '（%.2f，¥%s），也就是本页推荐值。'
            '阈值往左移＝拒得更狠（漏拒少、误拒多），往右移＝放得更松。<br>'
            '图上橙色虚线是最优阈值，灰色虚线是你当前正在看的阈值。'
            '两条线离得远，说明当前阈值离最省钱的设置还有距离。'
            '曲线在最低点附近都比较平——'
            '<b>这意味着小幅调整阈值不会明显改变成本</b>，'
            '业务上可以按人工审批量去挑一个附近的值，而不必死守最优点。</div>'
            % (float(best_row['阈值']), format(float(best_row['金额总成本']), ',.0f')),
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="note" style="margin-bottom:14px">'
        '<b>为什么要区分漏拒和误拒？</b>'
        '银行放出去的每一笔贷款只有两种结局：客户按期还款，'
        '银行收回本金并赚到一笔利息；或者客户违约，银行<b>连本金都收不回来</b>。'
        '所以两类错误对银行的意义完全不对等：<br>'
        '<b>漏拒（FN）</b>＝该拒的客户被放过去了。银行要承担<b>本金损失</b>，'
        '按平均放款额 ¥%s、LGD %.2f 算，一笔净损失约 ¥%s。<br>'
        '<b>误拒（FP）</b>＝本来能还清的客户被拒了。银行只是<b>少赚一笔利息</b>'
        '（按平均放款额 × 平均年利率 %.2f%% 算，一年约 ¥%s），'
        '本金没有损失。<br>'
        '<b>两者的量级差了 %.1f 倍</b>，这就是"惩罚机制"的由来：'
        '不能把两类错误当成同样严重，必须给漏拒更高的权重。'
        '下一组图就是在扫这个权重——<b>左侧横轴 5 : 1 的意思是'
        '"漏拒 1 笔，视同误拒 5 笔的代价"</b>。</div>'
        % (format(AVG_LOAN, ',.0f'), LGD, format(AVG_LOAN * LGD, ',.0f'),
           AVG_RATE, format(AVG_LOAN * AVG_RATE / 100, ',.0f'),
           (AVG_LOAN * LGD) / (AVG_LOAN * AVG_RATE / 100)),
        unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        ct = COST.copy()
        fig = go.Figure(go.Bar(x=ct['漏放:误拒'], y=ct['最优阈值'], marker_color=MAIN,
                               text=ct['最优阈值'].round(2), textposition='outside',
                               textfont=dict(size=11, color=INK)))
        fig.add_hline(y=float(rec['业务最优阈值']), line=dict(color=ALT, width=1.6,
                                                             dash='dash'),
                      annotation_text='本项目采用 %.2f' % float(rec['业务最优阈值']),
                      annotation_font=dict(color=ALT, size=11))
        fig.update_layout(yaxis=dict(title='最优阈值（越低＝拒得越狠）',
                                     range=[0, 0.62]))
        style(fig, 430)
        head(st, '惩罚力度与最优阈值的关系', '77_cost_ratio_threshold.csv')
        st.plotly_chart(fig, width='stretch')
    with r:
        ct_ratio0 = ct['漏放:误拒'].str.split(':').str[0].astype(float)
        _lo0 = ct.loc[ct['最优阈值'].idxmax()]
        _hi0 = ct.loc[ct['最优阈值'].idxmin()]
        _r_lo0 = float(ct_ratio0[ct['最优阈值'].idxmax()])
        _r_hi0 = float(ct_ratio0[ct['最优阈值'].idxmin()])
        show = ct[['漏放:误拒', '最优阈值', '最小加权成本', '漏放数 FN',
                   '误拒数 FP', '精确率', '召回率']].copy()
        show.columns = ['漏拒 1 笔 = 误拒几笔', '最优阈值', '最小加权成本',
                        '漏拒数 FN', '误拒数 FP', '精确率', '召回率']
        head(st, '各惩罚力度下自动选出的最优决策点', '77_cost_ratio_threshold.csv')
        st.dataframe(show.style.format({'最优阈值': '{:.2f}',
                                        '最小加权成本': '{:,.0f}',
                                        '精确率': '{:.3f}', '召回率': '{:.3f}'}),
                     width='stretch', hide_index=True)
        st.markdown(
            '<div class="note">第一列是惩罚力度'
            '（漏拒 1 笔算作误拒几笔），后面的列是在这个力度下'
            '系统自动选出的最优决策点。'
            '看"精确率"这一列往下掉、'
            '"误拒数 FP"这一列往上涨，就说明<b>拒绝线被拉紧了</b>。</div>',
            unsafe_allow_html=True)
    st.markdown(
        '<div class="note">惩罚越重，为了躲开漏拒就越要早下拒绝线，'
        '<b>误拒数会明显上升、精确率明显下降</b>。'
        '惩罚力度从 %g : 1 提到 %g : 1，'
        '最优阈值从 %.2f 压到 %.2f，'
        '要拒的人从 %s 笔涨到 %s 笔（约 %g 倍），'
        '代价是精确率从 %.3f 掉到 %.3f。</div>'
        % (_r_lo0, _r_hi0, float(_lo0['最优阈值']), float(_hi0['最优阈值']),
           format(int(_lo0['误拒数 FP']), ','), format(int(_hi0['误拒数 FP']), ','),
           round(float(_hi0['误拒数 FP']) / float(_lo0['误拒数 FP'])),
           float(_lo0['精确率']), float(_hi0['精确率'])),
        unsafe_allow_html=True)
    # 「漏放:误拒」列是字符串 '5:1' 形态。冒号左边是漏放侧的倍数（漏拒 1 笔 = 误拒 N 笔），
    #   注意不能取 str[1]（那是误拒侧，恒为 1）。
    ct_ratio = ct['漏放:误拒'].str.split(':').str[0].astype(float)
    _lo = ct.loc[ct['最优阈值'].idxmax()]      # 代价最接近 1:1 → 阈值最高
    _hi = ct.loc[ct['最优阈值'].idxmin()]      # 代价最悬殊 → 阈值最低
    _r_lo = float(ct_ratio[ct['最优阈值'].idxmax()])
    _r_hi = float(ct_ratio[ct['最优阈值'].idxmin()])
    st.markdown(
        '<div class="note"><b>本项目最终采用阈值 %.2f。</b>'
        '这是 5 : 1 惩罚口径下加权成本最小的阈值，'
        '和「风险分层」页是同一套口径。<br>'
        '为什么落在 5 : 1 这个量级？把两类错误的单笔代价都算出来对一下：'
        '按平均放款额 ¥%s、平均年利率 %.2f%% 算，'
        '<b>误拒 1 笔＝少赚约 ¥%s 利息</b>；'
        '按 LGD %.2f 算，<b>漏拒 1 笔＝净损失约 ¥%s</b>，'
        '两者相差 <b>%.1f 倍</b>。'
        '5 : 1 落在这个实测区间内，'
        '所以本项目不另外假设倍数，直接按 <b>1 : 5</b> 做加权成本最小化。<br>'
        '需要说明的是：真实的核销率要用银行实际数据替换 LGD 假设，'
        '这个倍数随之会变，'
        '<b>但"漏拒一定比误拒贵、必须给漏拒更高权重"这个方向不会变</b>。<br>'
        '另外还有一个<b>纯技术口径</b>算出的阈值 %.2f'
        '（Youden J = %.4f）：它只追求"抓得全、误判少"，'
        '<b>假设两类错误代价相同</b>，'
        '所以算出来的拒绝线更松。'
        '那个口径用来横向比较模型可以，用来定审批线不行——'
        '因为它假装漏拒和误拒一样严重。</div>'
        % (float(rec['业务最优阈值']), format(AVG_LOAN, ',.0f'), AVG_RATE,
           format(AVG_LOAN * AVG_RATE / 100, ',.0f'), LGD,
           format(AVG_LOAN * LGD, ',.0f'),
           (AVG_LOAN * LGD) / (AVG_LOAN * AVG_RATE / 100),
           float(rec['技术最优阈值']), float(rec['YoudenJ'])),
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif PAGE == '风险分层':
    st.markdown('## 风险分层')
    lv4 = LIFT[LIFT['模型'] == V4].sort_values('十分位')
    lv5 = LIFT[LIFT['模型'] == V5].sort_values('十分位')
    BASE_RATE = float(TARGET.loc[TARGET['isDefault'] == 1, '占比(%)'].iloc[0]) / 100
    dr5 = lv5['Lift'].values * BASE_RATE * 100      # 各档实际违约率（%）
    dr4 = lv4['Lift'].values * BASE_RATE * 100
    # 累计捕获率：前 k 档（各占 10%）累计拦住的违约客户占比
    cap5 = np.cumsum(lv5['Lift'].values) / lv5['Lift'].values.sum() * 100
    # 每档固定占 10% 人数 → 前 30% = 前 3 档，前 50% = 前 5 档
    n3, n5 = 3, 5
    cap30, cap50 = cap5[n3 - 1], cap5[n5 - 1]

    st.markdown(
        '<div class="note" style="margin-bottom:14px">'
        '<b>如果人工只够审一部分人，先审谁？</b><br>'
        '把测试集按模型打分<b>从高到低</b>排成 10 档（每档占全部申请的 10%%），'
        '统计每一档的违约率。结果是：'
        '风险最高的第 1 档违约率 <b>%.1f%%</b>，是整体 %.2f%% 的 <b>%.2f 倍</b>；'
        '风险最低的第 10 档只有 <b>%.1f%%</b>。<br>'
        '也就是说，只审打分最高的 30%% 的人，'
        '能拦住 <b>%.1f%%</b> 的违约客户。'
        '剩下 70%% 的人几乎不用占用人工审批，'
        '这正是限额审批、把人工尽调集中到高风险人群的前提。</div>'
        % (dr5[0], BASE_RATE * 100, lv5['Lift'].iloc[0], dr5[-1], cap30),
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, '风险最高那 10% 人', '%.1f%%' % dr5[0],
        '实际违约率（首档 Lift %.3f）' % lv5['Lift'].iloc[0], alt=True)
    kpi(c2, '审前 30% 能拦住', '%.1f%%' % cap30,
        '的违约客户')
    kpi(c3, '风险最低那 10% 人', '%.1f%%' % dr5[-1],
        '实际违约率（末档 Lift %.3f）' % lv5['Lift'].iloc[-1])
    kpi(c4, '首档相对倍数', '%.2f 倍' % lv5['Lift'].iloc[0],
        '最高档 ÷ 整体违约率')

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        head(st, '各档实际违约率',
             '68_lift_decile.csv × 01_target_distribution.csv'
             '（该档违约率 = Lift × 整体违约率）')
        fig = go.Figure()
        fig.add_bar(x=lv5['十分位'], y=dr4, name='未调优默认超参',
                    marker_color=LIGHT, marker_line=dict(color=GREY, width=0.5))
        fig.add_bar(x=lv5['十分位'], y=dr5, name='最优配置',
                    marker_color=MAIN, marker_line=dict(color=MAIN, width=0.5))
        fig.add_hline(y=BASE_RATE * 100, line=dict(color=ALT, width=1.4, dash='dash'),
                      annotation_text='整体 %.2f%%' % (BASE_RATE * 100),
                      annotation_font=dict(size=10, color=ALT))
        fig.update_layout(barmode='group',
                          xaxis=dict(title='风险档位（从左到右风险递减）',
                                     dtick=1),
                          yaxis=dict(title='该档实际违约率 (%)', range=[0, 58]))
        style(fig, 320)
        st.plotly_chart(fig, width='stretch')
    with b:
        head(st, '各档相对倍数（首档 Lift）', '68_lift_decile.csv')
        fig = go.Figure()
        fig.add_bar(x=lv4['十分位'], y=lv4['Lift'], name='未调优默认超参',
                    marker_color=LIGHT, marker_line=dict(color=GREY, width=0.5))
        fig.add_bar(x=lv5['十分位'], y=lv5['Lift'], name='最优配置',
                    marker_color=MAIN, marker_line=dict(color=MAIN, width=0.5))
        fig.add_hline(y=1.0, line=dict(color=LINE, width=1.2, dash='dash'),
                      annotation_text='1.0 倍 = 与整体持平（无区分度）',
                      annotation_font=dict(size=10, color=LINE))
        fig.update_layout(barmode='group',
                          xaxis=dict(title='风险档位（从左到右风险递减）', dtick=1),
                          yaxis=dict(title='该档违约率 ÷ 整体违约率'))
        style(fig, 320)
        st.plotly_chart(fig, width='stretch')
    st.markdown(
        '<div class="note">'
        '左图是绝对值——这群人里真的有多少会违约；'
        '右图是相对倍数——这档人比整体高或低多少。'
        '倍数＝该档违约率 ÷ 整体违约率，'
        '2.0 倍就是"违约率是整体的 2 倍"，0.4 倍就是"只有 0.4 倍"。<br>'
        '第 1 档 %.2f 倍、第 2 档 %.2f 倍，'
        '一路降到第 %d 档只有 %.2f 倍——'
        '<b>风险随档位单调下降，说明模型的风险排序是有效的</b>。'
        '从第 %d 档起已经低于 1 倍（比整体安全），'
        '这几档的申请人可以走快速审批通道，不必占用人工。</div>'
        % (lv5['Lift'].iloc[0], lv5['Lift'].iloc[1],
           len(lv5), lv5['Lift'].iloc[-1],
           int(lv5[lv5['Lift'] < 1.0]['十分位'].iloc[0])),
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    head(st, '十档逐档明细', '68_lift_decile.csv · 97_peak_round_choice.csv')
    st.markdown(
        '<div class="note" style="margin-bottom:8px">'
        '"累计拦住"一列是重点：如果人工预算只够审前面 k 档，'
        '看这一列就知道能拦住多少比例的违约客户。'
        '每一档都占全部申请的 10%，所以看到第 3 行就是"审前 30% 的人"。</div>',
        unsafe_allow_html=True)
    _rank = ['第 %d 档' % k for k in range(1, len(lv5) + 1)]
    # 「最危险」标在第 1 档那一格，不塞在表头里——表头保持干净
    _rank[0] = '第 1 档（最危险）'
    det = pd.DataFrame({
        '风险档位': _rank,
        '人数占比': ['10.0%'] * len(lv5),
        '实际违约率 (%)': dr5,
        '相对倍数（首档 Lift）': lv5['Lift'].values,
        '累计人数占比 (%)': lv5['十分位'].values * 10,
        '累计拦住违约客户 (%)': cap5,
    })
    st.dataframe(det.style.format({'实际违约率 (%)': '{:.1f}',
                                  '相对倍数（首档 Lift）': '{:.2f}',
                                  '累计人数占比 (%)': '{:.0f}',
                                  '累计拦住违约客户 (%)': '{:.1f}'}),
                 width='stretch', hide_index=True)
    st.markdown(
        '<div class="note"><b>结论（这就是风控最直接能用的产出）：</b>'
        '前 30%% 的人（%d 档）拦住 <b>%.1f%%</b> 的违约客户；'
        '前 50%% 的人（%d 档）拦住 <b>%.1f%%</b>。'
        '换句话说，<b>把人工审批量砍掉一半，仍然能抓到约 %.0f%% 的坏客户</b>。'
        '这就是"限额审批 + 人工尽调"这套做法能成立的数据依据。<br>'
        '需要注意的是：本页各档违约率由 Lift × 整体违约率换算得到，'
        '口径与"阈值决策"页的精确率 / 召回率不同，'
        '两边不可直接对照。</div>' % (n3, cap30, n5, cap50, cap50),
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    takeaway('<mark>人工审批量砍掉一半，'
             '仍能拦住约 %.0f%% 的坏客户</mark>。'
             '这就是"限额审批 + 人工尽调"能成立的数据依据——'
             '也是这套模型最直接能用的产出。' % cap50)

elif PAGE == '稳定性与校准':
    st.markdown('## 稳定性与校准')
    st.markdown(
        '<div class="note" style="margin-bottom:14px">'
        '<b>模型上线一年后还敢用吗？</b><br>'
        '这是风控模型必答的两个问题——'
        '<b>人群变了会不会失效</b>、<b>概率数字能不能直接信</b>。<br>'
        '<b>① 人群变了（PSI = %.5f）：</b>'
        'PSI 衡量"新一批客户和当初训练的人长得有多不像"。'
        '低于 0.10 算稳定，本项目 %.5f 远在这条线以内，可以继续用；'
        '一旦超过 0.25 就必须重建模型。<br>'
        '<b>② 概率能不能信（ECE = %.5f）：</b>'
        '模型说「这批人违约概率 20%%」，实际违约率是否接近 20%%。'
        '本项目偏差仅 %s 个百分点，'
        '意味着<b>概率数字可以直接当决策依据</b>，'
        '不用再额外做人工修正——这一点很重要，'
        '因为审批规则通常直接写成"概率超过 X%% 就拒"。</div>'
        % (float(PSI.loc[1, 'PSI_验证集对测试集']),
           float(PSI.loc[1, 'PSI_验证集对测试集']),
           float(CALIB.loc[1, 'ECE_测试集']),
           '%.2f' % (float(CALIB.loc[1, 'ECE_测试集']) * 100)),
        unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, '人群稳定性 PSI', '%.5f' % float(PSI.loc[1, 'PSI_验证集对测试集']),
        '关注线 0.10 · 判定：稳定', alt=True)
    kpi(c2, '概率准确度 ECE', '%.5f' % CALIB.loc[1, 'ECE_测试集'],
        '越小概率越可信')
    kpi(c3, '概率精度 Brier', '%.5f' % CALIB.loc[1, 'Brier'], '越低越好')
    kpi(c4, '最差分箱偏差 MCE', '%.5f' % CALIB.loc[1, 'MCE_测试集'],
        '单个分箱最差情况')

    st.markdown('')
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="note" style="margin-bottom:8px">'
        '做法：把测试集按模型给的概率从低到高排，'
        '平均分成 10 箱，看每一箱的预测概率和实际违约率差多少。'
        '灰色虚线是"完全对得上"的理想情况，点贴在线上就说明概率可信。</div>',
        unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        head(st, '概率校准曲线（预测概率 vs 实际违约率）',
             '45_calibration_figure_after.csv')
        fig = go.Figure()
        for name, colr in [('XGBoost_默认超参_第4阶段', LIGHT),
                           ('XGBoost_调优后_第5阶段', MAIN)]:
            g = CALIB_FIG[CALIB_FIG['模型'] == name].sort_values('分箱')
            fig.add_scatter(x=g['平均预测概率'], y=g['实际违约率'],
                            name=('未调优默认超参' if colr == LIGHT else '最优配置'),
                            mode='lines+markers',
                            line=dict(color=colr, width=2.2),
                            marker=dict(size=6, color=colr))
        fig.add_scatter(x=[0, 0.55], y=[0, 0.55], name='完美校准（实际=预测）',
                        mode='lines', line=dict(color=LINE, width=1.2, dash='dash'))
        fig.update_layout(xaxis=dict(title='模型给的平均预测概率'),
                          yaxis=dict(title='实际发生的违约率'))
        style(fig, 340)
        st.plotly_chart(fig, width='stretch')
        _mx = CALIB_FIG[CALIB_FIG['模型'] == 'XGBoost_调优后_第5阶段']
        st.markdown(
            '<div class="note">蓝线基本贴在虚线上，'
            '最大偏差 %+.2f 个百分点、平均偏差 %.5f。'
            '可以放心地在审批规则里直接写概率阈值，'
            '例如"模型给的风险概率超过 %.0f%% 就转人工"。</div>'
            % (float(_mx['偏差'].abs().max()) * 100,
               float(CALIB.loc[1, 'ECE_测试集']),
               100 * float(FINAL['业务最优阈值'])),
            unsafe_allow_html=True)
    with b:
        g = CALIB_FIG[CALIB_FIG['模型'] == 'XGBoost_调优后_第5阶段'].sort_values('分箱')
        colors = [ALT if abs(x) > 0.01 else MAIN for x in g['偏差']]
        fig = go.Figure(go.Bar(x=g['分箱'] + 1, y=g['偏差'] * 100, marker_color=colors,
                               text=['%+.2f' % (x * 100) for x in g['偏差']],
                               textposition='outside',
                               textfont=dict(size=9, color=INK)))
        fig.add_hline(y=0, line=dict(color=LINE, width=1.2))
        fig.update_layout(xaxis=dict(title='第几个箱（从低概率到高概率）',
                                     dtick=1),
                          yaxis=dict(title='实际违约率 − 预测概率（百分点）'))
        style(fig, 340)
        head(st, '每个概率档的预测偏差', '45_calibration_figure_after.csv')
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            '<div class="note">横条为正表示这箱人实际违约得比'
            '模型说的更多（模型低估了风险），为负则相反。'
            '阈值取 1 个百分点——%s。'
            '如果某一段偏差特别大，实际用的时候就要对那档概率做人工修正。</div>'
            % ('本次没有分箱超过这个量级，全部偏差都在 1 个百分点内'
               if int((g['偏差'].abs() > 0.01).sum()) == 0
               else '本次有 %d 个分箱超过' % int((g['偏差'].abs() > 0.01).sum())),
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        head(st, '各特征稳定性（PSI）', '08_feature_psi_stability.csv')
        st.markdown(
            '<div class="note" style="margin-bottom:8px">'
            'PSI 逐特征算：某一项特征在新一批人群里的分布，'
            '和当初训练时的分布差多少。差得越多说明这项特征'
            '"过期"得越快，模型用到它时越容易失准。</div>',
            unsafe_allow_html=True)
        fp = FPSI.sort_values('psi', ascending=False).head(15).iloc[::-1]
        colrs = [ALT if x > 0.25 else (GREY if x > 0.1 else MAIN) for x in fp['psi']]
        fig = go.Figure(go.Bar(x=fp['psi'], y=fp['feature'], orientation='h',
                               marker_color=colrs,
                               text=['%.4f' % x for x in fp['psi']],
                               textposition='outside',
                               textfont=dict(size=10, color=INK)))
        # 阈值线：标签放在绘图区顶部留出的空白带（y 范围上端之外），
        # 放在底部会压在最下面几根短条上
        for xv, lab, colr in [(0.1, '0.10 关注线', GREY), (0.25, '0.25 重大漂移', ALT)]:
            fig.add_vline(x=xv, line=dict(color=colr, width=1.2, dash='dash'))
            fig.add_annotation(x=xv, y=len(fp) - 0.35, text=lab, showarrow=False,
                               xanchor='left', xshift=4, yanchor='bottom',
                               font=dict(size=9, color=colr))
        fig.update_layout(xaxis=dict(title='PSI（越大表示分布漂移越严重）',
                                     range=[0, float(fp['psi'].max()) * 1.18]),
                          yaxis=dict(automargin=True, range=[-0.6, len(fp) + 0.9]))
        style(fig, 340)
        st.plotly_chart(fig, width='stretch')
        _n_unstable = int((fp['psi'] > 0.1).sum())
        _n_crit = int((fp['psi'] > 0.25).sum())
        st.markdown(
            '<div class="note">横条越长，漂移越严重。'
            '前 15 个特征里有 <b>%d 个超过 0.10 关注线</b>，'
            '其中 %d 个超过 0.25 重大漂移线。</b><br>'
            '排在最前的 issue_year（放款年份，PSI %.2f）、'
            'title_te（职位编码，%.2f）、regionCode_te（地区编码，%.2f）'
            '漂移特别大，这是<b>必然结果</b>：这张表是用'
            '"2013 年起训练、往后预测"的时间切分算的，'
            '年份分布本来就变了，两个目标编码字段也跟着变。'
            '不表示这些特征不可用，但上线后要按<b>季度</b>重算 PSI 做监控；'
            '真正需要警惕的是那种 PSI 悄悄爬升的特征。</div>'
            % (_n_unstable, _n_crit,
               float(FPSI.loc[FPSI['feature'] == 'issue_year', 'psi'].iloc[0]),
               float(FPSI.loc[FPSI['feature'] == 'title_te', 'psi'].iloc[0]),
               float(FPSI.loc[FPSI['feature'] == 'regionCode_te', 'psi'].iloc[0])),
            unsafe_allow_html=True)
    with r:
        head(st, '时间外推验证（2017—2018 年测试）', '97_peak_round_choice.csv（oot_y2013，各随机种子均值）')
        st.markdown(
            '<div class="note" style="margin-bottom:8px">'
            '这一块的成绩来自"用 2013~2016 年训练、'
            '拿 2017~2018 年测试"的时间外推切分，'
            '相当于直接考"模型能不能预测未来"。</div>',
            unsafe_allow_html=True)
        o = PEAK[(PEAK['切分'] == 'oot_y2013') & (PEAK['特征集'] == '全特征')]
        oo = o.groupby('配置').agg(
            测试AUC=('测试 auc', 'mean'), 验证AUC=('验证 AUC', 'mean')).reset_index()
        # 柱与散点必须共用同一套 x（都用显示名）；否则 plotly 会把内部代号
        # 当成另外 4 个分类刻度，x 轴上会出现 8 个标签。
        oo['显示名'] = oo['配置'].map(CFG_MAP).fillna(oo['配置'])
        _xo = list(range(len(oo)))
        fig = go.Figure()
        fig.add_bar(x=_xo, y=oo['测试AUC'], name='未来数据测试 AUC',
                    marker_color=MAIN,
                    text=['%.4f' % v for v in oo['测试AUC']],
                    textposition='outside',
                    textfont=dict(size=10, color=INK))
        fig.add_scatter(x=_xo, y=oo['验证AUC'], name='同期验证 AUC',
                        mode='markers',
                        marker=dict(size=9, color=ALT, symbol='diamond'))
        fig.update_layout(xaxis=dict(tickvals=_xo, ticktext=list(oo['显示名']),
                                     tickfont=dict(size=10), showgrid=False),
                          yaxis=dict(title='AUC', range=[0.70, 0.735]))
        style(fig, 340)
        st.plotly_chart(fig, width='stretch')
        _o_best = oo.loc[oo['测试AUC'].idxmax()]
        _o_cfg = _o_best['显示名']
        _gap = float(oo['测试AUC'].max()) - float(oo['测试AUC'].min())
        # 排序是否与随机切分一致：逐配置实算比较，不凭印象
        _r_rank = list(_PEAK_SEL.groupby('显示名')['测试 auc'].mean()
                       .sort_values(ascending=False).index)
        _o_rank = list(oo.sort_values('测试AUC', ascending=False)['显示名'])
        _same = (_o_rank == _r_rank)
        _tuned3 = (float(oo.nlargest(2, '测试AUC')['测试AUC'].iloc[0]
                     - oo.nlargest(2, '测试AUC')['测试AUC'].iloc[1])
                   if len(oo) > 3 else _gap)
        st.markdown(
            '<div class="note">在完全没见过的未来数据上，'
            '四组配置 AUC 落在 %.4f ~ %.4f，差距 %.4f，'
            '最高的是<b>%s</b>（%.4f）。<br>'
            '这组成绩比随机切分的平均分（%.4f）低了 %.4f——'
            '<b>"预测未来"确实比"预测同一时期"难</b>，这是正常且预期的。'
            '差距主要来自基线组偏低，三组调优配置彼此只差 %.4f。<br>'
            '排序%s：%s。</div>'
            % (float(oo['测试AUC'].min()), float(oo['测试AUC'].max()), _gap,
               _o_cfg, float(_o_best['测试AUC']),
               BEST_AUC, BEST_AUC - float(_o_best['测试AUC']),
               float(oo.nlargest(2, '测试AUC')['测试AUC'].iloc[0]
                     - oo.nlargest(2, '测试AUC')['测试AUC'].iloc[1]),
               '与随机切分完全一致' if _same else '与随机切分略有差异',
               ' → '.join(_o_rank)),
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif PAGE == '特征解释':
    st.markdown('## 特征解释')
    st.markdown(
        '<div class="note" style="margin-bottom:14px">'
        '<b>模型拒掉一笔申请，主要在看什么？</b><br>'
        '模型是一个决策树ensemble，它内部有几百棵树，'
        '要向业务解释"凭什么拒他"不能直接把树翻出来。'
        '这里用的 <b>SHAP 值</b>是一种通用的解释方法：'
        '把每个因素对最终结论的影响幅度折算成同一个刻度，'
        '<b>数值越大说明这项因素的话语权越大</b>（正值 = 把风险推高，'
        '负值 = 把风险压低）。<br>'
        '往下排的基本都是信用、负债、利率、收入这几类——'
        '和业务经验一致，说明模型没有跑到"看不懂"的地方。'
        '图表与表格里的字段名已全部替换成业务名称，'
        '原始字段名保留在明细表最后一列备查。</div>',
        unsafe_allow_html=True)

    n = st.slider('显示前多少个影响因素', 10, 30, 20, 1)
    s = SHAP.sort_values('mean_abs_shap', ascending=False).head(n).iloc[::-1].copy()
    s['显示名'] = s['feature'].map(lambda x: BIZ_NAME.get(x, x))
    s['业务含义'] = s['feature'].map(lambda x: BIZ_MEANING.get(x, ''))

    top1 = s['feature'].iloc[-1]
    r1c1, r1c2, r1c3 = st.columns(3)
    kpi(r1c1, '首要驱动因素', BIZ_NAME.get(top1, top1),
        BIZ_MEANING.get(top1, '原始字段 %s' % top1))
    kpi(r1c2, 'Top 5 解释力占比',
        '%.1f%%' % (100 * SHAP['mean_abs_shap'].nlargest(5).sum()
                    / SHAP['mean_abs_shap'].sum()),
        '五个因素解释了大部分判断')
    kpi(r1c3, 'Top 20 解释力占比',
        '%.1f%%' % (100 * SHAP['mean_abs_shap'].nlargest(20).sum()
                    / SHAP['mean_abs_shap'].sum()),
        '其余因素边际影响较小', alt=True)

    st.markdown('')
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    # 单一中文名做轴标签，避免「中文（英文）」双层文字与数值标签重叠
    fig = go.Figure(go.Bar(x=s['mean_abs_shap'], y=s['显示名'], orientation='h',
                           marker_color=MAIN,
                           text=['%.3f' % x for x in s['mean_abs_shap']],
                           textposition='outside',
                           textfont=dict(size=10, color=INK)))
    fig.update_layout(xaxis=dict(title='SHAP 值（对违约概率判断的影响幅度）',
                                 rangemode='tozero',
                                 range=[0, float(s['mean_abs_shap'].max()) * 1.16]),
                      yaxis=dict(tickfont=dict(size=11), automargin=True))
    head(st, '影响最大的 %d 个因素' % n, '09_shap_global_random.csv')
    style(fig, max(380, n * 22))
    st.plotly_chart(fig, width='stretch')
    st.markdown(
        '<div class="note">条形从下往上由长变短，'
        '最上面那根就是模型最看重的因素。'
        '调整左边的滑块可以看到更多因素——'
        '排在第 %d 名之后的影响都很小，实际审批时盯住前 20 项就够。</div>'
        % 20, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    head(st, '逐项业务解读', '09_shap_global_random.csv')
    dd = s.iloc[::-1][['显示名', 'feature', 'mean_abs_shap']].copy()
    dd['业务含义'] = s.iloc[::-1]['业务含义']
    dd['占比 (%)'] = 100 * dd['mean_abs_shap'] / SHAP['mean_abs_shap'].sum()
    dd.columns = ['业务名称', '原始字段', 'SHAP 值', '业务含义', '占比 (%)']
    dd = dd[['业务名称', '业务含义', 'SHAP 值', '占比 (%)', '原始字段']]
    st.dataframe(dd.style.format({'SHAP 值': '{:.4f}', '占比 (%)': '{:.2f}'}),
                 width='stretch', hide_index=True, height='auto')
    _anon = int(dd['原始字段'].isin(
        [f for f in dd['原始字段'] if f.startswith('n')]).sum())
    st.markdown(
        '<div class="note">"占比"一列是这一项在全部 %d 个特征里的权重份额。'
        '业务名称已经翻好；原始字段名留在最后一列，'
        '写代码或核对数据时才需要看。<br>'
        '其中 %d 项是原始数据里的<b>匿名字段</b>（n0~n14 之类，'
        '数据提供方没有给业务命名），它们只能按取值分布参与建模，'
        '无法用业务语言解释——落地时这类字段建议保留但不作为决策依据。'
        '</div>' % (len(SHAP), _anon), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif PAGE == '结论与局限':
    st.markdown('## 结论与局限')
    st.markdown(
        '<div class="note" style="margin-bottom:14px">'
        '这一页不引入新的图表，只回答三个问题：'
        '<b>① 现在这个模型能做到什么程度；② 它的天花板在哪里；'
        '③ 想更准的话，该往哪里投入。</div>', unsafe_allow_html=True)

    # ---- ① 现状：能力边界 ----
    st.markdown('## 现有模型能做到什么程度')
    e1, e2, e3 = st.columns(3)
    _oot_auc = float(PEAK[(PEAK['切分'] == 'oot_y2013') & (PEAK['特征集'] == '全特征')
                          & (PEAK['配置'].str.startswith('丙'))]['测试 auc'].mean())
    _top20 = SHAP.sort_values('mean_abs_shap', ascending=False).head(20)
    _anon_top20 = int(_top20['feature'].str.match(r'^n\d+$').sum())
    kpi(e1, '同批次判别能力（AUC）', '%.4f' % FINAL['测试 auc'],
        '审前 30%% 的人能抓 %.1f%% 坏账' % CAP30, alt=True)
    kpi(e2, '未来数据判别能力（AUC）', '%.4f' % _oot_auc,
        '2013~2016 训练 → 2017~2018 测试')
    kpi(e3, '人工只审 30% 能抓住', '%.1f%%' % CAP30, '其余 70% 走自动化')

    st.markdown(
        '<div class="note">把上面三个数连起来读，'
        '模型的<b>实际能力边界</b>是这样的：<br>'
        '<b>① 同一时期内，它能把坏账密度拉开约 %.1f 倍</b>'
        '（首档相对倍数 %.2f，KS %.4f）。<br>'
        '<b>② 放到未来数据上，准确度掉到 AUC %.4f</b>，'
        '掉了 %.4f——说明这份数据的风险规律本身会随时间漂移，'
        '模型必须定期重训。<br>'
        '<b>③ 它的最强用法不是"替代人工审批"，而是"把人工审批集中在最该看的人身上"</b>：'
        '审前 30%% 的人能抓 %.1f%% 的坏客户，'
        '等于<b>用三成的人工量覆盖掉五成半以上的坏账</b>。<br>'
        '换句话说，<b>这套模型的价值在于排序，不在于替人做决定</b>。</div>'
        % (float(FINAL['首档 Lift']), float(FINAL['首档 Lift']), KS5, _oot_auc,
           float(FINAL['测试 auc']) - _oot_auc, CAP30),
        unsafe_allow_html=True)

    takeaway('这套模型同批次 <mark>AUC %.4f</mark>、'
             '放到未来掉到 <mark>%.4f</mark>，'
             '<mark>用三成的人工量覆盖掉 %.1f%% 的坏账</mark>。'
             '<b>它的价值在于排序，不在于替人做决定</b>——'
             '要更准，得换信息来源，不是继续调参。'
             % (FINAL['测试 auc'], _oot_auc, CAP30))

    # ---- ② 天花板在哪 ----
    st.markdown('## 为什么停在这个水平（天花板）')
    _gap_auc = float(INFL['均值极差'].max())
    st.markdown(
        '<div class="note">调优一共跑了 %d 组实验，'
        '7 个超参每个都试了多个取值，'
        '<b>但把它们的影响强度排序后可以看到：最强的参数也只值 %.5f 的 AUC</b>'
        '（各参数极差区间 %.5f ~ %.5f），'
        '而四组候选配置的平均 AUC 总跨度只有 %.4f。'
        '这说明：<b>不是没调够，而是这份数据的可挖空间就这么大了</b>。<br>'
        '继续在超参上耗时间，收益上限是<b>千分之几</b>；'
        '要再往上走，得换别的信息来源。</div>'
        % (N_GROUPS, _gap_auc, float(INFL['均值极差'].min()), _gap_auc,
           BEST_AUC - float(_PEAK_SEL.groupby('显示名')['测试 auc'].mean().min())),
        unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    head(st, '四个已知局限，以及各自的实际影响',
         '97_peak_round_choice.csv · 09_shap_global_random.csv · '
         '45_calibration_figure_after.csv · 99_cv_final_c_random.csv')
    # 局限只谈"数据与信息采集"这一类（调参已经说明不是瓶颈，不再列为局限）
    # 表格单元格不解析 HTML，<b>/&gt;/<br> 一律不能进，否则会被当字面文字显示
    # 第三条按实测重写：10 个分箱最大偏差仅 %+.2f 个百分点，整体校准并不差，
    # 真正的弱点是高分段样本太少——最上面两箱合起来只占测试集 %.1f%%，
    # 人少的那一段概率估得最不稳，所以不能对外承诺「概率高于 X 就拒」。
    _cal = CALIB_FIG[CALIB_FIG['模型'] == 'XGBoost_调优后_第5阶段'].copy()
    _cal['_abs'] = _cal['偏差'].abs()
    _worst = _cal.loc[_cal['_abs'].idxmax()]
    _hi = _cal[_cal['分箱'] >= 8]
    lim = pd.DataFrame({
        '局限': ['未来数据上掉点明显', '匿名字段占比不低',
                 '高风险段的概率最不稳', '只验证过一次切分'],
        '实测表现': [
            '同批次 AUC %.4f → 未来数据 %.4f（掉 %.4f）'
            % (FINAL['测试 auc'], _oot_auc, float(FINAL['测试 auc']) - _oot_auc),
            '前 20 个影响因素里有 %d 项是 n0~n14 匿名字段' % _anon_top20,
            '偏差最大的是第 %d 箱（预测概率 %.4f），差 %+.2f 个百分点；'
            '最上面两箱合计 %d 笔样本，占测试集 %.1f%%'
            % (int(_worst['分箱']), float(_worst['平均预测概率']),
               float(_worst['偏差']) * 100, int(_hi['样本数'].sum()),
               int(_hi['样本数'].sum()) / float(_cal['样本数'].sum()) * 100),
            '五折 CV 折间标准差 %.4f、三种子极差 %.4f，未做独立样本验证'
            % (float(CV_R.iloc[0]['fold_std']), BEST_SEED_SPREAD)],
        '对业务的实际影响': [
            '客户结构随时间变化，建议每 6~12 个月用新一批数据重训，'
            '并在每次放款批次上重算 PSI',
            '这些字段模型在用，但没法向客户解释"因为你这项高所以拒"，'
            '建议定位为辅助项而非决策项',
            '高概率段的分数要打折看，审批规则不建议直接写"概率高于 X"，'
            '改用排序方式更稳',
            '成绩可能存在乐观偏差，上线首月的实际表现才是最终标尺'],
    })
    st.dataframe(lim, width='stretch', hide_index=True, height='auto')
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- ③ 提升路径 ----
    st.markdown('## 想更准，往哪里投入')
    st.markdown(
        '<div class="note">按<b>预期收益 / 实施难度</b>排序，'
        '下面是四条可行的路：</div>', unsafe_allow_html=True)
    _top5 = SHAP.nlargest(5, 'mean_abs_shap')
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    path = pd.DataFrame({
        '投入方向': ['① 补齐匿名字段的业务含义', '② 接入还款行为与账户数据',
                  '③ 加入申请到放款之间的过程变量', '④ 换模型或做模型融合'],
        '为什么值得做': [
            '前 20 个影响因素里有 %d 项是匿名字段，'
            '它们现在能参与建模但无法解释；'
            '弄清含义后既能提高可解释性，也可能通过重编码再涨一点准确度'
            % _anon_top20,
            '当前 %d 个特征几乎全是申请时点的静态信息'
            '（收入、负债率、信用分、职位）。'
            '银行手上通常还有放款后的还款表现——'
            '这类信息对预测违约最直接，但本项目数据里没有',
            '利率、期限这类定价变量已经在用，'
            '但审批过程本身的信息（人工改过的利率、补充提交的材料、'
            '审批耗时）没有进模型；这类变量往往反映审批员当时的判断',
            '本项目只用了 XGBoost。'
            '同框架下的 LightGBM、以及两者融合通常还能再挤出千分之二到三；'
            '但这部分收益有限，且增加维护成本'],
        '预期收益': ['中（主要提升可解释性）', '高', '中', '低到中'],
        '实施难度': ['低——找数据方补字段说明即可',
                 '高——需要跨系统取数与合规评估',
                 '中——需要改造审批流程埋点',
                 '低——现有脚本可直接跑'],
    })
    head(st, '四条提升路径的收益与难度',
         '09_shap_global_random.csv · 97_peak_round_choice.csv')
    st.dataframe(path, width='stretch', hide_index=True, height='auto')
    st.markdown(
        '<div class="note"><b>如果只能做一件事：</b>优先做第②条。'
        '当前模型的输入全部是"客户自己填的 + 征信报告上的"，'
        '都是<b>静态快照</b>；'
        '而违约的本质是"还款能力与意愿在未来 3~5 年里的变化"，'
        '这类动态信息缺失，'
        '是天花板到 %.4f 附近就难再上的主要原因。<br>'
        '<b>如果只能做一件事但不能加数据源：</b>那就先把监控建起来——'
        '按月重算 PSI、按季重训、用最新一批放款数据持续回测，'
        '<b>保证这 %.4f 的准确度不退化</b>，'
        '这比再花时间调参有价值得多。</div>'
        % (float(FINAL['测试 auc']), FINAL['测试 auc']),
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
if SRC_LOG:
    seen, uniq = set(), []
    for t in SRC_LOG:
        f = t.split('（')[0].strip()
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    st.markdown(
        '<details style="margin-top:26px">'
        '<summary style="font-size:14px;color:#7A8090;cursor:pointer;'
        'user-select:none">本页数据来源（%d 个结果表，鼠标悬停图表旁 ⓘ 可单看）</summary>'
        '<div style="font-size:14px;color:#6C7484;line-height:1.9;'
        'background:#F0F2F6;border-radius:8px;padding:12px 16px;margin-top:8px">'
        '全部结果表位于 <code>outputs/tables/</code>：%s</div>'
        '</details>' % (len(uniq), '、'.join('`%s`' % u for u in uniq)),
        unsafe_allow_html=True)
