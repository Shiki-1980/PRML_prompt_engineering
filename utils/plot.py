import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

# 1. 忽略无关警告
warnings.filterwarnings("ignore", category=UserWarning)

# ===========================
# 学术风格设置核心部分
# ===========================
# 使用更干净的基础风格
plt.style.use('seaborn-v0_8-whitegrid')

# 自定义学术配色方案 (Hex Codes)
# 这种深色系看起来更专业、稳重，且打印效果好
academic_palette = {
    'Baseline': '#4A708B',  # 深岩蓝：稳重的基础
    'Advanced': '#C0392B',  # 砖红色：醒目的提升 (强调色)
    'Code':     '#27AE60'   # 墨绿色：独特的类别
}

# 字体和线宽设置
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',     # 使用衬线字体 (如 Times New Roman)
    'axes.labelsize': 13,       # 坐标轴标签稍大
    'axes.titlesize': 14,       # 标题更大
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 1.2,      # 坐标轴线加粗
    'grid.alpha': 0.6           # 网格线稍微淡一点，不抢戏
})

# ===========================
# 数据准备
# ===========================
data = {
    'Strategy': ['Minimalist\nBaseline', 'Naive\nContext', 'CoT', 'Reflexion', 'SC (K=3)', 'Hybrid', 'PAL'],
    'Accuracy': [33.33, 13.33, 46.67, 53.33, 56.67, 50.00, 33.33],
    'Type': ['Baseline', 'Baseline', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Code']
}
df = pd.DataFrame(data)

# ===========================
# 绘图
# ===========================
plt.figure(figsize=(10, 6.5)) # 稍微增加一点高度

# 核心绘图函数
ax = sns.barplot(
    x='Strategy',
    y='Accuracy',
    data=df,
    hue='Type',                  # 指定分类依据
    palette=academic_palette,    # 使用自定义学术色板
    legend=False,                # 关闭图例 (因为X轴已经很清楚了)
    edgecolor='black',           # 给柱子加上黑色细边框，更有质感
    linewidth=0.8,
    width=0.7                    # 稍微调窄柱子宽度，显得更精致
)

# 标注数值
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.2f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', # 改为 bottom 让文字在柱子上方
                    fontsize=11, color='black',
                    xytext=(0, 3), # 稍微向上偏移
                    textcoords='offset points',
                    fontweight='medium') # 数字稍微加粗一点点

# ===========================
# 细节优化与美化
# ===========================
# 移除顶部和右侧的脊柱（边框），这是现代学术图表的常见做法
sns.despine(left=True, bottom=False) 
# 保持底部边框，移除左侧纵轴线（只保留网格）

plt.ylim(0, 65)
# 坐标轴标签加粗
plt.ylabel('Accuracy (%)', fontweight='bold', labelpad=10)
plt.xlabel('Prompting Strategy', fontweight='bold', labelpad=10)

# 标题
plt.title('Comparison of Accuracy across ARC Prompting Strategies (N=30)', 
          pad=20, fontweight='bold')

# 紧凑布局，防止遮挡
plt.tight_layout()

# 保存高分辨率图片
plt.savefig('accuracy_comparison_academic.png', dpi=300, bbox_inches='tight')
plt.show()