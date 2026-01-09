import matplotlib.pyplot as plt

# 模拟分析数据 (根据你的记录: SC对了17个，Hybrid对了15个) [cite: 58, 184]
# 其中必然存在部分 SC 正确但被 Hybrid 改错的情况
labels = ['SC Correct & Verified', 'Over-Correction\n(Wrongly Refined)', 'Refinement Failed\n(Initially Wrong)', 'Successful\nCorrection']
# 假设数据占比
sizes = [45, 10, 40, 5] 
colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99']

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
       explode=(0.05, 0.1, 0, 0), shadow=True)

plt.title('Hybrid Strategy Result Breakdown\n(Analyzing Performance Drop vs. SC)', pad=20)
plt.tight_layout()
plt.savefig('hybrid_error_analysis.png', dpi=300)
plt.show()