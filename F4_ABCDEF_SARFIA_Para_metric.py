import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['text.usetex'] = False
rcParams['svg.fonttype'] = 'none'
import pandas as pd
import seaborn as sns

# root_folder = 'C:\\Users\\hunte\\OneDrive\\Lab\\Gut Manucript\\figures\\#Gut Imaging Manuscript\\'
root_folder = 'Z:\\#Gut Imaging Manuscript\\Data\\F4\\F4_ABCDEF\\'

# t_params = ['f8t0.5', 'f8t1.5', 'f8t3']
# f_params = ['f5t1.5', 'f8t1.5', 'f15t1.5']

df = pd.read_excel(root_folder + 'SARFIA_metrics.xlsx')
df = df[df['Filter'] == 8]
# ax = sns.catplot(x="Threshold", y="Recall", kind="bar", data=df, ci=68,
#             palette={0.5: '#fffe9f', 1.5: '#fca180', 3: '#d92027'}, capsize=0.1,
#             edgecolor='#505050', linewidth=1)
ax = sns.catplot(x="Threshold", y="Precision", kind="bar", data=df, ci=68,
            palette={0.5: '#fffe9f', 1.5: '#fca180', 3: '#d92027'}, capsize=0.1,
            edgecolor='#505050', linewidth=1)
# ax = sns.catplot(x="Threshold", y="Merge Rate", kind="bar", data=df, ci=68,
#             palette={0.5: '#fffe9f', 1.5: '#fca180', 3: '#d92027'}, capsize=0.1,
#             edgecolor='#505050', linewidth=1)

df = df[df['Threshold'] == 1.5]
# ax = sns.catplot(x="Filter", y="Recall", kind="bar", data=df, ci=68,
#             palette={5: '#d9faff', 8: '#00bbf0', 15: '#005792'}, capsize=0.1,
#                  edgecolor='#505050', linewidth=1)
# ax = sns.catplot(x="Filter", y="Precision", kind="bar", data=df, ci=68,
#             palette={5: '#d9faff', 8: '#00bbf0', 15: '#005792'}, capsize=0.1,
#                  edgecolor='#505050', linewidth=1)
# ax = sns.catplot(x="Filter", y="Merge Rate", kind="bar", data=df, ci=68,
#             palette={5: '#d9faff', 8: '#00bbf0', 15: '#005792'}, capsize=0.1,
#                  edgecolor='#505050', linewidth=1)
ax.set(yticks=np.arange(0,1.1,0.2))
# ax.set(yticks=np.arange(0,0.16,0.03))
plt.savefig('Z:\\#Gut Imaging Manuscript\\V6\\ROI_param_Precision_T.svg', bbox_inches='tight')
plt.show()