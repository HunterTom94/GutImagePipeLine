from video_util import read_tif
from os import listdir
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from os.path import isfile, join, isdir
import pandas as pd
import seaborn as sns

root_folder = 'D:\\Gut Imaging\\Videos\\ParaTest\\2_SARFIA_paraTest\\'

video_folders = [folder for folder in listdir(root_folder) if isdir(join(root_folder, folder))]

sample_indices = np.unique([ind.split('_')[0] for ind in video_folders])

# params = np.unique([ind.split('_')[1] for ind in video_folders])
# f_params = ['F5T1.5', 'F8T1.5', 'F15T1.5']
#
# df = pd.DataFrame(columns=['sample', 'Parameter', 'False Positive Rate'])
# ax = plt.subplot()
# for param in f_params:
#     for sample in sample_indices:
#         folder = 'D:\\Gut Imaging\\Videos\\ParaTest\\2_SARFIA_paraTest\\{}_{}\\'.format(sample, param)
#
#         AP = pd.read_csv(folder + 'segment.csv')
#         FP = np.sum(np.any(AP.isnull(), axis=1)) / AP.shape[0]
#         para_name = '{}={}'.format('Filter', param.split('T')[0][1:])
#         df = df.append({'sample': sample, 'Parameter (Threshold=1.5)': para_name, 'False Positive Rate': np.round(FP,2)}, ignore_index=True)
#
# ax = sns.catplot(x="Parameter (Threshold=1.5)", y="False Positive Rate", kind="bar", data=df, ci=68,
#             palette={'Filter=5': '#d9faff', 'Filter=8': '#00bbf0', 'Filter=15': '#005792'}, capsize=0.1,
#                  edgecolor='#505050', linewidth=1)
# ax.set(yticks=np.arange(0,0.25,0.05))
# plt.savefig(root_folder + 'ROI_param_FP_F.svg', bbox_inches='tight')
# plt.show()
# exit()

# t_params = ['F8T0.5', 'F8T1.5', 'F8T3']
t_params = ['F8T3']

# df = pd.DataFrame(columns=['sample', 'Parameter (F=8)', 'FP Rate'])

for param in t_params:
    for sample in sample_indices:
        print(sample)
        folder = 'D:\\Gut Imaging\\Videos\\ParaTest\\2_SARFIA_paraTest\\{}_{}\\'.format(sample, param)

        AP = pd.read_csv(folder + 'segment.csv')
        print(AP.shape[0] - np.sum(np.any(AP.isnull(), axis=1)))
        # FP = np.sum(np.any(AP.isnull(), axis=1)) / AP.shape[0]
        # para_name = '{}={}'.format('Threshold', param.split('8')[1][1:])
        # df = df.append({'sample': sample, 'Parameter (Filter=8)': para_name, 'False Positive Rate': FP}, ignore_index=True)

# sns.catplot(x="Parameter (Filter=8)", y="False Positive Rate", kind="bar", data=df, ci=68,
#             palette={'Threshold=0.5': '#fffe9f', 'Threshold=1.5': '#fca180', 'Threshold=3': '#d92027'}, capsize=0.1,
#             edgecolor='#505050', linewidth=1)
# plt.savefig(root_folder + 'ROI_param_FP_T.svg', bbox_inches='tight')
# plt.show()