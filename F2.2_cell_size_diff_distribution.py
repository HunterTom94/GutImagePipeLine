import numpy as np
from video_util import read_tif
from os import listdir
from os.path import isfile, join, isdir
from scipy.signal import correlate2d
from Igor_related_util import read_igor_roi_matrix
from matplotlib import pyplot as plt
from skimage import measure
import seaborn as sns
import pandas as pd


root_folder = 'D:\\Gut Imaging\\Videos\\ParaTest\\2_SARFIA_paraTest\\'

video_folders = [folder for folder in listdir(root_folder) if isdir(join(root_folder, folder))]

sample_indices = np.unique([ind.split('_')[0] for ind in video_folders])

ROI_size = 10
cell_threshld = 0.2

# f_params = ['F5T1.5', 'F8T1.5', 'F15T1.5']
#
# df = pd.DataFrame(columns=['sample', 'Parameter (T=1.5)', 'Median'])
# ax = plt.subplot()
# for param in f_params:
#     for sample in sample_indices:
#         folder = 'D:\\Gut Imaging\\Videos\\ParaTest\\2_SARFIA_paraTest\\{}_{}\\'.format(sample, param)
#         roi_size_ls = []
#         centroid_ls = []
#         ROI = read_igor_roi_matrix(folder + 'roi.csv')
#         ROI[ROI != 0] = 1
#         labels = measure.label(ROI)
#         reg_prop = measure.regionprops(labels)
#
#         for i in range(len(reg_prop)):
#             roi_size_ls.append(reg_prop[i].area)
#             centroid_ls.append(reg_prop[i].centroid)
#
#         files = listdir(folder)
#         video_ls = [file for file in files if file.endswith('.tif')]
#         assert len(video_ls) == 1
#         raw_stack = read_tif(folder + '\\' + video_ls[0])
#         max_proj = np.max(raw_stack, axis=0)
#
#         diff_ls = []
#         for centroid in centroid_ls:
#             centroid = np.array(centroid).astype(int)
#
#             bbox_region = max_proj[
#                           np.max([0, centroid[0] - ROI_size]):np.min([max_proj.shape[0], centroid[0] + ROI_size]),
#                           np.max([0, centroid[1] - ROI_size]):np.min([max_proj.shape[1], centroid[1] + ROI_size])]
#             bbox_region = bbox_region - np.min(bbox_region)
#             ROI_bin = ROI[centroid[0] - ROI_size:centroid[0] + ROI_size, centroid[1] - ROI_size:centroid[1] + ROI_size]
#             bbox_region_bin = np.zeros(bbox_region.shape)
#             bbox_region_bin[bbox_region > np.max(bbox_region) * cell_threshld] = 1
#             diff_ls.append(np.sum(ROI_bin) - np.sum(bbox_region_bin))
#
#         para_name = '{}={}'.format('Filter', param.split('T')[0][1:])
#         df = df.append({'sample': sample, 'Parameter (Threshold=1.5)': para_name, 'Median': np.median(diff_ls)}, ignore_index=True)
#
# g = sns.catplot(x="Parameter (Threshold=1.5)", y="Median", kind="bar", data=df, ci=68,
#             palette={'Filter=5': '#d9faff', 'Filter=8': '#00bbf0', 'Filter=15': '#005792'}, capsize=0.1,
#                 edgecolor='#505050', linewidth=1)
# ax = g.axes[0,0]
# ax.spines['bottom'].set_position('zero')
# plt.savefig(root_folder + 'ROI_param_median_F.svg', bbox_inches='tight')
# plt.show()
# exit()

t_params = ['F8T0.5', 'F8T1.5', 'F8T3']

df = pd.DataFrame(columns=['sample', 'Parameter (F=8)', 'Median'])

for param in t_params:
    for sample in sample_indices:
        folder = 'D:\\Gut Imaging\\Videos\\ParaTest\\2_SARFIA_paraTest\\{}_{}\\'.format(sample, param)
        roi_size_ls = []
        centroid_ls = []
        ROI = read_igor_roi_matrix(folder + 'roi.csv')
        ROI[ROI != 0] = 1
        labels = measure.label(ROI)
        reg_prop = measure.regionprops(labels)

        for i in range(len(reg_prop)):
            roi_size_ls.append(reg_prop[i].area)
            centroid_ls.append(reg_prop[i].centroid)

        files = listdir(folder)
        video_ls = [file for file in files if file.endswith('.tif')]
        assert len(video_ls) == 1
        raw_stack = read_tif(folder + '\\' + video_ls[0])
        max_proj = np.max(raw_stack, axis=0)

        diff_ls = []
        for centroid in centroid_ls:
            centroid = np.array(centroid).astype(int)

            bbox_region = max_proj[
                          np.max([0, centroid[0] - ROI_size]):np.min([max_proj.shape[0], centroid[0] + ROI_size]),
                          np.max([0, centroid[1] - ROI_size]):np.min([max_proj.shape[1], centroid[1] + ROI_size])]
            bbox_region = bbox_region - np.min(bbox_region)
            ROI_bin = ROI[centroid[0] - ROI_size:centroid[0] + ROI_size, centroid[1] - ROI_size:centroid[1] + ROI_size]
            bbox_region_bin = np.zeros(bbox_region.shape)
            bbox_region_bin[bbox_region > np.max(bbox_region) * cell_threshld] = 1
            diff_ls.append(np.sum(ROI_bin) - np.sum(bbox_region_bin))

        para_name = '{}={}'.format('Threshold', param.split('8')[1][1:])
        df = df.append({'sample': sample, 'Parameter (Filter=8)': para_name, 'Median': np.median(diff_ls)}, ignore_index=True)

g = sns.catplot(x="Parameter (Filter=8)", y="Median", kind="bar", data=df, ci=68,
            palette={'Threshold=0.5': '#fffe9f', 'Threshold=1.5': '#fca180', 'Threshold=3': '#d92027'}, capsize=0.1,
                edgecolor='#505050', linewidth=1)
ax = g.axes[0,0]
ax.spines['bottom'].set_position('zero')
ax.set(yticks=np.arange(-80,15,10))
# ax.set(xticks=[])
plt.savefig(root_folder + 'ROI_param_median_T.svg', bbox_inches='tight')
plt.show()