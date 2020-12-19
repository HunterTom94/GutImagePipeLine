import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
from Igor_related_util import read_igor_roi_matrix
from skimage import measure
from video_util import read_tif
from scipy import stats
import pandas as pd
import seaborn as sns

root_folder = 'D:\\Gut Imaging\\Videos\\BackgrounSub\\'
videos = [file for file in listdir(root_folder) if isfile(root_folder + file)]
ROI = read_igor_roi_matrix(root_folder + 'd3403\\roi.csv')

roi_size_ls = []
bbox_ls = []
ROI[ROI != 0] = 1
labels = measure.label(ROI)
reg_prop = measure.regionprops(labels)

for i in range(len(reg_prop)):
    roi_size_ls.append(reg_prop[i].area)
    bbox_ls.append(reg_prop[i].bbox)

df = pd.DataFrame(columns=['Ball Size', 'Background', 'Correlation Coef','Peak Prominence', 'ind'])
no_substract = []
for video in videos:
    if video == 'MAX_3403_999.tif':
        raw_stack = read_tif(root_folder + video)
        baseline_ls = []
        corr_ls = []
        peak_prominence_ls = []
        for ind, bbox in enumerate(bbox_ls):
            bbox_region = raw_stack[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            ROI_bin = ROI[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            masked_bbox_region = np.multiply(bbox_region, ROI_bin)
            no_substract.append(masked_bbox_region.flatten())
for video in videos:
    raw_stack = read_tif(root_folder + video)
    baseline_ls = []
    corr_ls = []
    peak_prominence_ls = []
    for ind, bbox in enumerate(bbox_ls):
        bbox_region = raw_stack[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        ROI_bin = ROI[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        masked_bbox_region = np.multiply(bbox_region, ROI_bin)
        corr_ls.append(stats.pearsonr(masked_bbox_region.flatten(), no_substract[ind])[0])
        invert_ROI_bin = np.invert(ROI_bin.astype(bool)).astype(int)
        invert_masked_bbox_region = np.multiply(bbox_region, invert_ROI_bin)
        baseline_ls.append(np.mean(invert_masked_bbox_region[np.nonzero(invert_masked_bbox_region)]))
        peak_prominence_ls.append(np.max(masked_bbox_region) - baseline_ls[-1])
        df = df.append({'Ball Size': int(video.split('_')[-1].split('.')[0]), 'Background': np.nanmean(baseline_ls),
                        'Correlation Coef': np.mean(corr_ls), 'Peak Prominence': np.nanmean(peak_prominence_ls), 'ind':ind}, ignore_index=True)

df = df.sort_values(by='Ball Size')

ax = plt.subplot()
sns.barplot(x='Ball Size',y='Background', data=df, ax=ax, facecolor=sns.color_palette()[0])
label_ls = [str(int(size)) for size in df['Ball Size'].unique()]
label_ls[-1] = 'No Subtraction'
ax.set_xticklabels(label_ls)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.savefig(root_folder+'New folder\\Background.svg')
plt.show()
