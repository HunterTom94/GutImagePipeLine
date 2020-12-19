import numpy as np
from video_util import read_tif
from os import listdir
from os.path import isfile, join, isdir
from scipy.signal import correlate2d
from Igor_related_util import read_igor_roi_matrix
from matplotlib import pyplot as plt
from skimage import measure
import seaborn as sns


root_folder = 'D:\\Gut Imaging\\Videos\\cell_size\\'
files = listdir(root_folder)
cell_threshld = 0.2


video_folders = [folder for folder in listdir(root_folder) if isdir(join(root_folder, folder))]

sample_indices = []
for folder in video_folders:
    sample_indices.append(folder[1:7])

for sample in np.unique(sample_indices):
    sample_folders = [folder for folder in video_folders if sample in folder]
    ax = plt.subplot()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.1)
    roi_size_ls = []
    bbox_ls = []
    ROI = read_igor_roi_matrix(root_folder + sample_folders[0] + '\\roi.csv')
    ROI[ROI != 0] = 1
    labels = measure.label(ROI)
    reg_prop = measure.regionprops(labels)

    for i in range(len(reg_prop)):
        roi_size_ls.append(reg_prop[i].area)
        bbox_ls.append(reg_prop[i].bbox)

    files = listdir(root_folder+ folder)
    video_ls = [file for file in files if file.endswith('.tif')]
    assert len(video_ls) == 1
    raw_stack = read_tif(root_folder+ folder+'\\'+video_ls[0])
    max_proj = np.max(raw_stack,axis=0)

    cell_size_ls = []
    for bbox in bbox_ls:
        bbox_region = max_proj[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        bbox_region = bbox_region - np.min(bbox_region)
        ROI_bin = ROI[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        bbox_region_bin = np.zeros(bbox_region.shape)
        bbox_region_bin[bbox_region>np.max(bbox_region)*cell_threshld] = 1
        cell_size_ls.append(np.sum(bbox_region_bin))

    sns.distplot(roi_size_ls, kde=False, ax=ax, norm_hist=True)
    sns.distplot(cell_size_ls, kde=False, ax=ax, norm_hist=True)
    plt.savefig(root_folder + '{}.png'.format(sample))
    plt.show()