from skimage import measure, io
from Igor_related_util import read_igor_roi_matrix
import os
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

mean_pre = [35,36,44,32,36,41,40]
mean_post = [36,39,39,32,35,38,33]

median_pre = [35,36, 39,31, 36, 37, 40]
median_post = [37,38,38,32, 35,37, 35]

mode_pre = [35,34,38,32, 39, 34, 41]
mode_post = [35,39,38,36,36, 41, 40]

p_value = stats.ttest_rel(mode_pre, mode_post).pvalue

root_folder = 'D:\\Gut Imaging\\Videos\\ROI\\'
files = os.listdir(root_folder)

sample_indices = []
for file in files:
    if file.split('.')[1] == 'csv':
        sample_indices.append(file.split('_')[1].split('.')[0])

for sample in np.unique(sample_indices):
    sample_files = [file for file in files if (sample in file) and (file.split('.')[1] == 'csv')]
    ax = plt.subplot()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.1)
    for file in sample_files:
        cell_size_ls = []
        ROI = read_igor_roi_matrix(root_folder+file)
        ROI[ROI!=0] = 1
        labels = measure.label(ROI)
        reg_prop = measure.regionprops(labels)

        for i in range(len(reg_prop)):
            cell_size_ls.append(reg_prop[i].area)

        # cell_size_ls = np.array(cell_size_ls)
        cell_size_median = int(np.median(cell_size_ls))
        cell_size_mean = int(np.mean(cell_size_ls))
        cell_size_mode = stats.mode(cell_size_ls).mode[0]

        sns.distplot(cell_size_ls, kde=True, norm_hist=True, ax=ax)
        if 'pre' in file:
            ax.text(100, 0.09, str('PreMC'))
            ax.text(100, 0.085, str('mean: {}'.format(cell_size_mean)))
            ax.text(100, 0.08, str('median: {}'.format(cell_size_median)))
            ax.text(100, 0.075, str('mode: {}'.format(cell_size_mode)))
            ax.text(100, 0.07, str('total: {}'.format(len(cell_size_ls))))
        else:
            ax.text(150, 0.09, str('AfterMC'))
            ax.text(150, 0.085, str('mean: {}'.format(cell_size_mean)))
            ax.text(150, 0.08, str('median: {}'.format(cell_size_median)))
            ax.text(150, 0.075, str('mode: {}'.format(cell_size_mode)))
            ax.text(150, 0.07, str('total: {}'.format(len(cell_size_ls))))
    plt.savefig(root_folder+'cell_size_distribution_{}'.format(file.split('.')[0].split('_')[1]))
    plt.show()