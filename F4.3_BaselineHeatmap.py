import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sys import exit
from matplotlib.patches import Rectangle
from util import plot_colorbar
from os import listdir
from os.path import isfile, join, isdir

np.random.seed(0)
root_folder = 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\OA_UMAP\\'

org = pd.read_pickle(root_folder + 'Raw_OA_UMAP_organized_data.pkl')
scheme = pd.read_excel(root_folder + '\\DeliverySchemes.xlsx')

temp_baseline = np.empty((org.shape[0], 0))
for sti_ind in range(scheme.shape[0]):
    baseline_length = scheme.iloc[sti_ind]['stimulus_start'] - scheme.iloc[sti_ind]['video_start']
    stimulus_np = np.concatenate(org['{}_trace'.format(scheme.iloc[sti_ind]['stimulation'])].to_numpy()).reshape(org.shape[0],-1)
    temp_baseline = np.concatenate((temp_baseline, stimulus_np[:,:baseline_length]), axis=1)
    # temp_baseline[,:] = stimulus_np[:baseline_length]

def find_fluctuation(array):
    return np.max(array) - np.min(array) > 60

def find_flat(array):
    return np.max(array) - np.min(array) < 30 and np.max(array) < 50

range = [-0.5,5]
a = np.apply_along_axis(find_fluctuation, 1, temp_baseline)
# a = np.apply_along_axis(find_flat, 1, temp_baseline)
heatmap = temp_baseline[a,:][np.random.randint(np.sum(a), size=100), :]
print(np.sum(a))
print(np.min(heatmap))
print(np.max(heatmap))
ax = plt.subplot()
colorbar = plot_colorbar(ax, hue_nrom=range, cmap='jet', orientation='vertical')
plt.savefig('colorbar.svg')
# ax = plt.subplot()
# ax.imshow(temp_baseline[a, :], cmap='jet', aspect='auto', interpolation='none')
# ax.imshow(heatmap, cmap='jet', vmin=range[0], vmax=range[1], aspect='auto', interpolation='none')
# ax.set_yticks([20,40,60,80,100])
# ax.set_xlabel('frame index')
# ax.set_ylabel('cell index')
# plt.savefig('fluc_heatmap.svg')

exit()
