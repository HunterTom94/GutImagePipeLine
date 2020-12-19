import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['text.usetex'] = False
rcParams['svg.fonttype'] = 'none'
import seaborn as sns

input_folder = 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\Pros\\'
raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')
organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')

filtered_raw = raw_organized[raw_organized['ROI_index'].isin(organized['ROI_index'])]

bins = np.arange(0,200,5)
cols = [col for col in raw_organized.columns if 'f0_mean' in col]

ax = plt.subplot()

non_filter_array = raw_organized[cols].values.flatten()
count, bin = np.histogram(non_filter_array, bins=bins)
norm_count = count/np.sum(count) * 100
center = (bin[:-1] + bin[1:]) / 2
ax.bar(center, norm_count, align='center', width=5, color='#ABEDD890', edgecolor='#80808080', linewidth=0.4)

filter_array = filtered_raw[cols].values.flatten()
count, bin = np.histogram(filter_array, bins=bins)
norm_count = count/np.sum(count) * 100
center = (bin[:-1] + bin[1:]) / 2
ax.bar(center, norm_count, align='center', width=5, color='#edb6ab90', edgecolor='#80808080', linewidth=0.4)



ax.set_xlabel('Baseline F Value')
ax.set_ylabel('Percentage')
plt.savefig('Z:\\#Gut Imaging Manuscript\\V6\\baseline_dist_{}.svg'.format(len(raw_organized[cols].values.flatten())))
plt.show()
# exit()

ax1 = plt.subplot()
bins = np.arange(0,60,2)
non_filter_array = np.std(raw_organized[cols].values, axis=1).flatten()
count, bin = np.histogram(non_filter_array, bins=bins)
norm_count = count/np.sum(count) * 100
center = (bin[:-1] + bin[1:]) / 2
ax1.bar(center, norm_count, align='center', width=2, color='#ABEDD890', edgecolor='#808080', linewidth=0.4)

filter_array = np.std(filtered_raw[cols].values, axis=1).flatten()
count, bin = np.histogram(filter_array, bins=bins)
norm_count = count/np.sum(count) * 100
center = (bin[:-1] + bin[1:]) / 2
ax1.bar(center, norm_count, align='center', width=2, color='#edb6ab90', edgecolor='#808080', linewidth=0.4)

ax1.set_xlabel('Standard Deviation of Baseline F Values for Each Cell')
ax1.set_ylabel('Percentage')
plt.savefig('Z:\\#Gut Imaging Manuscript\\V6\\std_{}.svg'.format(raw_organized.shape[0]))
plt.show()