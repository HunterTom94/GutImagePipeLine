import pandas as pd
import numpy as np
from sys import exit
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
from scipy import stats, signal
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['text.usetex'] = False
rcParams['svg.fonttype'] = 'none'
import matplotlib.gridspec as gridspec
from sklearn.mixture import GaussianMixture
from util import low_pass_filter
from sklearn.decomposition import PCA
import hdbscan


train_folder_ls = []
test_folder_ls = []


# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\AstC\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\CCHa2\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros1\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros2\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros3\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros4\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_controls\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_Dosage3\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_KCl2\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_KCl\\')

def pearson_correlation(x,y):
    return stats.pearsonr(x,y)[0]

def drv_pearson_correlation(x,y):
    x = np.gradient(low_pass_filter(x, 1, 0.05))
    return stats.pearsonr(x,y)[0]

def drv_position(x,y):
    x = np.gradient(low_pass_filter(x, 1, 0.05))
    return x[y]

def histogram(data, ax, bins, color, bin_size):
    count, bin = np.histogram(data, bins=bins)
    norm_count = count/np.sum(count) * 100
    center = (bin[:-1] + bin[1:]) / 2
    ax.bar(center, norm_count, align='center', width=bin_size, color=color, edgecolor='#808080', linewidth=0.4)

def prepare_data(input_folder):
    organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
    raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')
    organized = organized[~organized["ROI_index"].str.contains('_b')]
    raw_organized = raw_organized[raw_organized['ROI_index'].isin(organized['ROI_index'])]
    raw_organized = raw_organized.sort_values(by=['ROI_index'], ascending=True)
    organized = organized[organized['ROI_index'].isin(raw_organized['ROI_index'])]
    organized = organized.sort_values(by=['ROI_index'], ascending=True)
    raw_organized.index = range(raw_organized.shape[0])
    organized.index = range(organized.shape[0])
    assert raw_organized['ROI_index'].equals(organized['ROI_index'])


    sti_df = pd.DataFrame()
    for sample in raw_organized['sample_index'].unique():
        raw_copy = raw_organized.copy()
        org_copy = organized.copy()
        raw_copy = raw_copy[raw_copy['sample_index'] == sample]
        org_copy = org_copy[org_copy['sample_index'] == sample]

        temp_sti_df = pd.DataFrame()
        temp_sti_df['sample_index'] = raw_copy['sample_index']
        temp_sti_df['ROI_index'] = org_copy['ROI_index']
        temp_sti_df['trace'] = org_copy['KCl_trace']
        temp_sti_df['raw_trace'] = raw_copy['KCl_trace']
        temp_sti_df['peak'] = org_copy['{}_average_peak'.format('KCl')]
        temp_sti_df['peak_index'] = org_copy['KCl_average_peak_index']
        temp_sti_df['peak_individual'] = org_copy['{}_individual_peak'.format('KCl')]
        temp_sti_df['peak_index_individual'] = org_copy['KCl_individual_peak_index']

        temp_sti_df['drv_peak'] = org_copy['{}_max_first_derivative'.format('KCl')]
        temp_sti_df['drv_peak_ind'] = org_copy['{}_max_first_derivative_index'.format('KCl')]
        temp_sti_df['f0_mean'] = org_copy['KCl_f0_mean']
        temp_sti_df['raw_f0_mean'] = raw_copy['KCl_f0_mean']
        temp_sti_df['f0_std'] = org_copy['KCl_f0_std']
        temp_sti_df['response'] = org_copy['KCl_average_response']

        temp_sti_df['average_response'] = org_copy['{}_average_response'.format('KCl')]

        stimulus_np = np.concatenate(temp_sti_df['trace'].to_numpy()).reshape(temp_sti_df.shape[0],-1)
        if stimulus_np.shape[1] == 91:
            stimulus_np = stimulus_np[:, 0::4]
            # temp_sti_df['peak_index'] = (org_copy['KCl_average_peak_index']/4).astype(int)
            # temp_sti_df['peak_index_individual'] = (org_copy['KCl_individual_peak_index']/4).astype(int)
        ave_trace = np.mean(stimulus_np, axis=0)
        # plt.plot(ave_trace)
        # plt.show()
        ave_drv = np.gradient(low_pass_filter(ave_trace, 0.25, 0.05))
        ave_drv_peak_ind = np.argmax(ave_drv)
        temp_sti_df['peak_ave_drv_dff'] = stimulus_np[:, ave_drv_peak_ind]
        temp_sti_df['peak_ave_drv_drv'] = np.apply_along_axis(drv_position, 1, stimulus_np, ave_drv_peak_ind)
        temp_sti_df['pearson'] = np.apply_along_axis(pearson_correlation, 1, stimulus_np, ave_trace)
        temp_sti_df['drv_pearson'] = np.apply_along_axis(drv_pearson_correlation, 1, stimulus_np, ave_drv)
        temp_sti_df['ave_drv'] = [ave_drv]*temp_sti_df.shape[0]

        temp_sti_df['AUC'] = np.mean(stimulus_np, axis=1)

        sti_df = sti_df.append(temp_sti_df, ignore_index=True)

    return sti_df

sti_df = pd.DataFrame()
for train_folder in train_folder_ls:
    temp_sti_df = prepare_data(train_folder)
    sti_df = sti_df.append(temp_sti_df, ignore_index=True)

keep = sti_df[(sti_df['drv_pearson'] > 0) & (sti_df['AUC'] > 0)]
throw = sti_df[(sti_df['drv_pearson'] <= 0) | (sti_df['AUC'] <= 0)]

ax = plt.subplot()
bin_size = 0.1
bins = np.arange(-10,3,bin_size)
data = np.log(np.multiply(keep['AUC'].values, keep['drv_pearson'].values))
histogram(data=data, ax=ax, bins=bins, color='#ABEDD890', bin_size=bin_size)

mixture = GaussianMixture(n_components=2).fit(data.reshape(-1, 1))
means_hat = mixture.means_
sds_hat = np.sqrt(mixture.covariances_)
weights_hat = mixture.weights_

# g_ind = np.argmax(means_hat)
g_ind = np.argsort(means_hat.flatten())[::-1][0]
c = np.exp(means_hat[g_ind] - 1.5*sds_hat[g_ind][0])
print('exp: {}'.format(c))

histogram(data=mixture.sample(100000)[0].reshape(-1,1), ax=ax, bins=bins, color='#edb6ab90', bin_size=bin_size)
ax.axvline(x=np.log(c))
ax.set_xlabel('Log(product)')
ax.set_ylabel('Percentage')
print('log: {}'.format(np.log(c)))
plt.savefig('Z:\\#Gut Imaging Manuscript\\V6\\KCl_product_log_dist.svg')
plt.show()
print(means_hat)
print(sds_hat)
print(weights_hat)
# exit()
plt.clf()

ax = plt.subplot()
embedding = np.concatenate((keep['AUC'].values.reshape(-1, 1), keep['drv_pearson'].values.reshape(-1, 1)),axis=1)
clusterDf = pd.DataFrame(data = embedding, columns = ['AUC', 'drv_pearson'])
keep['label'] = keep['drv_pearson'] > c/keep['AUC']
throw['label'] = False
sti_df = pd.concat((keep, throw)).reset_index(drop=True)
# sti_df['label'] = sti_df['label']


x = np.arange(c, 10, 0.01)
y = c/(x)
clusterDf['label'] = clusterDf['drv_pearson'] > c/clusterDf['AUC']
print(clusterDf.shape[0])
clusterDf = clusterDf.sample(n=np.min([3000, clusterDf.shape[0]]), random_state=0)
print(np.mean(clusterDf['label'].astype(bool)))
sns.scatterplot(x='AUC', y='drv_pearson', data=clusterDf, hue='label', s=0.5, linewidth=0, ax=ax, alpha=1)
ax.plot(x,y, 'r--')
ax.set_xlim([0,6])
ax.set_ylim([0,1])
plt.savefig('Z:\\#Gut Imaging Manuscript\\V6\\KCl_product_scatter.svg')

plt.show()
plt.clf()

# sti_df = sti_df.sample(100, random_state=0)
# sti_df = sti_df[sti_df['peak'] < 3]
# sti_df = sti_df[sti_df['drv_peak'] > 0.3]
print(sti_df.shape[0])
# sti_df['pred'] = mixture.predict(np.concatenate((sti_df['peak'].values.reshape(-1, 1), sti_df['drv_peak_interval'].values.reshape(-1, 1)),axis=1))
sti_df = sti_df.sample(50, random_state=0)
sti_df = sti_df.sort_values(by=['peak'])
sti_df = sti_df.reset_index(drop=True)
# sti_df['pred'] = sti_df['pred'] == response_ind
# print(sti_df.pred)
# sti_df['label'] = np.logical_not(sti_df.pred.astype('bool')) & sti_df.average_response.astype('bool')
# sti_df['label'] = sti_df.pred & sti_df.average_response.astype('bool')
# sti_df['pearson_label'] = sti_df['pearson'] > 0.8
# sti_df['drv_label'] = sti_df['drv_peak'] > 0.04
# sti_df['label'] = sti_df.pearson_label & sti_df.drv_label
row_num = 2*(sti_df.shape[0]//10 + 1)

fig2 = plt.figure(constrained_layout=True, figsize=[10*2, row_num*2])
spec2 = gridspec.GridSpec(ncols=10, nrows=row_num, figure=fig2)

for row_index,row in sti_df.iterrows():
    sampling_freq = 0.25
    if len(row['trace']) == 91:
        trace = row['trace'][0::4]
        raw_trace = row['raw_trace'][0::4]
    else:
        trace = row['trace']
        raw_trace = row['raw_trace']
    peak_ind = row['peak_index']
    row_ind = int(row_index//10*2)
    col_ind = int(row_index%10)
    ax = fig2.add_subplot(spec2[row_ind, col_ind])
    if row['label']:
        color = '#ff9234'
        filter_color = '#d92027'
    else:
        color = '#4cbbb9'
        filter_color = '#0779e4'
    ax1 = ax.twinx()
    ax.plot(trace, color=color)
    ax1.plot(raw_trace, color='k')
    ax1.set_ylim([-0, 200])
    filtered = low_pass_filter(trace, sampling_freq, 0.05)
    ax.plot(filtered, color=filter_color)
    ax.set_ylim([-1,4])
    if row['AUC'] > 0 and row['drv_pearson'] > 0:
        log = (row['AUC'] * row['drv_pearson'])
    else:
        log = -999
    ax.set_title(
        '{}, {}, {}'.format(np.round(row['drv_pearson'], 2), np.round(row['AUC'], 2), np.round(log, 2)))
    ax.axvline(x=10, color='red', linewidth=1)

    ax = fig2.add_subplot(spec2[row_ind+1, col_ind])
    ax.plot(np.gradient(filtered), color='#2b580c')
    ax.plot(row['ave_drv'], color='k')
    ax.set_ylim([-0.3, 1])
    ax.set_yticks(np.arange(-0.3, 1.1,0.1))

plt.savefig('D:\\Gut Imaging\\Videos\\Dh31_svmTest2\\KCl_trace_peak_below_3_random_sample_20.svg')
plt.show()
