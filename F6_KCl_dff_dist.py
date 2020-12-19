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
import matplotlib.gridspec as gridspec
from sklearn.mixture import GaussianMixture
from util import low_pass_filter



train_folder_ls = []
test_folder_ls = []

# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\AstC\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\CCHa2\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros1\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros2\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros3\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros4\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_controls\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_Dosage3\\')

def prepare_data(input_folder):
    organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
    raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')
    organized = organized[~organized["ROI_index"].str.contains('_b')]


    sti_df = pd.DataFrame()
    for sample in raw_organized['sample_index'].unique():
        raw_copy = raw_organized.copy()
        org_copy = organized.copy()
        raw_copy = raw_copy[raw_copy['sample_index'] == sample]
        org_copy = org_copy[org_copy['sample_index'] == sample]

        temp_sti_df = pd.DataFrame()
        temp_sti_df['ROI_index'] = org_copy['ROI_index']
        temp_sti_df['trace'] = org_copy['KCl_trace']
        temp_sti_df['peak'] = org_copy['{}_average_peak'.format('KCl')]
        temp_sti_df['peak_index'] = org_copy['KCl_average_peak_index']
        # temp_sti_df['peak'] = org_copy['{}_max_first_derivative'.format('KCl')]
        # temp_sti_df['peak'] = org_copy['{}_max_first_derivative_index'.format('KCl')] / len(
        #     temp_sti_df['trace'].values[0])
        temp_sti_df['f0_mean'] = org_copy['KCl_f0_mean']
        temp_sti_df['f0_std'] = org_copy['KCl_f0_std']
        temp_sti_df['response'] = org_copy['KCl_average_response']


        temp_sti_df['average_response'] = org_copy['{}_average_response'.format('KCl')]

        sti_df = sti_df.append(temp_sti_df, ignore_index=True)

    return sti_df

sti_df = pd.DataFrame()
for train_folder in train_folder_ls:
    temp_sti_df = prepare_data(train_folder)
    sti_df = sti_df.append(temp_sti_df, ignore_index=True)
ax = plt.subplot()

sns.distplot(sti_df['peak'], bins=np.arange(-5, 26, 0.1), ax=ax)

mixture = GaussianMixture(n_components=2).fit(sti_df['peak'].values.reshape(-1, 1))
means_hat = mixture.means_.flatten()
weights_hat = mixture.weights_.flatten()
sds_hat = np.sqrt(mixture.covariances_).flatten()

print(mixture.converged_)
print(means_hat)
print(sds_hat)
print(weights_hat)

# sns.distplot(samples, bins=np.arange(0, 26, 0.1), ax=ax, hist=False)
# ax.set_xlim([sti_df['peak'].min(), sti_df['peak'].max()])
# plt.savefig('Z:\\Christina\\FromYinan\\KCl_dist\\DFF_Peak_distribution.pdf')
plt.show()
exit()
length_ls = []
for row_index,row in sti_df.iterrows():
    length_ls.append(len(row['trace']))

sti_df = sti_df[sti_df['peak'] < 3]
print(sti_df.shape[0])
sti_df =sti_df.sample(100, random_state=0)
sti_df = sti_df.sort_values(by=['peak'])
sti_df = sti_df.reset_index(drop=True)

row_num = 2*(sti_df.shape[0]//10 + 1)

fig2 = plt.figure(constrained_layout=True, figsize=[10*2, row_num*2])
spec2 = gridspec.GridSpec(ncols=10, nrows=row_num, figure=fig2)

for row_index,row in sti_df.iterrows():
    sampling_freq = 0.25
    if len(row['trace']) == 91:
        trace = row['trace'][0::4]
    else:
        trace = row['trace']
    row_ind = int(row_index//10*2)
    col_ind = int(row_index%10)
    ax = fig2.add_subplot(spec2[row_ind, col_ind])
    if row['response']:
        color = '#ff9234'
        filter_color = '#d92027'
    else:
        color = '#4cbbb9'
        filter_color = '#0779e4'
    ax.plot(trace, color=color)
    filtered = low_pass_filter(trace, sampling_freq, 0.05)
    ax.plot(signal.resample(filtered, 23), color=filter_color)
    ax.title.set_text('peak value = {}'.format(np.round(row['peak'], 2)))
    ax.set_ylim([-1,4])

    ax = fig2.add_subplot(spec2[row_ind+1, col_ind])
    ax.plot(np.diff(filtered), color='#2b580c')
    ax.set_ylim([-0.3, 1])
    ax.set_yticks(np.arange(-0.3, 1.1,0.1))
    # ax.axhline(y=row['f0_mean'] + 2 * row['f0_std'],color='green', linewidth=1)
    # ax.axhline(y=row['f0_mean'] + 3 * row['f0_std'],color='green', linewidth=1)
    # ax.axhline(y=row['f0_mean'] + 4 * row['f0_std'],color='green', linewidth=1)
    # ax.axvline(x=row['peak_index'],color='green', linewidth=1)
# plt.savefig('Z:\\Christina\\FromYinan\\KCl_dist\\KCl_trace_peak_below_3_random_sample_200.png')
plt.show()
