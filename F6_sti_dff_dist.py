import pandas as pd
import numpy as np
from sys import exit
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.mixture import GaussianMixture
from itertools import starmap

#

train_folder_ls = []
test_folder_ls = []

train_folder_ls.append('D:\\Gut Imaging\\Videos\\Temp_UMAP\\Pros\\')

def prepare_data(input_folder, stimulus):
    organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
    raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')
    organized = organized[~organized["ROI_index"].str.contains('_b')]

    sti_df = pd.DataFrame()
    for sample in raw_organized['sample_index'].unique():
        org_copy = organized.copy()
        org_copy = org_copy[org_copy['sample_index'] == sample]

        temp_sti_df = pd.DataFrame()
        temp_sti_df['ROI_index'] = org_copy['ROI_index']
        temp_sti_df['trace'] = org_copy['{}_trace'.format(stimulus)]
        temp_sti_df['peak'] = org_copy['{}_average_peak'.format(stimulus)]
        temp_sti_df['peak_index'] = org_copy['{}_average_peak_index'.format(stimulus)]

        temp_sti_df['f0_mean'] = org_copy['{}_f0_mean'.format(stimulus)]
        temp_sti_df['f0_std'] = org_copy['{}_f0_std'.format(stimulus)]
        temp_sti_df['response'] = org_copy['{}_average_response'.format(stimulus)]


        temp_sti_df['average_response'] = org_copy['{}_average_response'.format(stimulus)]

        sti_df = sti_df.append(temp_sti_df, ignore_index=True)

    return sti_df

for sti in ['AHL','BCAA','EAA', 'LCFA', 'MCFA', 'MoSug', 'NEAA', 'SCFA']:
# for sti in ['KCl']:
    for train_folder in train_folder_ls:
        sti_df = prepare_data(train_folder, sti)

    ax = plt.subplot()

    sns.distplot(sti_df['peak'], bins=np.arange(0, 26, 0.1), ax=ax)

    mixture = GaussianMixture(n_components=2).fit(sti_df['peak'].values.reshape(-1, 1))
    means_hat = mixture.means_.flatten()
    weights_hat = mixture.weights_.flatten()
    sds_hat = np.sqrt(mixture.covariances_).flatten()
    #
    # print(mixture.converged_)
    # print(means_hat)
    # print(sds_hat)
    # print(weights_hat)
    #
    sns.distplot(mixture.sample(10000)[0].reshape(-1,1), bins=np.arange(0, 26, 0.1), ax=ax, hist=False)
    ax.set_xlim([sti_df['peak'].min(), sti_df['peak'].max()])
    ax.set_title('{}_mean_{}_std_{}_weight{}'.format(sti, np.round(means_hat, 1), np.round(sds_hat, 1), np.round(weights_hat, 2)))
    # plt.savefig('Z:\\Christina\\FromYinan\\KCl_dist\\{}_DFF_Peak_distribution.pdf'.format(sti))
    plt.show()
exit()