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

train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros1\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros2\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros3\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros4\\')

def prepare_data(input_folder, stimulus):
    organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
    raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')
    organized = organized[~organized["ROI_index"].str.contains('_b')]
    scheme = pd.read_excel(input_folder + 'DeliverySchemes.xlsx')

    sti_df = pd.DataFrame()
    for sample in raw_organized['sample_index'].unique():
        org_copy = organized.copy()
        org_copy = org_copy[org_copy['sample_index'] == sample]

        temp_sti_df = pd.DataFrame()
        temp_sti_df['ROI_index'] = org_copy['ROI_index']
        temp_sti_df['trace'] = org_copy['{}_trace'.format(stimulus)]
        temp_sti_df['drv_peak'] = org_copy['{}_max_first_derivative'.format(stimulus)]
        temp_sti_df['f0_mean'] = org_copy['{}_f0_mean'.format(stimulus)]
        temp_sti_df['f0_std'] = org_copy['{}_f0_std'.format(stimulus)]
        temp_sti_df['response'] = org_copy['{}_average_response'.format(stimulus)]


        temp_sti_df['average_response'] = org_copy['{}_average_response'.format(stimulus)]

        sti_df = sti_df.append(temp_sti_df, ignore_index=True)

    return sti_df

for sti in ['KCl']:
    for train_folder in train_folder_ls:
        sti_df = prepare_data(train_folder, sti)

    ax = plt.subplot()

    # sns.scatterplot(x="peak", y="peak_index", data=sti_df)

    g = sns.jointplot(x="peak", y="peak_index", data=sti_df, kind="kde", color="m")
    plt.show()

    # g.plot_joint(plt.scatter, c="w", s=1, linewidth=1, marker="+")
    # g.ax_joint.collections[0].set_alpha(0)
    # g.set_axis_labels("$peak$", "$peak_index$")

    # sns.distplot(sti_df['peak'], bins=np.arange(0, 26, 0.1), ax=ax)
    X = np.concatenate((sti_df['peak'].values.reshape(-1, 1), sti_df['peak_index'].values.reshape(-1, 1)),axis=1)
    mixture = GaussianMixture(n_components=2).fit(X)
    means_hat = mixture.means_
    weights_hat = mixture.weights_
    sds_hat = np.sqrt(mixture.covariances_)

    df = pd.DataFrame(mixture.sample(1000)[0], columns=['peak','peak_index'])

    sns.jointplot(x="peak", y="peak_index", data=df, kind="kde");
    plt.show()

    mixture = GaussianMixture(n_components=2).fit(X[:,0].reshape(-1, 1))
    means_hat = mixture.means_
    weights_hat = mixture.weights_
    sds_hat = np.sqrt(mixture.covariances_)
    sns.distplot(mixture.sample(10000)[0],hist=False)
    plt.show()

    print(mixture.converged_)
    print(means_hat)
    print(sds_hat)
    print(weights_hat)
    #
    # sns.distplot(samples, bins=np.arange(0, 26, 0.1), ax=ax, hist=False)
    # ax.set_title(sti)
    # plt.savefig('Z:\\Christina\\FromYinan\\KCl_dist\\{}_DFF_Peak_distribution.pdf'.format(sti))
    plt.show()
exit()