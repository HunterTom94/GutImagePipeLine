import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sys import exit
from matplotlib.patches import Rectangle
from util import low_pass_filter
import itertools
from sklearn import linear_model, svm
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from joblib import dump, load


#
#
def low_pass(x, af, fc, n, b):
    y = af / np.sqrt(1 + np.power(x / fc, 2 * n)) + b
    return y


throw = pd.read_pickle('D:\\#Yinan\\svm_test\\throw.pkl')
keep = pd.read_pickle('D:\\#Yinan\\svm_test\\keep.pkl')
organized = pd.read_pickle('D:\\#Yinan\\KCl_fit_test3_sti\\analysis\\analysis_organized_data.pkl')
keep.to_csv('D:\\#Yinan\\KCl_fit_test3_sti\\analysis\\keep.csv')
exit()
throw_org = organized.loc[organized['ROI_index'].isin(throw['ROI_index']),:]
keep_org = organized.loc[organized['ROI_index'].isin(keep['ROI_index']),:]

total_np = np.empty((len(organized['fine_region'].unique()), len(organized['sample_index'].unique())))
throw_np = np.empty((len(organized['fine_region'].unique()), len(organized['sample_index'].unique())))
keep_np = np.empty((len(organized['fine_region'].unique()), len(organized['sample_index'].unique())))

for sample_i, sample in enumerate(organized['sample_index'].unique()):
    for region_i, region in enumerate(organized['fine_region'].unique()):
        total_np[region_i, sample_i] = organized[
            (organized['fine_region'] == region) & (organized['sample_index'] == sample)].shape[0]
        throw_np[region_i, sample_i] = throw_org[
            (throw_org['fine_region'] == region) & (throw_org['sample_index'] == sample)].shape[0]
        keep_np[region_i, sample_i] = keep_org[
            (keep_org['fine_region'] == region) & (keep_org['sample_index'] == sample)].shape[0]

throw_perc = throw_np / total_np
keep_perc = keep_np / total_np

pd.DataFrame(total_np, columns=organized['sample_index'].unique(), index=organized['fine_region'].unique()).to_excel(
    'D:\\#Yinan\\KCl_fit_test3_sti\\analysis\\total.xlsx')
pd.DataFrame(throw_np, columns=organized['sample_index'].unique(), index=organized['fine_region'].unique()).to_excel(
    'D:\\#Yinan\\KCl_fit_test3_sti\\analysis\\throw.xlsx')
pd.DataFrame(keep_np, columns=organized['sample_index'].unique(), index=organized['fine_region'].unique()).to_excel(
    'D:\\#Yinan\\KCl_fit_test3_sti\\analysis\\keep.xlsx')
pd.DataFrame(throw_perc, columns=organized['sample_index'].unique(), index=organized['fine_region'].unique()).to_excel(
    'D:\\#Yinan\\KCl_fit_test3_sti\\analysis\\throw_perc.xlsx')
pd.DataFrame(keep_perc, columns=organized['sample_index'].unique(), index=organized['fine_region'].unique()).to_excel(
    'D:\\#Yinan\\KCl_fit_test3_sti\\analysis\\keep_perc.xlsx')

exit()

# df = pd.read_pickle('D:\\#Yinan\\svm_test\\name.pkl')
# print(df.shape)
# print(df.iloc[:,0].unique().shape)
# exit()

# df1 = pd.read_pickle('D:\\#Yinan\\svm_test\\1_name.pkl')
# df2 = pd.read_pickle('D:\\#Yinan\\svm_test\\2_name.pkl')
# df = df1.append(df2, ignore_index=True)
# print(df)
# df.to_pickle('D:\\#Yinan\\svm_test\\name.pkl')
# exit()

input_folder = 'D:\\#Yinan\\Dh31_AA\\'
scheme = pd.read_excel(input_folder + '\\DeliverySchemes.xlsx')

stimulus_ls = scheme['stimulation'].to_list()
frame_ls = scheme['video_end'].to_numpy() - scheme['video_start'].to_numpy() + 1
stimulus_length_ls = scheme['stimulus_end'].to_numpy() - scheme['stimulus_start'].to_numpy() + 1
time_stamp_ls = np.insert(np.cumsum(frame_ls), 0, 0)
organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')

organized = organized[~organized["ROI_index"].str.contains('_b')]

raw_organized = raw_organized[raw_organized['ROI_index'].isin(organized['ROI_index'])]
raw_organized = raw_organized.sort_values(by=['ROI_index'], ascending=True)
organized = organized.sort_values(by=['ROI_index'], ascending=True)
raw_organized.index = range(raw_organized.shape[0])
organized.index = range(organized.shape[0])
assert raw_organized.shape[0] == organized.shape[0]

sti_ls = [sti for sti in stimulus_ls if 'KCl' not in sti and 'AHL' not in sti]

sti_df = pd.DataFrame()
for sti in sti_ls:
    for sample in raw_organized['sample_index'].unique():
        raw_copy = raw_organized.copy()
        org_copy = organized.copy()
        raw_copy = raw_copy[raw_copy['sample_index'] == sample]
        org_copy = org_copy[org_copy['sample_index'] == sample]

        temp_sti_df = pd.DataFrame()
        temp_sti_df['ROI_index'] = org_copy['ROI_index']
        temp_sti_df['df'] = raw_copy['{}_average_peak'.format(sti)] - raw_copy['{}_f0_mean'.format(sti)]
        temp_sti_df['basal'] = raw_copy['{}_f0_mean'.format(sti)]
        temp_sti_df['average_response'] = org_copy['{}_average_response'.format(sti)]
        temp_sti_df['n_mean_basal'] = temp_sti_df['basal'] / temp_sti_df['basal'].mean()
        temp_sti_df['std_n_mean_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].mean()) / temp_sti_df[
            'basal'].std()
        temp_sti_df['std_n_median_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].median()) / temp_sti_df[
            'basal'].std()

        sti_df = sti_df.append(temp_sti_df, ignore_index=True)

X = np.hstack((sti_df['n_mean_basal'].to_numpy().reshape((-1, 1)),
               sti_df['std_n_mean_basal'].to_numpy().reshape((-1, 1)),
               sti_df['std_n_median_basal'].to_numpy().reshape((-1, 1))))
#
# X = np.hstack((sti_df['std_n_mean_basal'].to_numpy().reshape((-1, 1)),
#                    sti_df['std_n_median_basal'].to_numpy().reshape((-1, 1))))

y = sti_df['average_response'].to_numpy().flatten()

np.save('D:\\#Yinan\\svm_test\\sti_AA_x.npy', X)
np.save('D:\\#Yinan\\svm_test\\sti_AA_y.npy', y)

# pd.DataFrame(sti_df['ROI_index']).to_pickle('D:\\#Yinan\\svm_test\\name.pkl')
