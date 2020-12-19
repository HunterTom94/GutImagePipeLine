import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sys import exit
from matplotlib.patches import Rectangle
from util import low_pass_filter
import itertools
from sklearn import linear_model
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.gridspec as gridspec

def low_pass(x, af, fc, n, b):
    y = af/np.sqrt(1+np.power(x/fc, 2*n)) + b
    return y

kcl_resp = 1
# non_resp = 1
third = 1

# sti_ls = ['100mM EAA', '100mM NEAA']
sti_ls = ['KCl']
# sti_ls = ['2.5mM', '10mM','25mM', '100mM', '150mM', '10mM2']
#
# input_folder = 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\Dh31_Dosage4\\'
# input_folder = 'D:\\#Yinan\\untitled folder\\'
input_folder = 'D:\\#Yinan\\KCl_fit_test4\\analysis\\'
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

sti_df = pd.DataFrame()
temp_sti_df = pd.DataFrame()
temp_sti_df['ROI_index'] = organized['ROI_index']
temp_sti_df['df'] = raw_organized['{}_average_peak'.format('KCl')] - raw_organized['{}_f0_mean'.format('KCl')]
temp_sti_df['basal'] = raw_organized['{}_f0_mean'.format('KCl')]
temp_sti_df['average_response'] = organized['{}_average_response'.format('KCl')]
temp_sti_df['n_mean_basal'] = temp_sti_df['basal'] / temp_sti_df['basal'].mean()
temp_sti_df['std_n_mean_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].mean()) / temp_sti_df[
    'basal'].std()
temp_sti_df['std_n_median_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].median()) / temp_sti_df[
    'basal'].std()
sti_df = sti_df.append(temp_sti_df, ignore_index=True)

fig, ax = plt.subplots(1)
sns.scatterplot(x='std_n_mean_basal', y='std_n_median_basal', data=sti_df, linewidth=0, s=4, ax=ax, hue='average_response')
df = pd.DataFrame()
ax.set_xlim([0, 8])
ax.set_ylim([0, 8])

plt.savefig(input_folder + 'nstd_KCl_sti_filter_scatter.pdf')