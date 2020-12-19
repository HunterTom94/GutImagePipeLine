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
input_folder = 'D:\\#Yinan\\untitled folder\\'
# input_folder = 'D:\\#Yinan\\KCl_fit_test\\analysis\\'
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

para_df = pd.DataFrame()
for sample in raw_organized['sample_index'].unique():
    raw_copy = raw_organized.copy()
    org_copy = organized.copy()
    raw_copy = raw_copy[raw_copy['sample_index'] == sample]
    org_copy = org_copy[org_copy['sample_index'] == sample]

    sti_df = pd.DataFrame()
    for i in range(len(sti_ls)):
        temp_sti_df = pd.DataFrame()
        temp_sti_df['ROI_index'] = org_copy['ROI_index']
        temp_sti_df['df'] = raw_copy['{}_average_peak'.format(sti_ls[i])] - raw_copy['{}_f0_mean'.format(sti_ls[i])]
        temp_sti_df['basal'] = raw_copy['{}_f0_mean'.format(sti_ls[i])]
        temp_sti_df['average_response'] = org_copy['{}_average_response'.format(sti_ls[i])]
        sti_df = sti_df.append(temp_sti_df, ignore_index=True)

    fig, ax = plt.subplots(1)
    sns.distplot(sti_df['basal'], kde=False, ax=ax, bins=np.arange(0, 250, 10))
    ax.set_xlim([0, 250])
    plt.savefig(input_folder + '{}_hist.png'.format(sample))
    plt.clf()

    f1_np = np.empty((0, 2))
    for f1_screen in np.linspace(sti_df['basal'].min(), temp_sti_df['basal'].max(), 500):
        temp_sti_df = sti_df.copy()
        tp = temp_sti_df[(temp_sti_df['basal'] <= f1_screen) &(temp_sti_df['average_response'] == 1)].shape[0]
        fp = temp_sti_df[(temp_sti_df['basal'] <= f1_screen) &(temp_sti_df['average_response'] == 0)].shape[0]
        fn = temp_sti_df[(temp_sti_df['basal'] > f1_screen) &(temp_sti_df['average_response'] == 1)].shape[0]
        if tp == 0:
            f1 = 0
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1 = 2*precision*recall/(precision+recall)
        f1_np = np.append(f1_np, np.array([f1_screen, f1]).reshape((1,2)), axis=0)
    f1_np[:,1] = low_pass_filter(f1_np[:,1],2, 0.05)

    sti_df = sti_df.sort_values(by=['basal'])
    x_name = 'basal'
    y_name = 'df'

    new_pts = np.empty((0, 2))
    bins = np.arange(0, np.max(sti_df[x_name]), 10)
    for ind in range(len(bins) - 1):
        temp_sampled = sti_df[[x_name, y_name]]
        temp_sampled = temp_sampled[(temp_sampled[x_name] >= bins[ind]) & (temp_sampled[x_name] < bins[ind + 1])]
        new_pts = np.append(new_pts,
                            np.array([np.mean([bins[ind], bins[ind + 1]]), np.mean(temp_sampled[y_name])]).reshape(
                                1, 2), axis=0)

    x_fit = new_pts[:,0]
    y_fit = new_pts[:,1]
    x_fit = x_fit[~np.isnan(y_fit)]
    y_fit = y_fit[~np.isnan(y_fit)]

    popt, pcov = curve_fit(low_pass, x_fit, y_fit, [50,70,2, 0], maxfev = 100000000, bounds=([0,10,1,-np.inf], [200, 200, np.inf, np.inf]))

    print(popt)
    lm_x = np.linspace(sti_df['basal'].min(), sti_df['basal'].max(), 100)
    lm_y = low_pass(lm_x, *popt)

    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=1, nrows=8, figure=fig)
    ax1 = fig.add_subplot(spec[0:5, 0])
    sns.scatterplot(x=x_name, y=y_name, data=sti_df, linewidth=0, s=4, ax=ax1, hue='average_response')
    plt.scatter(x_fit, y_fit, linewidth=0, s=8, c='red')
    df = pd.DataFrame()
    df['x'] = lm_x.flatten()
    df['y'] = lm_y.flatten()
    sns.lineplot(x='x', y='y', data=df, ax=ax1, linewidth=1, c='orange')
    ax1.vlines(x= f1_np[:,0][np.argmax(f1_np[:, 1])], ymin=-500, ymax=1000, colors='r', linestyles='--')
    plt.vlines(x= popt[1], ymin=-500, ymax=1000, colors='b', linestyles='--')
    plt.vlines(x= sti_df['basal'].mean(), ymin=-500, ymax=1000, colors='g', linestyles='--')
    ax1.set_xlim([0,200])
    ax1.set_ylim([-50, 250])

    ax2 = fig.add_subplot(spec[5:, 0])
    ax2.plot(f1_np[:,0], f1_np[:, 1])
    ax2.vlines(x= f1_np[:,0][np.argmax(f1_np[:, 1])], ymin=-500, ymax=1000, colors='r', linestyles='--')
    plt.vlines(x= popt[1], ymin=-500, ymax=1000, colors='b', linestyles='--')
    plt.vlines(x= sti_df['basal'].mean(), ymin=-500, ymax=1000, colors='g', linestyles='--')
    ax2.set_xlim([0,200])
    ax2.set_ylim([0, 1])
    plt.savefig(input_folder + '{}_KCl_sti_filter_scatter.pdf'.format(sample))
    para_df = para_df.append(
        {'mean': sti_df['basal'].mean(), 'median': sti_df['basal'].median(), 'std': sti_df['basal'].std(),
         'mean_n': (f1_np[:, 0][np.argmax(f1_np[:, 1])] - sti_df['basal'].mean()) / sti_df['basal'].std(),
         'median_n': (f1_np[:, 0][np.argmax(f1_np[:, 1])] - sti_df['basal'].median()) / sti_df['basal'].std(),
         'maxF1': f1_np[:, 0][np.argmax(f1_np[:, 1])], 'sample': str(sample)}, ignore_index=True)
    # n_ls.append((f1_np[:,0][np.argmax(f1_np[:, 1])] - sti_df['basal'].median())/sti_df['basal'].std())

    # exit()
# print(n_ls)
# para_df = para_df[['']]
para_df.to_excel(input_folder + 'n_para.xlsx')