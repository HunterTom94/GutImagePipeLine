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



def sigmoid(x, x0, k, s, b):
    y = s / (1 + np.exp(-k * (x - x0))) + b
    return y

def low_pass(x, af, fc, n):
    y = af/np.sqrt(1+np.power(x/fc, 2*n))
    return y

kcl_resp = 1
# non_resp = 1
third = 1

# first_ls = ['ave','max','sec_max']
# second_ls = ['all','nonresp']
sti_ls = ['100mM EAA', '100mM NEAA']

# input_folder = 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\Dh31_Dosage4\\'
# input_folder = 'D:\\#Yinan\\untitled folder\\'
input_folder = 'D:\\#Yinan\\KCl_fit_test\\analysis\\'
scheme = pd.read_excel(input_folder + '\\DeliverySchemes.xlsx')

stimulus_ls = scheme['stimulation'].to_list()
frame_ls = scheme['video_end'].to_numpy() - scheme['video_start'].to_numpy() + 1
stimulus_length_ls = scheme['stimulus_end'].to_numpy() - scheme['stimulus_start'].to_numpy() + 1
time_stamp_ls = np.insert(np.cumsum(frame_ls), 0, 0)
organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')

organized = organized[~organized["ROI_index"].str.contains('_b')]
# organized = organized[organized['KCl_average_response'] == kcl_resp]

# organized = organized[organized['ROI_index'] == '291621c_23']

raw_organized = raw_organized[raw_organized['ROI_index'].isin(organized['ROI_index'])]
raw_organized = raw_organized.sort_values(by=['ROI_index'], ascending=True)
organized = organized.sort_values(by=['ROI_index'], ascending=True)
raw_organized.index = range(raw_organized.shape[0])
organized.index = range(organized.shape[0])
assert raw_organized.shape[0] == organized.shape[0]

organized = organized[(organized['KCl_average_response'] == 0) & (raw_organized['KCl_average_peak'] - raw_organized['KCl_f0_mean'] > 150)& (raw_organized['KCl_f0_mean'] < 25)]
raw_organized = raw_organized[(organized['KCl_average_response'] == 0) & (raw_organized['KCl_average_peak'] - raw_organized['KCl_f0_mean'] > 150)& (raw_organized['KCl_f0_mean'] < 25)]

max_trace_num = 200
col_num = 2
row_num = int(np.ceil(np.min([max_trace_num, raw_organized.shape[0]])/col_num))
fig = plt.figure(figsize=(col_num*4, row_num*2), constrained_layout=True)
grid = plt.GridSpec(row_num, col_num, wspace=0.3, hspace=0.5)

# for parameters in itertools.product([0,1,2], [0,1],[0,1]):
# first = parameters[0]
# second = parameters[1]
# third = parameters[2]
# def find_basal(first, second):
#     max_f0 = []
#     for roi_index in range(organized.shape[0]):
#         temp_max = []
#         for stimulus_ind, stimulus in enumerate(stimulus_ls):
#             if 'KCl' in stimulus:
#                 continue
#             if second == 0:
#                 temp_max.append(raw_organized.iloc[roi_index, :]['{}_f0_mean'.format(stimulus)])
#             elif second == 1:
#                 if organized.iloc[roi_index,:]['{}_average_response'.format(stimulus)] == 0:
#                     temp_max.append(raw_organized.iloc[roi_index, :]['{}_f0_mean'.format(stimulus)])
#         if first == 0:
#             max_f0.append(np.mean(temp_max))
#         elif first == 1:
#             max_f0.append(np.max(temp_max))
#         elif first == 2:
#             if len(temp_max) >= 2:
#                 max_f0.append(np.sort(temp_max)[-2])
#             elif len(temp_max) == 1:
#                 max_f0.append(temp_max[0])
#             else:
#                 max_f0.append(0)
#     return np.asarray(max_f0)
#
# def count_continuous(row):
#     count_ls = []
#     count = 0
#     for i in range(len(row) - 1):
#         if row[i] * row[i+1] > 0:
#             count += 1
#         else:
#             count_ls.append(count)
#             count = 0
#     count_ls.append(count)
#     return np.max(count_ls)
#
# f0_columns = [column for column in raw_organized.columns if 'f0_mean' in column and 'KCl' not in column]
# f0_np = raw_organized[f0_columns].to_numpy()
# diff_np = np.diff(f0_np)
#
# mono_count = np.apply_along_axis(count_continuous, 1, diff_np) + np.random.uniform(low=-0.4, high=0.4, size=(diff_np.shape[0],))
# abs_diff = np.sum(np.abs(diff_np), axis=1)

# df = pd.DataFrame(organized['KCl_average_response'])
# df['basal'] = max_f0
# df['abs'] = abs_diff
# df['mono'] = mono_count
# df = df[df['basal'] < 85]
# plt.figure()
# sns.scatterplot(x='abs', y='mono', data=df, hue='KCl_average_response', linewidth=0, s=3)
# plt.savefig(input_folder + 'scatter_test.pdf')
# exit()

sampled = organized
# sampled['df'] = raw_organized['{}_average_peak'.format(sti_ls[third])] - raw_organized['{}_f0_mean'.format(sti_ls[third])]
sampled['dff_all_trace'] = raw_organized['all_trace']
# sampled['basal_1'] = find_basal(0,0)
# sampled['basal_2'] = find_basal(1,0)
# sampled['mono'] = mono_count
# sampled['abs'] = abs_diff
# sampled['sti_peak'] = pd.DataFrame(organized['{}_average_peak'.format(sti_ls[third])])
# sampled['KCl_average_response'] = organized['KCl_average_response']
#
# average_peak_columns = [column for column in organized.columns if 'average_peak' in column and 'KCl' not in column and 'index' not in column and 'AHL' not in column and '10mM2' not in column]
# average_peak_np = organized[average_peak_columns].to_numpy()
#
# def if_drop(row):
#     temp_holder = np.empty((0,3))
#     for i in range(len(row) - 1)[1:]:
#         temp_holder = np.append(temp_holder, np.asarray([i+1, np.diff(row)[i], np.diff(row)[i]/row[i]]).reshape((1,-1)), axis=0)
#     return temp_holder[np.argmin(temp_holder[:, 1]), [0,2]]
# drop_np = np.apply_along_axis(if_drop, 1, average_peak_np)
#
# sampled['drop'] = drop_np[:,1]
# basal = []
# basal_3_peak = []
# for ind, i in enumerate(drop_np[:,0]):
#     basal.append(raw_organized['{}_f0_mean'.format(stimulus_ls[int(i)])].to_numpy()[ind])
#     basal_3_peak.append(raw_organized['{}_average_peak'.format(stimulus_ls[int(i)])].to_numpy()[ind])
# sampled['basal_3'] = basal
# sampled['basal_3_peak'] = basal_3_peak
# sampled['sti_ind'] = drop_np[:,0]
#
# sampled_copy = sampled.copy()
# slope_np = np.empty((0,2))
#
# # for basal_high in np.linspace(150,30,120):
# for basal_high in [100,70,50,30]:
#     sampled = sampled_copy.copy()
#     lm = linear_model.LinearRegression()
#
#     prev_dff = []
#     basal_3_high = []
#     for _, row in raw_organized.iterrows():
#         added=0
#         for stimulus_ind, stimulus in enumerate(sti_ls):
#             if row['{}_f0_mean'.format(stimulus)] > basal_high:
#                 basal_3_high.append(1)
#                 added = 1
#                 break
#         if not added:
#             basal_3_high.append(0)
#     sampled['basal_3_high'] = basal_3_high
#     sampled = sampled.sort_values(by=['basal_1'])
#
#     # sampled = sampled.loc[sampled[['{}_average_response'.format(stimulus) for stimulus in
#     #                                                               sti_ls if 'KCl' not in stimulus]].any(axis=1), :]
#     # sampled = sampled[sampled['basal_3_high'] == 0]
#
#
#     # x_name = 'basal_1'
#     # y_name = 'sti_peak'
#
#     x_name = 'basal_1'
#     y_name = 'df'
#
#     # x_name = 'basal_3'
#     # y_name = 'basal_3_peak'
#
#     new_pts = np.empty((0, 2))
#     bins = np.arange(0, 150, 10)
#     for ind in range(len(bins) - 1):
#         temp_sampled = sampled[[x_name, y_name]]
#         temp_sampled = temp_sampled[(temp_sampled[x_name] >= bins[ind]) & (temp_sampled[x_name] < bins[ind + 1])]
#         new_pts = np.append(new_pts,
#                             np.array([np.mean([bins[ind], bins[ind + 1]]), np.mean(temp_sampled[y_name])]).reshape(
#                                 1, 2), axis=0)
#
#     # x_fit = sampled[x_name].to_numpy()
#     # y_fit = sampled[y_name].to_numpy()
#     #
#     x_fit = new_pts[:,0]
#     y_fit = new_pts[:,1]
#     x_fit = x_fit[~np.isnan(y_fit)]
#     y_fit = y_fit[~np.isnan(y_fit)]
#
#     # degree = 1
#     # model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
#     #                   ('linear', LinearRegression(fit_intercept=True))])
#     # model = model.fit(x_fit, y_fit)
#     # print(model.named_steps['linear'].coef_)
#     # lm_x = np.linspace(np.min(x_fit), np.max(x_fit), 100).reshape((-1,1))
#     # lm_y = model.predict(lm_x)
#
#     popt, pcov = curve_fit(low_pass, x_fit, y_fit, [50,70,2], maxfev = 100000000)
#     print(popt)
#     lm_x = np.linspace(sampled['basal_1'].min(), sampled['basal_1'].max(), 100)
#     lm_y = low_pass(lm_x, *popt)
#
#     # slope_np = np.append(slope_np, np.array([basal_high, popt[1]]).reshape((1,2)), axis=0)
#     plt.figure()
#     ax = plt.subplot()
#     sns.scatterplot(x=x_name, y=y_name, data=sampled, linewidth=0, s=4, ax=ax, hue='basal_3_high')
#     plt.scatter(x_fit, y_fit, linewidth=0, s=8, c='red')
#     df = pd.DataFrame()
#     df['x'] = lm_x.flatten()
#     df['y'] = lm_y.flatten()
#     sns.lineplot(x='x', y='y', data=df, ax=ax, linewidth=1)
#     # ax.set_xlim([0,150])
#     # ax.set_ylim([-1, 11])
#     # plt.savefig(input_folder + '{}_degree_scatter.pdf'.format(degree))
#     plt.savefig(input_folder + 'filter_scatter.pdf')
#     # plt.savefig(input_folder + '{}_sigmoid_scatter.pdf'.format(basal_high))
#     exit()
#     # plt.clf()
# # plt.figure()
# # plt.plot(slope_np[:,0], slope_np[:,1])
# # plt.show()
# # print(slope_np)
# exit()
# for _, row in sampled.iterrows():
#     prev_dff.append(row['{}_average_peak'.format(stimulus_ls[int(row['sti_ind'] - 1)])])
# sampled['prev_dff'] = prev_dff
# # sampled = sampled[sampled['basal_1'] < 85]
# # sampled = sampled[sampled['drop'] > -0.2]
#
#
# sampled = sampled.sort_values(by=['drop'], ascending=True)
# total_num_after_filtered = np.sum(sampled[['{}_average_response'.format(stimulus) for stimulus in stimulus_ls if
#                                                 'KCl' not in stimulus and 'AHL' not in stimulus and '10mM2' not in stimulus]].any(axis=1))
# if total_num_after_filtered == 0:
#     total_num_after_filtered = 1
# # sampled = sampled[sampled['prev_dff'].abs() > 1]
# # sampled = sampled.iloc[:max_trace_num, :]
#
#
# # f0_columns = [column for column in raw_organized.columns if 'f0_mean' in column and 'KCl' not in column]
# # raw_f0 = raw_organized[f0_columns].std(axis=1).to_numpy()/raw_organized[f0_columns].mean(axis=1).to_numpy()
# # sampled = raw_organized.reindex(np.argsort(raw_f0)[::-1]).iloc[:row_num, :]
#
# # sampled = raw_organized.sample(np.min([50, raw_organized.shape[0]]), random_state=0)
#
# # plot_df = pd.DataFrame(organized['{}_average_peak'.format(sti_ls[third])])
# # plot_df['KCl_average_response'] = pd.DataFrame(organized['KCl_average_response'])
# # plot_df['basal'] = max_f0
# plt.figure()
# ax = plt.subplot()
# # # sns.scatterplot(x='basal', y='{}_average_peak'.format(sti_ls[third]), data=plot_df, hue='KCl_average_peak', linewidth=0, s=4, palette='jet', hue_norm=(-0.2, 2))
# sns.scatterplot(x=x_name, y=y_name, data=sampled, linewidth=0, s=4, ax=ax, hue='basal_3_high')
# lm_x = np.linspace(sampled['basal_1'].min(), sampled['basal_1'].max(), 1000).reshape((-1,1))
# lm_y = lm.predict(lm_x)
# df = pd.DataFrame()
# df['x'] = lm_x.flatten()
# df['y'] = lm_y.flatten()
# sns.lineplot(x='x', y='y', data=df, ax=ax, linewidth=1)
# # sns.scatterplot(x='basal_1', y='sti_peak', data=sampled, linewidth=0, s=4, ax=ax, hue='KCl_average_response')
# # plt.hlines(y=-0.3, xmin=sampled['basal_3'].min(), xmax=sampled['basal_3'].max(), color='r', linestyles='--', lw=0.5)
# # plt.hlines(y=-0.4, xmin=sampled['basal_3'].min(), xmax=sampled['basal_3'].max(), color='g', linestyles='--', lw=0.5)
# plt.savefig(input_folder + '{}_line_scatter.pdf'.format(basal_high))
# # plt.savefig(input_folder + '{}_{}_{}_scatter.pdf'.format(first_ls[first], second_ls[second], sti_ls[third]))
# exit()

# plot_df = pd.DataFrame(organized['KCl_average_response'])
# plot_df = pd.DataFrame(organized['KCl_average_peak'])
# plot_df['basal'] = max_f0
# plot_df['coef_var'] = raw_f0
# plt.figure()
# sns.scatterplot(x='basal', y='coef_var', data=plot_df, hue='KCl_average_peak', linewidth=0, s=4, palette='jet', hue_norm=(-0.2, 2))
# plt.savefig(input_folder + 'All_max_scatter.pdf')
# exit()

total_num_after_filtered = 1
trace_ind = 0
for _, row in sampled.iterrows():
    dff = row['all_trace']
    trace = row['dff_all_trace']

    # if row['ROI_index'] == '148995b_35':
    #     plt.figure()
    #     # plt.plot(np.concatenate(sampled[sampled['ROI_index'] == '148995b_35']['all_trace'].to_numpy()).reshape((-1,)))
    #     # plt.plot(np.concatenate(sampled[sampled['ROI_index'] == '148995b_35']['dff_all_trace'].to_numpy()).reshape((-1,)))
    #     plt.plot(trace)
    #     plt.plot(dff)
    #     plt.show()
        # exit()

    grid_row = int(trace_ind//col_num)
    grid_col = trace_ind % col_num

    filtered_row = low_pass_filter(trace, 0.25, 0.03)

    ax = fig.add_subplot(grid[grid_row, grid_col])
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim([-10, 230])

    # ax.title.set_text(row['ROI_index'] + '_{}_{}%'.format(str(int(row['basal_2'])), str(int(100*np.round(trace_ind/total_num_after_filtered,2)))))
    # ax.title.set_text(row['ROI_index'] + '_{}_{}_{}_{}_{}_{}%'.format(str(int(row['basal_3'])), np.round(row['drop'], 2),
    #                                                               int(1 + row['sti_ind']), np.round(
    #         row['{}_average_peak'.format(stimulus_ls[int(row['sti_ind'])])], 2), np.round(
    #         row['{}_average_peak'.format(stimulus_ls[int(row['sti_ind'] - 1)])], 2), int(100*np.round(trace_ind/total_num_after_filtered,2))))

    ax2 = ax.twinx()
    ax2.set_ylim([-5, 3])
    sns.lineplot(x=range(len(trace)), y=trace, ax=ax, c='k')
    sns.lineplot(x=range(len(dff)), y=dff, ax=ax2, c='b')

    for stimulus_ind, stimulus in enumerate(stimulus_ls):
        ax.add_patch(Rectangle((scheme['stimulus_start'].to_numpy()[stimulus_ind], -5), stimulus_length_ls[stimulus_ind], 1000, alpha=0.5, facecolor='orange'))
        ax.add_patch(
            Rectangle((scheme['F0_start'].to_numpy()[stimulus_ind], -5), scheme['F0_end'].to_numpy()[stimulus_ind] - scheme['F0_start'].to_numpy()[stimulus_ind]+1, 1000,
                      alpha=0.5, facecolor='#c9c8c7'))
    # ax.axvline(x=row['{}_average_peak_index'.format(stimulus)], ymin=-1, ymax=1, color='r', ls='--')

    # ax.axhline(y=row['{}_average_peak'.format(stimulus)], xmin=0, xmax=1, color='g', ls='--', lw=0.5)

    trace_ind += 1

plt.tight_layout()
plt.savefig(input_folder + 'trace.pdf'.format(kcl_resp))
plt.show()