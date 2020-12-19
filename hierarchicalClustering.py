import pandas as pd
import numpy as np
import itertools
from scipy.stats.stats import pearsonr
from time import time
from sys import exit
from util import low_pass_filter

organized = pd.read_pickle('D:\\Yinan\\GutImageDataAnalysis\\SensingMap\\SensingMap_organized_data.pkl')
stimulus_ls_ls = ['BCAA', 'EAA', 'NEAA', 'Non nutritive', 'Di sugar', 'Mono sugar', 'SCFA', 'MCFA', 'LCFA']
for stimulus in stimulus_ls_ls:
    stimulus_ls = [stimulus]

    response_only = organized.loc[organized[['{}_response'.format(stimulus) for stimulus in stimulus_ls]].loc[

                                  (organized[
                                                   ['{}_response'.format(stimulus) for stimulus in stimulus_ls]] == 1).any(
                                                  axis=1)].index, :]
    trace_np = np.empty((response_only.shape[0], 0))
    for stimulus in stimulus_ls:
        trace_np = np.hstack((trace_np, np.concatenate(organized.loc[response_only.index, :]['{}_trace'.format(stimulus)].to_numpy()).reshape(response_only.shape[0], -1)))
    trace_np = np.apply_along_axis(low_pass_filter, 1, trace_np)
    ####################################################################################################################

    combo_ls = list(itertools.combinations(range(trace_np.shape[0]), 2))

    corr_mx = np.ones((trace_np.shape[0], trace_np.shape[0]))

    for ind, combo in enumerate(combo_ls):
        corr = pearsonr(trace_np[combo[0], :], trace_np[combo[1], :])[0]
        corr_mx[combo[0], combo[1]] = corr
        corr_mx[combo[1], combo[0]] = corr
        print(np.round(ind/len(combo_ls), 3))

    distance_mx = 1 - corr_mx
    np.fill_diagonal(distance_mx, 0)

    distance_mx_df = pd.DataFrame(distance_mx)
    distance_mx_df.columns = organized.loc[response_only.index, :]['ROI_index']
    indexNamesArr = distance_mx_df.index.values
    indexNamesArr = organized.loc[response_only.index, :]['ROI_index']
    # np.save('D:\\Yinan\\Clustering\\distance_mx.npy', distance_mx)
    # np.save('D:\\Yinan\\Clustering\\distance_mx_sugar.npy', distance_mx)
    distance_mx_df.to_csv('D:\\Yinan\\GutImageDataAnalysis\\SensingMap\\distance_mx_{}_filtered.txt'.format(stimulus_ls[0]), sep='\t')
