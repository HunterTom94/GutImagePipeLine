import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sys import exit
from matplotlib.patches import Rectangle
from util import low_pass_filter
from os import listdir
from os.path import isfile, join, isdir

root_folder = 'D:\\Gut Imaging\\Videos\\baseline_trace\\'

file_ls = listdir(root_folder)

raw_ls = [file for file in file_ls if 'Raw' in file]
non_raw_ls= [file for file in file_ls if 'Raw' not in file]

raw_df_ls = []
for raw in raw_ls:
    raw_df_ls.append(pd.read_pickle(root_folder+raw))

raw_df = pd.concat(raw_df_ls, axis=0, ignore_index=True)
raw_df = raw_df.sort_values(by='ROI_index')

non_raw_df_ls = []
for non_raw in non_raw_ls:
    non_raw_df_ls.append(pd.read_pickle(root_folder+non_raw))

non_raw_df = pd.concat(non_raw_df_ls, axis=0, ignore_index=True)
non_raw_df = non_raw_df.sort_values(by='ROI_index')

on_ROIs = non_raw_df[non_raw_df['KCl_average_response'] == 1]['ROI_index'].to_frame().merge(raw_df['ROI_index'],
                                                                                            how='inner', on='ROI_index')
on_traces = pd.DataFrame()
on_traces['ROI_index'] = on_ROIs['ROI_index'].tolist()
on_traces = on_traces.merge(raw_df[['ROI_index', 'KCl_trace']], how='inner', on='ROI_index')
on_traces.columns = ['ROI_index', 'raw_trace']
on_traces = on_traces.merge(non_raw_df[['ROI_index', 'KCl_trace']], how='inner', on='ROI_index')
on_traces.columns = ['ROI_index', 'raw_trace', 'non_trace']
on_traces = on_traces.sample(n=50, random_state=0)

off_ROIs = non_raw_df[non_raw_df['KCl_average_response'] == 0]['ROI_index'].to_frame().merge(raw_df['ROI_index'],
                                                                                            how='inner', on='ROI_index')
off_traces = pd.DataFrame()
off_traces['ROI_index'] = off_ROIs['ROI_index'].tolist()
off_traces = off_traces.merge(raw_df[['ROI_index', 'KCl_trace']], how='inner', on='ROI_index')
off_traces.columns = ['ROI_index', 'raw_trace']
off_traces = off_traces.merge(non_raw_df[['ROI_index', 'KCl_trace']], how='inner', on='ROI_index')
off_traces.columns = ['ROI_index', 'raw_trace', 'non_trace']
off_traces = off_traces.sample(n=500, random_state=0)

big_df_index = []
for ind in range(off_traces.shape[0]):
    raw_trace = off_traces.iloc[ind,:]['raw_trace']
    if np.max(raw_trace) - np.min(raw_trace) > 20 and len(raw_trace)>30:
        big_df_index.append(ind)

off_traces = off_traces.iloc[big_df_index, :]

col_num = 2
row_num = int(len(big_df_index)/2) + 1
print(row_num)
fig = plt.figure(figsize=(col_num*4, row_num*3), constrained_layout=True)
grid = plt.GridSpec(row_num, col_num, wspace=1, hspace=0.5)

trace_ind = 0
for ind in range(off_traces.shape[0]):
    raw_trace = off_traces.iloc[ind,:]['raw_trace']
    non_trace = off_traces.iloc[ind,:]['non_trace']
    time_ind = np.array(range(len(raw_trace)))
    grid_row = int(ind/2)


    ax = fig.add_subplot(grid[grid_row, ind%2])
    ax.vlines(x=31, ymin=-20, ymax=120, colors='red', linestyles='dashed')
    # ax.set_ylim([0, np.max(raw_trace) * 1.1])
    ax.set_ylim([-20, 120])
    ax.set_yticks([0, 50,100])
    sns.lineplot(x=time_ind, y=raw_trace, ax=ax)
    ax2 = ax.twinx()
    sns.lineplot(x=time_ind, y=non_trace, ax=ax2, c='green')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('F')
    ax2.set_ylabel('DFF')
    ax2.set_ylim([-1,6])
    ax2.set_yticks([0, 2.5, 5])

plt.tight_layout()
plt.savefig('off_traces.svg')
plt.show()