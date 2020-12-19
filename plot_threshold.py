import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sys import exit
from matplotlib.patches import Rectangle
from util import low_pass_filter
from os import listdir
from os.path import isfile, join, isdir

n=200

input_folder = 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\Receptor_CG13575\\'

org = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
org = org.sample(n=n, random_state=0)

col_num = 2
row_num = int(n/2) + 1
print(row_num)
fig = plt.figure(figsize=(col_num*4, row_num*3), constrained_layout=True)
grid = plt.GridSpec(row_num, col_num, wspace=1, hspace=0.5)

trace_ind = 0
for ind in range(org.shape[0]):
    trace = org.iloc[ind,:]['NEAA_trace']
    time_ind = np.array(range(len(trace)))
    grid_row = int(ind/2)

    ax = fig.add_subplot(grid[grid_row, ind%2])
    ax.hlines(y=org.iloc[ind,:]['NEAA_f0_std']*3.5+org.iloc[ind,:]['NEAA_f0_mean'], xmin=0, xmax=time_ind[-1], colors='red', linestyles='dashed')
    ax.hlines(y=org.iloc[ind, :]['NEAA_background_threshold'], xmin=0, xmax=time_ind[-1], colors='green', linestyles='dashed')
    # ax.set_ylim([0, np.max(raw_trace) * 1.1])
    # ax.set_ylim([0, 100])
    sns.lineplot(x=time_ind, y=trace, ax=ax)

    # ax.set_xlabel('time (s)')
    # ax.set_ylabel('F')

plt.tight_layout()
plt.savefig('traces.svg')
plt.show()