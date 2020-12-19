from util import roi_plot
from Igor_related_util import read_igor_roi_matrix
import numpy as np


folder = 'D:\\Gut Imaging\\Videos\\Temp\\d334546\\'
# folder = 'D:\\Gut Imaging\\Videos\\temp_temp\\d278487\\'
igor_roi = read_igor_roi_matrix(folder + 'roi' + '.csv')

# corr_sum = np.load(folder + 'corr_sum.npy')
# throw_ls_idx = np.load(folder + 'sort_idx.npy')

# roi_plot(igor_roi,folder,label_font=15, throw_ls=throw_ls_idx[:50])

targets = [418,	422,	423,	424,	426,	427,	428,	429,	430,	431,	432,	433,	434,	435,	436,	437,	438,	439,	440,	441,	442,	443,	444,	445,	446,	447,	448,	449]
roi_plot(igor_roi,folder,label_font=15, targets=targets)