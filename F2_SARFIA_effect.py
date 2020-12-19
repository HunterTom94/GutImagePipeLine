from video_util import read_tif
from os import listdir
import numpy as np
from matplotlib import pyplot as plt

folder = 'D:\\Gut Imaging\\Videos\\BackgrounSub\\New folder (2)\\'
videos = listdir(folder)

# params = np.unique([ind.split('_')[1].split('.')[0] for ind in videos]).astype(int)
params = [999,100,70,50,30,10,5,1]

# color_dict = {50:'#2EAC66', 10:'#7B623C',1:'#C81912'}
cell_traces = np.empty((len(params), 123))
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
for ind, param in enumerate(params):
# for ind, param in enumerate(params[::-1]):
    video = read_tif(folder+'247274_{}.tif'.format(param))
    if param == 999:
        mask_template = video[34, :, :]
        cell_mask = np.zeros_like(mask_template)
        bg_mask = np.zeros_like(mask_template)
        cell_mask[807:825, 315:328] = mask_template[807:825, 315:328] > 400
        bg_mask[807:825, 315:328] = mask_template[807:825, 315:328] <= 400
        # continue
    masked_cell = video * cell_mask[np.newaxis, :, :]
    masked_bg = video * bg_mask[np.newaxis, :, :]
    cell_trace = np.empty((video.shape[0],))
    bg_trace = np.empty((video.shape[0],))
    for frame_ind in range(len(cell_trace)):
        cell_frame = masked_cell[frame_ind,:,:]
        bg_frame = masked_bg[frame_ind, :, :]
        cell_trace[frame_ind] = np.mean(cell_frame[np.nonzero(cell_frame)])
        bg_trace[frame_ind] = np.mean(bg_frame[np.nonzero(bg_frame)])
    cell_trace = cell_trace - np.mean(cell_trace[:10])
    label = param
    if label == 999:
        label = 'No Background Subtraction'
        # ax3 = ax1.twinx()
        # ax4 = ax2.twinx()
        ax1.plot(cell_trace, label=label)
        ax2.plot(bg_trace, label=label)
    else:
        # ax1.plot(cell_trace, label=label, color=color_dict[label])
        # ax2.plot(bg_trace, label=label, color = color_dict[label])
        ax1.plot(cell_trace, label=label)
        ax2.plot(bg_trace, label=label)

ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Ball Size')
ax1.set_xlabel('frame index')
ax1.set_ylabel('Relative F Value')
# ax3.set_ylabel('F Value Before Background Subtraction')
ax1.set_xlim([0, 60])
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Ball Size')
ax2.set_xlabel('frame index')
ax2.set_ylabel('F Value')
# ax4.set_ylabel('F Value Before Background Subtraction')
fig1.savefig('Cell_Trace_Ball_Size.svg')
fig1.show()
# fig2.savefig('Bg_Trace_Ball_Size.svg')
# fig2.show()

# cropped = read_tif(folder + 'crop_247274_999.tif')
# mask = cropped[34, :, :] > 440
