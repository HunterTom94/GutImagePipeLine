from video_util import read_tif
from os import listdir
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['text.usetex'] = False
rcParams['svg.fonttype'] = 'none'

# folder = 'H:\\GutImagingData\\BackgroundEffect\\'
folder = 'Z:\\#Gut Imaging Manuscript\\Data\\F2\\F2_BCDEF\\'
videos = listdir(folder)

params = [999, 100, 70, 50, 40, 20]

color_dict = {999: '#000000', 100: '#48466d', 70: '#4a7096', 50: '#549ab4', 40: '#75c4c9', 20: '#abedd8'}
cell_traces = np.empty((len(params), 123))
fig2, (ax, ax2) = plt.subplots(2, 1, sharex=True)
for ind, param in enumerate(params):
    video = read_tif(folder + '247274_{}.tif'.format(param))
    max_proj = np.max(video, axis=0)

    label = param
    if label == 999:
        label = 'No Background Subtraction'
        ax.plot(max_proj[785, 343:368], label=label, color=color_dict[999])
    else:
        ax2.plot(max_proj[785, 343:368], label=label, color=color_dict[label])

ax.set_ylim(400,650)  # outliers only
ax2.set_ylim(0, 250)  # most of the data

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Ball Size')
# ax1.set_xlabel('Frame Index')
# ax1.set_ylabel('Relative F Value')
# ax1.set_xlim([0, 60])
# ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Ball Size')
ax2.set_xlabel('Relative Position (pixel)')
ax2.set_ylabel('F Value')
# fig1.savefig('Cell_Trace_Ball_Size.svg')
# fig1.show()
fig2.savefig('Z:\\#Gut Imaging Manuscript\\V6\\LineProf.svg')
fig2.show()

# cropped = read_tif(folder + 'crop_247274_999.tif')
# mask = cropped[34, :, :] > 440
