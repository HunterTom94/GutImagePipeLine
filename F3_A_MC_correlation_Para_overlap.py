from video_util import read_tif
from os import listdir
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['text.usetex'] = False
rcParams['svg.fonttype'] = 'none'
from os.path import isfile, join, isdir

# root_folder = 'H:\\GutImagingData\\overlap\\'
root_folder = 'Z:\\#Gut Imaging Manuscript\\Data\\F3\\F3_ABC\\'

video_folders = [folder for folder in listdir(root_folder) if isdir(join(root_folder, folder))]

sample_indices = np.unique([ind.split('_')[0] for ind in video_folders])

color_dict = {'0':'#000000', '10':'#2EAC66', '20':'#617B4A','40':'#944A2E','60':'#C81912'}

# ave_diff = np.empty((0,6))
for sample_indice in ['d246874']:
    sample_videos = [video for video in video_folders if sample_indice in sample_indices]
    params = np.unique([ind.split('_')[1] for ind in sample_videos])
    params = np.array([param.split('o')[1].split('.')[0] for param in params ])
    params = np.sort(params.astype(int))[::-1]

    # traces = np.empty((0, 287))
    # fig, ax = plt.subplots()
    fig2, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 7))

    for param in params.astype(str):
        if param == '0':
            label = 'No Motion Correction'
            folder = sample_indice + '_g0' + 'o{}\\'.format(param)
        else:
            label = param
            folder = sample_indice + '_g40' +'o{}\\'.format(param)
        files = listdir(root_folder+ folder)
        video_ls = [file for file in files if file.startswith('nocrop')]
        assert len(video_ls) == 1
        video = read_tif(root_folder+ folder+video_ls[0])
        video = video[:,12:-12, 12:-12]
        mean_proj = np.mean(video, axis=0)
        corr_ls = []
        for frame in range(video.shape[0]):
            corr_ls.append(stats.pearsonr(mean_proj.flatten(), video[frame,:,:].flatten())[0])
        corr_ls = np.array(corr_ls)
        corr_dist_ls = 1 - corr_ls
        # traces = np.append(traces, corr_dist_ls.reshape(1,-1), axis=0)


        ax.plot(corr_dist_ls, label=label, color = color_dict[param])

    ax.set_ylim(0.15, 0.4)  # outliers only
    ax2.set_ylim(0, 0.05)  # most of the data

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    # traces = np.mean(traces - traces[-1,:], axis=1)
    # ave_diff = np.append(ave_diff, traces.reshape(1,-1), axis=0)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='overlap size')
    ax2.set_xlabel('Frame Index')
    ax.set_ylabel('Pearson Correlation Distance')
    # plt.title(sample_indice)
    plt.savefig('Z:\\#Gut Imaging Manuscript\\V6\\pearson_correlation_distance_{}.svg'.format(sample_indice), bbox_inches='tight')
    plt.show()
# np.savetxt(root_folder + 'diff.txt',ave_diff, delimiter='\t')