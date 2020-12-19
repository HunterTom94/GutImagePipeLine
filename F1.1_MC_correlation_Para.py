from video_util import read_tif
from os import listdir
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from os.path import isfile, join, isdir

root_folder = 'D:\\Gut Imaging\\Videos\\ParaTest\\1_MC_paraTest\\'

video_folders = [folder for folder in listdir(root_folder) if isdir(join(root_folder, folder))]

sample_indices = np.unique([ind.split('_')[0] for ind in video_folders])

color_dict = {'10':'#2EAC66', '20':'#4C8E55','40':'#6B7144','60':'#8A5333','100':'#A93622','200':'#C81912'}

ave_diff = np.empty((0,6))
for sample_indice in sample_indices:
    sample_videos = [video for video in video_folders if sample_indice in sample_indices]
    params = np.unique([ind.split('_')[1] for ind in sample_videos])
    params = np.array([param.split('g')[1].split('o')[0] for param in params ])
    params = np.sort(params.astype(int))[::-1]

    traces = np.empty((0, 287))
    fig, ax = plt.subplots()

    for param in params.astype(str):
        folder = sample_indice + '_g' + param+'o10\\'
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
        traces = np.append(traces, corr_dist_ls.reshape(1,-1), axis=0)
        label = param
        ax.plot(corr_dist_ls, label=label, color = color_dict[label])
    traces = np.mean(traces - traces[-1,:], axis=1)
    ave_diff = np.append(ave_diff, traces.reshape(1,-1), axis=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='grid size')
    ax.set_xlabel('frame index')
    ax.set_ylabel('Pearson Correlation Distance')
    plt.title(sample_indice)
    plt.savefig(root_folder + 'pearson_correlation_distance_{}.svg'.format(sample_indice), bbox_inches='tight')
    plt.show()
np.savetxt(root_folder + 'diff.txt',ave_diff, delimiter='\t')