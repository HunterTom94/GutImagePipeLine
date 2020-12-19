from video_util import read_tif
from os import listdir
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from os.path import isfile, join, isdir

root_folder = 'D:\\Gut Imaging\\Videos\\MC_pre\\'

video_folders = [folder for folder in listdir(root_folder) if isdir(join(root_folder, folder))]

sample_indices = []
for folder in video_folders:
    sample_indices.append(folder[1:7])

for sample in np.unique(sample_indices):
    sample_folders = [folder for folder in video_folders if sample in folder]
    for folder in sample_folders:
        files = listdir(root_folder+ folder)
        video_ls = [file for file in files if file.endswith('.tif')]
        assert len(video_ls) == 1
        video = read_tif(root_folder+ folder+'\\'+video_ls[0])
        video = video[:,12:-12, 12:-12]
        mean_proj = np.mean(video, axis=0)
        corr_ls = []
        for frame in range(video.shape[0]):
            corr_ls.append(stats.pearsonr(mean_proj.flatten(), video[frame,:,:].flatten())[0])
        corr_ls = np.array(corr_ls)
        corr_dist_ls = 1 - corr_ls
        plt.plot(corr_dist_ls)
    plt.title(sample)
    plt.savefig(root_folder + 'pearson_correlation_distance_{}.png'.format(sample))
    plt.show()