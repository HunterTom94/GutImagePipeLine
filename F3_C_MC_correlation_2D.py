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
import matplotlib
from os.path import isfile, join, isdir
import pandas as pd
import seaborn as sns
from sys import exit
from util import plot_colorbar

# root_folder = 'D:\\Gut Imaging\\Videos\\ParaTest\\2D\\'
root_folder = 'Z:\\#Gut Imaging Manuscript\\Data\\F3\\F3_ABC\\'

# ax = plt.subplot()
# colorbar = plot_colorbar(ax, hue_nrom=[0,1], cmap=matplotlib.cm.get_cmap(), orientation='vertical')
# plt.savefig(root_folder + 'colorbar.svg')
#
# exit()

df = pd.read_pickle('Z:\\#Gut Imaging Manuscript\\V6\\sample_scatter.pkl')
df = df[['grid_size','overlap_size','norm_distance']]
df = df[df['grid_size'] != 200]

g_ls = np.sort(df['grid_size'].unique()).astype(int)[1:]
o_ls = np.sort(df['overlap_size'].unique()).astype(int)[1:]
heat_np = np.empty((len(o_ls), len(g_ls)))
heat_np[:,:] = np.nan


df = df.groupby(['grid_size', 'overlap_size']).mean()
df['norm_distance'] = df['norm_distance']/df['norm_distance'].max()
df = df.iloc[1:]


plot_df = pd.DataFrame(columns=['grid_size', 'overlap_size', 'norm_distance'])
for _, row in df.iterrows():
    col_ind = np.where(g_ls == row.name[0])[0][0]
    row_ind = np.where(o_ls == row.name[1])[0][0]
    heat_np[row_ind, col_ind] = np.round(row.values[0],2)
    # plot_df = plot_df.append({'grid_size':row.name[0], 'overlap_size':row.name[1], 'norm_distance':row.values[0]}, ignore_index=True)

heat_np = heat_np[::-1, :]
o_ls = o_ls[::-1]
fig, ax = plt.subplots()
# current_cmap = matplotlib.cm.get_cmap()
current_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                                               [(0, '#abedd8'),
                                                                # (0.33, '#20A585'),
                                                                (0.5, '#46cdcf'),
                                                                (0.75, '#3d84a8'),

                                                                # (0.66, 'Yellow'),
                                                                (1, '#48466d')], N=126)
current_cmap.set_bad(color='gray')
im = ax.imshow(heat_np, cmap=current_cmap, vmin = 0.65, vmax= 0.82)

# current_cmap = matplotlib.cm.get_cmap('jet')
# current_cmap.set_bad(color='gray')
# im = ax.imshow(heat_np, cmap=current_cmap)

ax.set_xticks(np.arange(len(g_ls)))
ax.set_yticks(np.arange(len(o_ls)))
ax.set_xticklabels(g_ls)
ax.set_yticklabels(o_ls)
ax.set_xlabel('Grid Size')
ax.set_ylabel('Overlap Size')

plt.setp(ax.get_xticklabels(), rotation=0,# ha="right",
         rotation_mode="anchor")

for i in range(len(o_ls)):
    for j in range(len(g_ls)):
        if heat_np[i, j] > 0.75:
            color = 'w'
        else:
            color = 'k'
        text = ax.text(j, i, heat_np[i, j],
                       ha="center", va="center", color=color)

ax.set_title("Normalize Distance")
fig.tight_layout()
plt.savefig('Z:\\#Gut Imaging Manuscript\\V6\\2D_heat.svg')
plt.show()
# sns.scatterplot(x='grid_size', y='overlap_size', hue='norm_distance', data=plot_df, palette='jet', hue_norm=(0,1), legend=False, s=170)
exit()

video_folders = [folder for folder in listdir(root_folder) if isdir(join(root_folder, folder))]

sample_indices = np.unique([ind.split('_')[0] for ind in video_folders])

g_ls = np.unique([ind.split('g')[1].split('o')[0] for ind in video_folders]).astype(int)
o_ls = np.unique([ind.split('o')[1] for ind in video_folders]).astype(int)

color_dict = {'10':'#2EAC66', '20':'#4C8E55','40':'#6B7144','60':'#8A5333','100':'#A93622','200':'#C81912'}

plot_df = pd.DataFrame(columns=['sample','grid_size', 'overlap_size', 'norm_distance'])

for sample in sample_indices:
    folder = root_folder + '{}_g0o0\\'.format(sample)
    files = listdir(folder)
    video_ls = [file for file in files if file.startswith('nocrop')]
    print(folder)
    assert len(video_ls) == 1
    video = read_tif(folder + video_ls[0])
    video = video[:, 12:-12, 12:-12]
    mean_proj = np.mean(video, axis=0)
    corr_ls = []
    for frame in range(video.shape[0]):
        corr_ls.append(stats.pearsonr(mean_proj.flatten(), video[frame, :, :].flatten())[0])
    corr_ls = np.array(corr_ls)
    baseline = 1 - corr_ls
    for g in g_ls:
        for o in o_ls:
            folder = root_folder + '{}_g{}o{}\\'.format(sample, g, o)
            try:
                files = listdir(folder)
                video_ls = [file for file in files if file.startswith('nocrop')]
                print(folder)
                assert len(video_ls) == 1
                video = read_tif(folder + video_ls[0])
                video = video[:, 12:-12, 12:-12]
                mean_proj = np.mean(video, axis=0)
                corr_ls = []
                for frame in range(video.shape[0]):
                    corr_ls.append(stats.pearsonr(mean_proj.flatten(), video[frame, :, :].flatten())[0])
                corr_ls = np.array(corr_ls)
                corr_dist_ls = 1 - corr_ls
                plot_df = plot_df.append({'sample':sample, 'grid_size':g, 'overlap_size':o, 'norm_distance':np.mean(corr_dist_ls)/np.mean(baseline)}, ignore_index=True)
            except FileNotFoundError:
                print('error')
                pass

plot_df.to_pickle(root_folder + 'sample_scatter.pkl')
exit()