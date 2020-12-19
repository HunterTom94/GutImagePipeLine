import pandas as pd
import numpy as np
from organized_Clustering import UMAP_gen, UMAP_gen_paralle
import itertools
from DataOrganizer import organized_gen, data_plus_label, roiName2trace, parameters_mean_sem, parameters_mean_sem_cluster
from DataPlotter import plot_cluster_trace_grid_new
from util import colorful_roi_map
import matplotlib.pyplot as plt
from video_util import read_tif
from os import listdir
from os.path import isfile, join
import cv2
from joblib import Parallel, delayed
import multiprocessing as mp

from sys import exit

def read_organized():
    return pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')


def roi_filter(organized, roi_names_file):
    if roi_names_file == '':
        return organized
    roi_name_df = pd.read_csv(input_folder + roi_names_file, header=None)
    roi_name_df.columns = ['ROI_index']
    return organized.loc[organized['ROI_index'].isin(roi_name_df['ROI_index']), :]


# File name for delivery scheme is "DeliverySchemes.xlsx"
# File name for trace ends with "Ave.txt"
# File name for AP info contains "Peak"

# input_folder = 'D:\\#Yinan\\KCl_fit_test\\analysis\\'
# input_folder = 'D:\\#Yinan\\Cys\\'
# input_folder = 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\Map_new_NoDiSC\\'
input_folder = 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\Map_20200113\\'
output_folder = input_folder + 'SelectedCluster\\'

scheme = pd.read_excel(input_folder + '\\DeliverySchemes.xlsx')
stimulus_ls = scheme.iloc[:, 1].to_list()

# ###############################################     Database Generator       ###########################################
# filter_ls = [] # Exact Stimuli Names to be Excluded from Filter
# raw_organized = organized_gen(input_folder, raw=1)
# organized = organized_gen(input_folder, raw=0, svm_filter=1, filter_ls=filter_ls)
# exit()

# ######################################     Mean SEM Parameters Generator       ########################################
# # # *********************************************************************************************
# roi_names_file = ''
# # *********************************************************************************************
#
# organized = read_organized()
# organized = roi_filter(organized, roi_names_file)
#
# parameters_mean_sem(organized, input_folder, roi_names_file, scheme)
# exit()

# #############################################      Selected ROI Trace TXT    #########################################
# # # # # *********************************************************************************************
# roi_names_file = 'Dh31Dos_148995b4heatmap3.txt'
# # # # *********************************************************************************************
#
# organized = read_organized()
#
# output_trace_file = roi_names_file.split('.')[0] + '_trace.txt'
# roi_name_df = pd.read_csv(input_folder + roi_names_file,header=None)
# roiName2trace(organized, roi_name_df).to_csv(input_folder + output_trace_file,sep='\t')
# exit()

# ############################################        Ave Trace Figure         #########################################
# # # #*********************************************************************************************
# roi_names_file = ''
# stimulus_ls_remove = []
#
# color = 'black'
# ylim = [-2, 8]
# linewidth = 1
# height = 2
# width = 2
# stimulus_bar_y=-0.5
# stimulus_bar_lw=2
# scalebar_frame_length=12
# scalebar_amplitude_length=1
# svg = 1
# # *********************************************************************************************
#
# organized = read_organized()
# organized = roi_filter(organized, roi_names_file) # comment if want to use the entire dataset
# for stimulus in stimulus_ls_remove:
#     stimulus_ls.remove(stimulus)
#
# artificial_label_df = pd.DataFrame(organized['ROI_index'])
# artificial_label_df['label'] = 0
# organized_pls_label = data_plus_label(organized, artificial_label_df)
#
# plot_cluster_trace_grid_new(output_folder, organized_pls_label, scheme, 'label', stimulus_ls, ylim=ylim,
#                             linewidth=linewidth, color=color, height=height, width=width, svg=svg,
#                             stimulus_bar_y=stimulus_bar_y, stimulus_bar_lw=stimulus_bar_lw,
#                             scalebar_frame_length=scalebar_frame_length,
#                             scalebar_amplitude_length=scalebar_amplitude_length)
# exit()


# ######################################################      UMAP       ###############################################
# # ******************************************************************************************************************
roi_names_file = ''  # a list of ROI names
stimulus_ls_remove = ['AHL']
# stimulus_ls_remove = []
screen = 1

# UMAP_nearest_neighbor_ls = [3, 4, 5, 6, 7,8,9,10,13,15,18,21,25,30,35,40,50,60,75,90,100,120,150,190,250,300,350,400,450,500,550,600] # UMAP neigher #
# UMAP_nearest_neighbor_ls = [3, 4, 5, 6, 7,8,9,10,13,15,18,21,25,30,35,40,50]
UMAP_nearest_neighbor_ls = [100]
# HDBSCAN_para_ls = list(itertools.permutations([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,40,50,70,90,120,150,200,300,400,500,600],2))
HDBSCAN_para_ls = [[20,10]]  # cluster parameter
random_state_ls = [1,2,3,5,8,10,12,15,20,30,35,40,45,50,60,80,90,100,120,150,160,180,200,220,250,300,350,400,500,600,800]  # seed
# random_state_ls = [150]  # seed
average_peak = 1
z_score_norm = 1
anyresp = 1  # Assumption: exclude all stimuli whose names contain 'KCl'
exclude_kcl = 0 # when 1, only keep KCl positive
UMAP_region_as_input = 0 # when 1, add region (R1-R5) as UMAP input

heatmap_range = (0, 2.5)
AP_heatmap_range = (0,100)
color_bar_tick_labels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
AP_color_bar_tick_labels = [0,20,40,60,80,100]
dot_size = 12
dot_edgewidth = 0.1
cluster_fontsize = 8
# xlim_para=[-8,8,2] #Note: min, max, interval
# ylim_para=[-4,6,2]
ylim_para=[]
xlim_para=[] #Note: min, max, interval
# Note: row*col >= stimulus # + 7
row_num = 5
col_num = 4
hspace = 8  # height space between figures
wspace = 4  # width space between figures
svg = 0

region_type = 'region'  # RGBA Values Below; choose between 'fine_region' and 'region'

region_palatte_dict = {'R1': [255, 76, 76, 255],
                       'R2': [255, 165, 0, 255],
                       'R3': [50, 153, 50, 255],
                       'R4': [50,50,255, 255],
                       'R5': [0, 0, 0, 255]}
fine_region_palatte_dict = {'R1': [153, 93, 19, 255],
                            'R2a': [195, 85, 58, 255],
                            'R2b': [213, 138, 119, 255],
                            'R2c': [233, 192, 182, 255],
                            'R3': [255, 217, 47, 255],
                            'R4a': [169, 195, 183, 255],
                            'R4b': [93, 131, 112, 255],
                            'R4c': [18, 68, 44, 255],
                            'R5a': [138, 99, 175, 255],
                            'R5b': [83, 28, 135, 255]}
cluster_palatte_dict = {0: ['#ff2400', 255],
                        1: ['#ff7548', 255],
                        2: ['#ff4877', 255],
                        3: ['#cd00cd', 255],
                        4: ['#ea3a00', 255],
                        5: ['#1651ff', 255],
                        6: ['#ff92ff', 255],
                        7: ['#fb0040', 255],
                        8: ['#ff748c', 255],
                        9: ['#b300b3', 255],
                        10: ['#d13400', 255]}
# # ******************************************************************************************************************
organized = read_organized()
organized = roi_filter(organized, roi_names_file)
for stimulus in stimulus_ls_remove:
    stimulus_ls.remove(stimulus)

for dict in [region_palatte_dict, fine_region_palatte_dict]:
    for key, value in dict.items():
        dict.update({key: (np.array(value) / 255).tolist()})
region_dict = {'region': region_palatte_dict, 'fine_region': fine_region_palatte_dict}

for key, value in cluster_palatte_dict.items():
    RGBA = []
    hex = value[0]
    if '#' in hex:
        hex = hex.replace("#", "")
    for hex_split in [hex[i:i + 2] for i in range(0, len(hex), 2)]:
        RGBA.append(int(hex_split, 16))
    RGBA.append(value[1])
    cluster_palatte_dict.update({key: (np.array(RGBA) / 255).tolist()})

def UMAP_exe(parallel_input):
    UMAP_gen_paralle(output_folder, organized, stimulus_ls, hdbscan_para=parallel_input[1], n_n=parallel_input[0],
             random_state=parallel_input[2], row_num=row_num, col_num=col_num, average_peak=average_peak,
             z_score_norm=z_score_norm, svg=svg, hue_norm=heatmap_range, AP_hue_norm=AP_heatmap_range,
             dot_size=dot_size, cluster_fontsize=cluster_fontsize, region_palatte_dict=region_dict[region_type],
             region_type=region_type, hspace=hspace, wspace=wspace, color_bar_tick_labels=color_bar_tick_labels,
             AP_color_bar_tick_labels=AP_color_bar_tick_labels, dot_edgewidth=dot_edgewidth, anyresp=anyresp,
             xlim_para=xlim_para, ylim_para=ylim_para, exclude_kcl=exclude_kcl, UMAP_region_as_input=UMAP_region_as_input)

parallel_input_ls = list(itertools.product(UMAP_nearest_neighbor_ls, HDBSCAN_para_ls, random_state_ls))
num_cores = mp.cpu_count()

if screen:
    if __name__ == '__main__':

        pool = mp.Pool(num_cores)
        pool.map(UMAP_exe, parallel_input_ls)

        pool.close()
        pool.join()

else:
    UMAP_gen(output_folder, organized, stimulus_ls, HDBSCAN_para_ls, UMAP_nearest_neighbor_ls, random_state_ls, row_num,
             col_num, average_peak=average_peak, z_score_norm=z_score_norm, svg=svg, hue_norm=heatmap_range,
             AP_hue_norm=AP_heatmap_range, dot_size=dot_size, cluster_fontsize=cluster_fontsize,
             region_palatte_dict=region_dict[region_type], region_type=region_type, hspace=hspace, wspace=wspace,
             color_bar_tick_labels=color_bar_tick_labels, AP_color_bar_tick_labels=AP_color_bar_tick_labels,
             dot_edgewidth=dot_edgewidth, anyresp=anyresp, xlim_para=xlim_para, ylim_para=ylim_para, cluster_palatte_dict=cluster_palatte_dict, exclude_kcl=exclude_kcl, UMAP_region_as_input=UMAP_region_as_input)

#####################################    Cluster Mean SEM Parameters Generator       ########################################
# *********************************************************************************************
# roi_names_file = ''
# cluster_label_name = '100_28_12_150cluster_label.txt'
# # *********************************************************************************************
#
# organized = read_organized()
# organized = roi_filter(organized, roi_names_file)
#
# parameters_mean_sem_cluster(organized, save_dir=input_folder, roi_names_file=roi_names_file, cluster_label_file=cluster_label_name, scheme=scheme)
# exit()

##############################################          Cluster Ave Trace          #####################################
# # # ********************************************************************************************************************
# roi_names_file = ''
# stimulus_ls_remove = []
# # stimulus_ls_remove = ['Trp','Gly','Val','Ser','Met','Pro','Gln','Ala','His','Glu','Ile','Thr','Leu','Arg','Phe','Lys','KCl']
# cluster_label_name = '100_5_14_150cluster_label.txt'
# min_cell_per_sample = 3
#
# ylim = [-1, 8]
# color = 'black'
# linewidth = 4
# height = 8
# width = 4
# cluster_order = [3,10,6,9,1,12,11,13,8,7,5,4,2]
# stimulus_bar_y= -1
# stimulus_bar_lw= 8
# scalebar_frame_length= 15
# scalebar_amplitude_length= 2
# svg = 0
#
#
# # ********************************************************************************************************************
#
# organized = read_organized()
# organized = roi_filter(organized, roi_names_file)
# for stimulus in stimulus_ls_remove:
#     stimulus_ls.remove(stimulus)
#
# organized_pls_label = data_plus_label(organized, pd.read_csv(output_folder + cluster_label_name, sep='\t').loc[:,
#                                                  ['ROI_index', 'label']])
#
# for label in organized_pls_label['label'].unique():
#     for sample_index in organized_pls_label['sample_index'].unique():
#         print('Label {} has {} cells from {}'.format(label, organized_pls_label[(organized_pls_label['sample_index'] == sample_index)&(organized_pls_label['label'] == label)].shape[0], sample_index))
# # exit()
#
# plot_cluster_trace_grid_new(output_folder, organized_pls_label, scheme, 'label', stimulus_ls, ylim=ylim,
#                             linewidth=linewidth, color=color, height=height, width=width, svg=svg,
#                             cluster_order=cluster_order, stimulus_bar_y=stimulus_bar_y, stimulus_bar_lw=stimulus_bar_lw,
#                             scalebar_frame_length=scalebar_frame_length,
#                             scalebar_amplitude_length=scalebar_amplitude_length, min_cell_per_sample=min_cell_per_sample)
# exit()
# #
# ##############################################          Colorful Plots          ########################################
# # # ********************************************************************************************************************
# BGR_dict = {'gray': [128, 128, 128], 'red': [0, 0, 255], 'purple': [130, 0, 75], 'light_purple': [255, 0, 255],
#                 'blue': [255, 0, 0], 'pink': [221, 160, 221], 'green': [0, 255, 0]}
# root_folder = ''
# # ********************************************************************************************************************
#
# colorful_roi_map(root_folder, BGR_dict)
#
# ##############################################          Peak Frame          ########################################
#
# color_min, color_max = [0,30]
# exe_folder = 'D:\\Gut Imaging\\Videos\\Temp\\d325455\\'
#
#
# organized = read_organized()
# onlyfiles = [f for f in listdir(exe_folder) if isfile(join(exe_folder, f))]
# video_name = [f for f in onlyfiles if
#               f.split('.')[-1] == 'tif' and f.split('_')[0].lower() != 'MAX'.lower() and 'immuno' not in f]
# assert len(video_name) == 1, "More than one tif file found in" + exe_folder
# video_name = video_name[0]
# raw_stack, video_index = read_tif(exe_folder, video_name)
# print(raw_stack.shape)
#
# average_peak_ls = []
# stimulus_ls = scheme.iloc[:, 1].to_list()
# for stimulus in stimulus_ls:
#     assert len(organized['{}_average_peak_index'.format(stimulus)].unique()) == 1
#     average_peak_ls.append(np.unique(organized['{}_average_peak_index'.format(stimulus)])[0])
# df = pd.DataFrame()
# df['average_peak_index'] = average_peak_ls + scheme['video_start'] - 1
# df['F0_start'] = scheme['F0_start'] - 1
# df['F0_end'] = scheme['F0_end']
#
# for stimulus_ind in range(len(stimulus_ls)):
#     average_peak = df.loc[stimulus_ind, 'average_peak_index']
#     average_frame = np.mean(raw_stack[average_peak-1:average_peak+2, :, :],axis=0)
#     f0_frame = np.mean(raw_stack[df.loc[stimulus_ind, 'F0_start']:df.loc[stimulus_ind, 'F0_end'], :, :],axis=0)
#     delta_frame = average_frame - f0_frame
#     delta_frame = np.clip(delta_frame, a_min=color_min, a_max=color_max)
#     delta_frame = cv2.bilateralFilter(delta_frame, 9, 15, 15)
#     delta_frame = cv2.normalize(delta_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
#     im_color = cv2.applyColorMap(delta_frame, cv2.COLORMAP_JET)
#     print(stimulus_ls[stimulus_ind])
#     # cv2.imsave()
#     cv2.imshow('a', im_color)
#     cv2.waitKey(-1)
#     # fig,ax = plt.subplots(1)
#     # fig.subplots_adjust(left=0,right=302,bottom=0,top=696)
#     # ax.imshow(delta_frame,cmap='jet', vmin=color_min, vmax=color_max, extent=(0, 696, 302, 0))
#     # ax.axis('tight')
#     # ax.axis('off')
#     # # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
#     # #                 hspace=0, wspace=0)
#     # # plt.margins(0, 0)
#     # # plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     # # plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     # plt.savefig(input_folder + '{}_peak_frame.png'.format(stimulus_ls[stimulus_ind]))
#     # plt.clf()
