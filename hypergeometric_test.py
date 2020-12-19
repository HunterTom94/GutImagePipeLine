import numpy as np
import pandas as pd
import scipy
from sys import exit
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
import os

input_folder = 'D:\\Yinan\\Clustering\\New folder (2)\\'

file_ls = []

directory = os.fsencode(input_folder)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        label_df = pd.read_csv(input_folder + filename)
        label_df = label_df.iloc[:, 1:]
        label_df.columns = ['ROI_index', 'region', 'fine_region', 'x', 'y', 'label']

        region_total = [160, 661, 525, 901, 1248, 246, 261, 399, 551, 254]
        total_num = np.sum(region_total)

        plot_df = pd.DataFrame(columns=['label', 'region', 'fold', 'pval'])

        for label in np.sort(label_df['label'].unique()):
            cluster_df = label_df.loc[label_df['label'] == label, :]
            cluster_num = cluster_df.shape[0]
            for ind, region in enumerate(np.sort(label_df['region'].unique())):
                region_df = cluster_df.loc[cluster_df['region'] == region, :]
                region_num = region_df.shape[0]
                if region_num >= cluster_num * (region_total[ind] / total_num):
                    pval = hypergeom.sf(region_num - 1, total_num, region_total[ind], cluster_num)
                if region_num < cluster_num * (region_total[ind] / total_num):
                    pval = hypergeom.cdf(region_num, total_num, region_total[ind], cluster_num)
                fold = region_num / (cluster_num * (region_total[ind] / total_num))
                plot_df = plot_df.append({'label': label, 'region': region, 'fold': fold, 'pval': pval},
                                         ignore_index=True)
        sns.catplot(x="label", y="fold", hue="region", data=plot_df, kind="bar")
        plt.axhline(y=1, xmin=0, xmax=1)
        plt.title('_'.join(filename.split('_')[:2]))
        plt.savefig(
            'D:\\Yinan\\Clustering\\New folder (4)\\hypergeometric_{}.png'.format('_'.join(filename.split('_')[:2])))
        plt.show()
        plot_df.to_csv('D:\\Yinan\\Clustering\\New folder (4)\\hypergeometric_data{}.txt'.format(
            '_'.join(filename.split('_')[:2])), sep='\t')
