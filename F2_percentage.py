import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

df = pd.read_csv('Z:\\#Gut Imaging Manuscript\\old files\\Figures\\F5\\13\\percentage.csv')

a = sns.barplot(x='Cluster Index', y='Proportion among All Cells', data=df, hue='Genotype',ci=68, palette=['#000000', '#808080'], capsize=0.1)
a.legend_.remove()
a.set_ylim([0,0.6])
plt.savefig('Z:\\#Gut Imaging Manuscript\\percentage.svg')
plt.show()