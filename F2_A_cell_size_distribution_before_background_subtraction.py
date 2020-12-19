import numpy as np
from video_util import read_tif
from os import listdir
from os.path import isfile, join, isdir
from scipy.signal import correlate2d
from Igor_related_util import read_igor_roi_matrix
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['text.usetex'] = False
rcParams['svg.fonttype'] = 'none'
from skimage import measure
import seaborn as sns
from scipy.spatial.distance import pdist, squareform


# roi = read_igor_roi_matrix('H:\\GutImagingData\\before_subtraction_roi.csv')
roi = read_igor_roi_matrix('Z:\\#Gut Imaging Manuscript\\Data\\F2\\F2_A\\d2467\\roi.csv')

dist_ls = []
roi_indices = np.unique(roi)
for roi_index in roi_indices[1:]:
    x,y = np.where(roi == roi_index)
    coordinates = np.array(list(zip(x,y)))
    dist_mx = squareform(pdist(coordinates, metric='euclidean'))
    dist_ls.append(np.max(dist_mx))

a = plt.subplot()
bins= np.arange(0, 25, 0.5)
count, bin = np.histogram(dist_ls,bins=bins)
norm_count = count/np.sum(count) * 100
center = (bin[:-1] + bin[1:]) / 2
a.bar(center, norm_count, align='center', width=0.5, color='#FFFFFF', edgecolor='#808080', linewidth=1)

# a = sns.distplot(dist_ls, kde=False,color='#FFFFFF', )
a.set_xlabel('Max Distance Between Two Points in One ROI')
a.set_ylabel('Percentage')
a.set_xlim([0,25])
# a.set(xticks=np.arange(0,27,5))
plt.savefig('Z:\\#Gut Imaging Manuscript\\V6\\cell_size_dist_before_bg_sub.svg')
plt.show()