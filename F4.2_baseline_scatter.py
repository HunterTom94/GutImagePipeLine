import pandas as pd
import numpy as np
from sys import exit
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
from util import low_pass_filter
from sklearn.mixture import GaussianMixture


def drv_pearson_correlation(x,y):
    x = np.gradient(low_pass_filter(x, 1, 0.05))
    return stats.pearsonr(x,y)[0]

train_folder_ls = []
test_folder_ls = []

# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\AstC\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\CCHa2\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros1\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros2\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros3\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros4\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_controls\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_Dosage3\\')

model_name = 'Pros'

test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\AstC\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\CCHa2\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros1\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros2\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros3\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros4\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_controls\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_Dosage3\\')

def prepare_data(input_folder):
    organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
    raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')
    organized = organized[~organized["ROI_index"].str.contains('_b')]
    raw_organized = raw_organized[raw_organized['ROI_index'].isin(organized['ROI_index'])]
    raw_organized = raw_organized.sort_values(by=['ROI_index'], ascending=True)
    organized = organized[organized['ROI_index'].isin(raw_organized['ROI_index'])]
    organized = organized.sort_values(by=['ROI_index'], ascending=True)
    raw_organized.index = range(raw_organized.shape[0])
    organized.index = range(organized.shape[0])
    assert raw_organized['ROI_index'].equals(organized['ROI_index'])


    sti_df = pd.DataFrame()
    for sample in raw_organized['sample_index'].unique():
        raw_copy = raw_organized.copy()
        org_copy = organized.copy()
        raw_copy = raw_copy[raw_copy['sample_index'] == sample]
        org_copy = org_copy[org_copy['sample_index'] == sample]

        temp_sti_df = pd.DataFrame()
        temp_sti_df['sample_index'] = raw_copy['sample_index']
        temp_sti_df['ROI_index'] = org_copy['ROI_index']
        temp_sti_df['basal'] = raw_copy['{}_f0_mean'.format('KCl')]
        temp_sti_df['std_basal'] = raw_copy['{}_f0_std'.format('KCl')]
        temp_sti_df['df'] = raw_copy['{}_f0_df'.format('KCl')]
        temp_sti_df['ddf'] = raw_copy['{}_f0_ddf'.format('KCl')]

        basal_describe = stats.describe(temp_sti_df['basal'].values.flatten(), nan_policy='omit')
        std_describe = stats.describe(temp_sti_df['std_basal'].values.flatten(), nan_policy='omit')
        df_describe = stats.describe(temp_sti_df['df'].values.flatten(), nan_policy='omit')
        ddf_describe = stats.describe(temp_sti_df['ddf'].values.flatten(), nan_policy='omit')

        temp_sti_df['1_basal'] = basal_describe.mean
        temp_sti_df['1_std_basal'] = std_describe.mean
        temp_sti_df['1_df'] = df_describe.mean
        temp_sti_df['1_ddf'] = ddf_describe.mean

        temp_sti_df['2_basal'] = basal_describe.variance
        temp_sti_df['2_std_basal'] = std_describe.variance
        temp_sti_df['2_df'] = df_describe.variance
        temp_sti_df['2_ddf'] = ddf_describe.variance

        temp_sti_df['3_basal'] = basal_describe.skewness
        temp_sti_df['3_std_basal'] = std_describe.skewness
        temp_sti_df['3_df'] = df_describe.skewness
        temp_sti_df['3_ddf'] = ddf_describe.skewness

        temp_sti_df['4_basal'] = basal_describe.kurtosis
        temp_sti_df['4_std_basal'] = std_describe.kurtosis
        temp_sti_df['4_df'] = df_describe.kurtosis
        temp_sti_df['4_ddf'] = ddf_describe.kurtosis


        temp_sti_df['trace'] = org_copy['KCl_trace']
        stimulus_np = np.concatenate(temp_sti_df['trace'].to_numpy()).reshape(temp_sti_df.shape[0], -1)
        if stimulus_np.shape[1] == 91:
            stimulus_np = stimulus_np[:, 0::4]
        ave_trace = np.mean(stimulus_np, axis=0)
        ave_drv = np.gradient(low_pass_filter(ave_trace, 1, 0.05))
        temp_sti_df['drv_pearson'] = np.apply_along_axis(drv_pearson_correlation, 1, stimulus_np, ave_drv)
        temp_sti_df['AUC'] = np.mean(stimulus_np, axis=1)

        sti_df = sti_df.append(temp_sti_df, ignore_index=True)

    X = sti_df[
        ['basal', 'std_basal', 'df', 'ddf', '1_basal', '1_std_basal', '1_df', '1_ddf',
         '2_basal', '2_std_basal', '2_df', '2_ddf', '3_basal', '3_std_basal', '3_df', '3_ddf', '4_basal', '4_std_basal',
         '4_df', '4_ddf']]
    X = pd.DataFrame(data=stats.zscore(X.values, axis=0), columns=X.columns)

    y = sti_df[['sample_index','ROI_index','drv_pearson', 'AUC']]

    return X, y

def product2label(y_df):
    keep = y_df[(y_df['drv_pearson'] > 0) & (y_df['AUC'] > 0)]
    throw = y_df[(y_df['drv_pearson'] <= 0) | (y_df['AUC'] <= 0)]
    data = np.log(np.multiply(keep['AUC'].values, keep['drv_pearson'].values))

    mixture = GaussianMixture(n_components=3).fit(data.reshape(-1, 1))
    means_hat = mixture.means_
    sds_hat = np.sqrt(mixture.covariances_)

    g_ind = np.argsort(means_hat.flatten())[::-1][0]
    c = np.exp(means_hat[g_ind] - 1.5 * sds_hat[g_ind][0])

    keep['label'] = keep['drv_pearson'] > c / keep['AUC']
    throw['label'] = False
    sti_df = pd.concat((keep, throw)).sort_index()

    return sti_df


x_ls = []
y_ls = []
for train_folder in train_folder_ls:
    X, y = prepare_data(train_folder)
    x_ls.append(X)
    y_ls.append(y)

x_df = pd.DataFrame()
for x in x_ls:
    x_df = x_df.append(x, ignore_index=True)
x_df = x_df.reset_index(drop=True)
y_df = pd.DataFrame()
for y in y_ls:
    y_df = y_df.append(y, ignore_index=True)
y_df = y_df.reset_index(drop=True)

y_df = product2label(y_df)

X = x_df.values[:, :].astype(float)
y = y_df['label'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

print('Train Size: {}'.format(X_train.shape[0]))
print('Test Size: {}'.format(X_test.shape[0]))
print()

clf = svm.SVC(gamma='scale', class_weight='balanced', kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('precision: {}'.format(precision_score(y_test,y_pred)))
print('recall: {}'.format(recall_score(y_test,y_pred)))
print('F1: {}'.format(f1_score(y_test,y_pred)))
print('confusion matrix: {}'.format(confusion_matrix(y_test, y_pred)))
print()

pred_df = pd.DataFrame(X_test[:,0], columns=['Normalized Cell F0'])
pred_df['Normalized Global Variance F0'] = X_test[:,8]
pred_df['pred'] = y_pred
pred_df['label'] = y_test
pred_df['Evaluation'] = 'nan'

pred_df['Response'] = pred_df['label'].map({0: 'No', 1: 'Yes'})

ax = plt.subplot()
ax.set_xlim([pred_df['Normalized Cell F0'].min()-1, pred_df['Normalized Cell F0'].max()+1])
ax.set_ylim([pred_df['Normalized Global Variance F0'].min()-1, pred_df['Normalized Global Variance F0'].max()+1])
sns.scatterplot(ax=ax, x='Normalized Cell F0', y='Normalized Global Variance F0', hue='Response', data=pred_df,
                s=3, edgecolor='k', linewidth=0)

xlim = ax.get_xlim()
w = clf.coef_[0]
a = -w[0] / w[8]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - clf.intercept_[0] / w[8]
ax.plot(xx, yy, 'r--')
# plt.title('Precision: 0.98, Recall: 0.85, F1: 0.91')
plt.savefig('baseline_scatter.svg')
plt.show()

print('precision: {}'.format(precision_score(y_test,y_pred)))
print('recall: {}'.format(recall_score(y_test,y_pred)))
print('F1: {}'.format(f1_score(y_test,y_pred)))
print('confusion matrix: {}'.format(confusion_matrix(y_test, y_pred)))
print()