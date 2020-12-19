import pandas as pd
import numpy as np
from sys import exit
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
#
#

train_folder_ls = []
test_folder_ls = []

# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\AstC\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\CCHa2\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros1\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros2\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros3\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros4\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_ind\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_NEAA\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_Dosage\\')
# train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_KCl1\\')
train_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_KCl2\\')

model_name = 'Dtest'

# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\AstC\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros1\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros2\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros3\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Pros4\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_ind\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_NEAA\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_Dosage\\')
# test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_KCl1\\')
test_folder_ls.append('D:\\Gut Imaging\\Videos\\svm\\Dh31_KCl2\\')

def prepare_data(input_folder):
    organized = pd.read_pickle(input_folder + input_folder.split('\\')[-2] + '_organized_data.pkl')
    raw_organized = pd.read_pickle(input_folder + 'Raw_' + input_folder.split('\\')[-2] + '_organized_data.pkl')
    organized = organized[~organized["ROI_index"].str.contains('_b')]
    raw_organized = raw_organized[raw_organized['ROI_index'].isin(organized['ROI_index'])]
    raw_organized = raw_organized.sort_values(by=['ROI_index'], ascending=True)
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
        temp_sti_df['ROI_index'] = org_copy['ROI_index']
        temp_sti_df['basal'] = raw_copy['{}_f0_mean'.format('KCl')]
        temp_sti_df['std_basal'] = raw_copy['{}_f0_std'.format('KCl')]
        temp_sti_df['cv'] = temp_sti_df['std_basal']/temp_sti_df['basal']
        temp_sti_df['average_response'] = org_copy['{}_average_response'.format('KCl')]
        temp_sti_df['average_basal'] = temp_sti_df['basal'].mean()
        temp_sti_df['median_basal'] = temp_sti_df['basal'].median()
        temp_sti_df['std_basal'] = temp_sti_df['basal'].std()
        temp_sti_df['n_mean_basal'] = temp_sti_df['basal']/temp_sti_df['basal'].mean()
        temp_sti_df['n_median_basal'] = temp_sti_df['basal']/temp_sti_df['basal'].median()
        temp_sti_df['std_n_mean_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].mean())/temp_sti_df['basal'].std()
        temp_sti_df['std_n_median_basal'] = (temp_sti_df['basal'] - temp_sti_df['basal'].median())/temp_sti_df['basal'].std()

        sti_df = sti_df.append(temp_sti_df, ignore_index=True)

    X = np.hstack((sti_df['basal'].to_numpy().reshape((-1, 1)), sti_df['std_basal'].to_numpy().reshape((-1, 1)), sti_df['cv'].to_numpy().reshape((-1, 1)), sti_df['average_basal'].to_numpy().reshape((-1, 1)),
                   sti_df['median_basal'].to_numpy().reshape((-1, 1)), sti_df['std_basal'].to_numpy().reshape((-1, 1)),
                   sti_df['n_mean_basal'].to_numpy().reshape((-1, 1)), sti_df['n_median_basal'].to_numpy().reshape((-1, 1)),
                   sti_df['std_n_mean_basal'].to_numpy().reshape((-1, 1)), sti_df['std_n_median_basal'].to_numpy().reshape((-1, 1))))

    y = sti_df['average_response'].to_numpy().flatten()

    return X, y

def train(x_ls, y_ls):
    X = np.vstack(tuple(x_ls))
    y = np.vstack(tuple(y_ls))

    print('Train Size: {}'.format(X.shape[0]))
    print()

    clf = svm.SVC(gamma='scale', class_weight='balanced', kernel='linear')
    clf.fit(X, y)
    return clf

def test(x_ls, y_ls, clf):
    X = np.vstack(tuple(x_ls))
    y = np.vstack(tuple(y_ls))

    print('Test Size: {}'.format(X.shape[0]))
    print()

    y_pred = clf.predict(X)
    print('precision: {}'.format(precision_score(y, y_pred)))
    print('recall: {}'.format(recall_score(y, y_pred)))
    print('F1: {}'.format(f1_score(y, y_pred)))
    print('confusion matrix: {}'.format(confusion_matrix(y, y_pred)))
    print()
    return y_pred

def train_test(x_ls, y_ls):
    X = np.vstack(x_ls)
    y = np.vstack(y_ls)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

    print('Train Size: {}'.format(X_train.shape[0]))
    print('Test Size: {}'.format(X_test.shape[0]))
    print()

    clf = svm.SVC(gamma='scale', class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('precision: {}'.format(precision_score(y_test,y_pred)))
    print('recall: {}'.format(recall_score(y_test,y_pred)))
    print('F1: {}'.format(f1_score(y_test,y_pred)))
    print('confusion matrix: {}'.format(confusion_matrix(y_test, y_pred)))
    print()

train_x = []
train_y = []
for train_folder in train_folder_ls:
    X, y = prepare_data(train_folder)
    train_x.append(X)
    train_y.append(y.reshape(-1, 1))

test_x = []
test_y = []
for test_folder in test_folder_ls:
    X, y = prepare_data(test_folder)
    test_x.append(X)
    test_y.append(y.reshape(-1, 1))

clf = train(train_x, train_y)
y_pred = test(test_x, test_y, clf)

# train_test(train_x, train_y)
# print(clf.coef_)
# print(clf.intercept_)

dict_name = 'D:\\Gut Imaging\\Videos\\CommonFiles\\svm_models.pkl'

if os.path.exists(dict_name):
    with open(dict_name, 'rb') as handle:
        model_dict = pickle.load(handle)
else:
    model_dict = {}

model_dict.update({'{}_coef'.format(model_name): clf.coef_})
model_dict.update({'{}_intercept'.format(model_name): clf.intercept_})

with open(dict_name, 'wb') as handle:
    pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)