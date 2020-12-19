import pandas as pd
from organized_Clustering_custom import h_cluster_multi

scheme = pd.read_excel('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\Mixture\\DeliverySchemes.xlsx')
stimulus_ls = scheme.iloc[:, 1].to_list()



Pros_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\Pros\\Pros_organized_data.pkl')
CCHa2_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\CCHa2\\CCHa2_organized_data.pkl')
AstA_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\AstA\\AstA_organized_data.pkl')
AstC_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\AstC\\AstC_organized_data.pkl')
Tk_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\Tk\\Tk_organized_data.pkl')
Dh31_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\Dh31\\Dh31_organized_data.pkl')
CCHa2_Mix_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\CCHa2_Mix\\CCHa2_Mix_organized_data.pkl')
Mixture_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\Mixture\\Mixture_organized_data.pkl')
NEAACont_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\NEAACont\\NEAACont_organized_data.pkl')
DgluCont_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\DgluCont\\DgluCont_organized_data.pkl')
AstC_Mix_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\AstC_Mix\\AstC_Mix_organized_data.pkl')
Dh31_Mix_org = pd.read_pickle('D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\Dh31_Mix\\Dh31_Mix_organized_data.pkl')

Pros_org['genotype'] = 'Pros'
CCHa2_org['genotype'] = 'CCHa2'
AstA_org['genotype'] = 'AstA'
AstC_org['genotype'] = 'AstC'
Tk_org['genotype'] = 'Tk'
Dh31_org['genotype'] = 'Dh31'
CCHa2_Mix_org['genotype'] = 'CCHa2_Mix'
Mixture_org['genotype'] = 'Mixture'
NEAACont_org['genotype'] = 'NEAACont'
DgluCont_org['genotype'] = 'DgluCont'
AstC_Mix_org['genotype'] = 'AstC_Mix'
Dh31_Mix_org['genotype'] = 'Dh31_Mix'

h_cluster_multi([CCHa2_Mix_org, Mixture_org, AstC_Mix_org, Dh31_Mix_org], 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\',scheme, stimulus_ls, threshold=4.6, n_pc_ls=range(5,26),norm=1)
# h_cluster_multi([DgluCont_org], 'D:\\Gut Imaging\\Videos\\Temp_UMAP\\4clustering_Gal4_new\\',scheme, stimulus_ls, threshold=4.6,norm=1)