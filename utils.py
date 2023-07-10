import os
import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.data import Data
import scipy.sparse as sp
import random
import scanpy as sc
from operator import itemgetter



def load_sc_proteomic_features(stage1):

    # -----for data provided with peptide-level data, scPROTEIN starts from stage 1-----
    if stage1:
        feature_file = pd.read_csv('./data/Peptides-raw.csv')

        if os.path.exists('./peptide_uncertainty_estimation/peptide_uncertainty.npy'):
            peptide_uncertainty = np.load('./peptide_uncertainty_estimation/peptide_uncertainty.npy')

        else:
            # print('-----To start from stage 1, you have to run peptide_uncertainty.py in the folder peptide_uncertainty_estimation to obtain the estimated peptide uncertainty first-----')
            raise Exception('-----To start from stage 1, you have to run peptide_uncertainty.py in the folder peptide_uncertainty_estimation to obtain the estimated peptide uncertainty first-----')

        cell_list = feature_file.columns.tolist()[2:]
        feature_fill = feature_file.fillna(0.)
        proteins_all = list(pd.unique(feature_fill['protein']))
        features_all = feature_fill.values[:,2:]
        weighted_protein_feature_all = []



        # obtain the uncertainty-guided protein-level data
        for i in proteins_all:
            protein_index = feature_fill[(feature_fill.protein==i)].index.tolist()
            peptide_uncertainty_subset = peptide_uncertainty[protein_index,:]
            peptide_scores = 1./peptide_uncertainty_subset
            peptide_features_subset = features_all[protein_index,:]

            weighted_peptide_features_subset = np.multiply(peptide_features_subset, peptide_scores)
            weighted_protein_feature = np.sum(weighted_peptide_features_subset, axis=0)
            weighted_protein_feature_all.append(weighted_protein_feature)
        
        features = np.array(weighted_protein_feature_all)

    # -----for data provided directly from protein-level data, scPROTEIN starts from stage 2-----
    else:
        feature_file = pd.read_csv('./data/Peptides-raw.csv')
        cell_list = feature_file.columns.tolist()[2:]
        feature_fill = feature_file.fillna(0.)
        features = feature_fill.groupby('protein').sum()
        proteins_all = list(features.index)
        features = np.array(features.values)

    return proteins_all, cell_list, features.T


def graph_generation(features, threshold, feature_preprocess):
    # take cell*protein matrix as input
    # output cell*cell graph
    features = features.astype(float)  
    features_pd = pd.DataFrame(features.T)
    adj = features_pd.corr()
    adj_matrix = np.where(adj>threshold,1,0)

    if feature_preprocess:
        features = sp.coo_matrix(features)
        features, _ = preprocess_features(features)

    adj_matrix_sp = sp.coo_matrix(adj_matrix)
    edge_index = torch.tensor(np.vstack((adj_matrix_sp.row, adj_matrix_sp.col)), dtype=torch.long)
    features = torch.tensor(features,dtype=torch.float32)

    data = Data(x=features, edge_index=edge_index)
    return data


def preprocess_graph(adj):

    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized, sparse_to_tuple(adj_normalized)


def sparse_to_tuple(sparse_mx):

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def load_cell_type_labels():
    cell_type_file = pd.read_csv('./data/Cells.csv')
    labels = list(cell_type_file.iloc[0,1:].values)
    return labels





# load overlap protein data from two datasets
def integrate_sc_proteomic_features(dataset1, dataset2):
    
    # load individual scp data
    adata1 = sc.read_h5ad('./integration_dataset/{}.h5ad'.format(dataset1))
    adata2 = sc.read_h5ad('./integration_dataset/{}.h5ad'.format(dataset2))
    protein_data1, protein_data2 = adata1.X, adata2.X
    protein_data1 = np.nan_to_num(protein_data1)
    protein_data2 = np.nan_to_num(protein_data2)
    cell_num1, cell_num2 = protein_data1.shape[0], protein_data2.shape[0]
    proteins1, proteins2 = list(adata1.var_names),list(adata2.var_names)
    
    # define batch label and cell type labels for both two datasets
    batch_label = np.concatenate((np.zeros(cell_num1),np.ones(cell_num2))).astype(int)
    cell_type1,cell_type2 = list(adata1.obs['cell_type']), list(adata2.obs['cell_type'])
    cell_type_with_dataname = cell_type1+cell_type2
    
    cell_type1 = [i.split('(')[0] for i in cell_type1]
    cell_type2 = [i.split('(')[0] for i in cell_type2]
    overlap_cell_type = list(set(cell_type1) & set(cell_type2))
    print('overlap celltype:',overlap_cell_type)
    
    cell_type_all = cell_type1+cell_type2
    cell_type_dic = dict(zip(set(cell_type_all), range(len(set(cell_type_all)))))
    cell_type_label = np.array(itemgetter(*list(cell_type_all))(cell_type_dic))
    overlap_cell_type_label = [cell_type_dic[i] for i in overlap_cell_type]
    
    
    # search overlap protein from both two datasets
    proteins1_pd,proteins2_pd = pd.DataFrame(proteins1,columns=['protein_name']),pd.DataFrame(proteins2,columns=['protein_name'])
    overlap_protein = pd.merge(proteins1_pd, proteins2_pd, on=['protein_name'])
    overlap_protein = list(overlap_protein['protein_name'])
    print('overlap protein nums:',len(overlap_protein))
                           

    # construct overlap protein features
    features_concat = np.zeros((cell_num1+cell_num2,len(overlap_protein)))
    for i,protein in enumerate(overlap_protein):
        index1 = proteins1.index(protein)
        protein_data1_slice = protein_data1[:,index1]
        index2 = proteins2.index(protein)
        protein_data2_slice = protein_data2[:,index2]
        protein_data_slice_concat = np.concatenate([protein_data1_slice,protein_data2_slice])
        features_concat[:,i] = protein_data_slice_concat    
    
    return batch_label,cell_type_with_dataname,cell_type_label,overlap_cell_type_label, features_concat

def setup_seed(seed):
    #--- Fix random seed ---#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

