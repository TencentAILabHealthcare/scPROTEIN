import os
import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.data import Data
import scipy.sparse as sp






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
        proteins_all = list(set(feature_fill['protein']))
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

    return proteins_all, cell_list, features




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
    cell_type_file = pd.read_csv('./scope2/Cells.csv')
    labels = list(cell_type_file.iloc[0,1:].values)
    return labels


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)



