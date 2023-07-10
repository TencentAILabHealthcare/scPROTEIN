import numpy as np
import pandas as pd
import os 
import argparse
import os.path as osp
import random
from time import perf_counter as t
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
import scipy.sparse as sp
from numpy import linalg as LA
from torch.nn.utils.rnn import pad_sequence
import re
from operator import itemgetter





# load data
def load_peptide(file_path):

    # load raw peptide data
    data_path = os.path.join(file_path)
    # data_path = os.path.join(os.path.abspath('..'),'data/Peptides-raw.csv')
    feature_file = pd.read_csv(data_path)
    feature_fill = feature_file.fillna(0.)
    peptides = list(feature_fill['peptide'])
    proteins = list(feature_fill['protein'])
    expression_data = np.array(feature_fill.iloc[:,2:])
    cell_list = feature_file.columns.tolist()[2:]
    num_cells = len(cell_list)
    print('peptides nums in total:', len(peptides))
    print('cell nums:',num_cells)
    return peptides, proteins, torch.FloatTensor(expression_data), cell_list,num_cells


# extract peptide sequence
def extract_peptide_seq(peptides):
    peptide_list = []
    peptide_list_charge = []
    for i in peptides:
        seq = re.sub(r'\(.*?\)','',i).replace(')', '')
        seq_split = seq.split('_')
        # print(seq_split)
        peptide_list.append(seq_split[1])
        peptide_list_charge.append(seq_split[1]+'_'+seq_split[2])

    return peptide_list,peptide_list_charge



# # extract amino acid set
# def extract_amino_acid_set(peptide_list):
#     amino_acid_set = set()
#     max_length = 0
#     for i in peptide_list:
#         seq_set = set(list(i))
#         amino_acid_set = amino_acid_set.union(seq_set)
#         seq_length = len(list(i))
#         if seq_length>max_length:
#             max_length = seq_length

#     return list(amino_acid_set),max_length



# one hot encoding
def one_hot_encode(peptide_list,amino_acid_dict):
    num_amino_acid = len(amino_acid_dict)  
    peptide_onehot_list = []
    for i in peptide_list:
        peptide_tensor = torch.tensor(itemgetter(*list(i))(amino_acid_dict))
        peptide_onehot = F.one_hot(peptide_tensor,num_classes = num_amino_acid)
        peptide_onehot_list.append(peptide_onehot)
    peptide_onehot_list_padding = pad_sequence(peptide_onehot_list,batch_first = True).permute(0,2,1)
    return peptide_onehot_list_padding.float(),num_amino_acid


def peptide_encode(peptides):

    peptide_list, peptide_list_charge = extract_peptide_seq(peptides)   
    
    # fixed dic
    amino_acid_dict = {'N': 0, 'M': 1, 'K': 2, 'W': 3, 'V': 4, 'C': 5, 'R': 6, 'Q': 7, 'G': 8, 'F': 9, 
                       'P': 10, 'S': 11, 'H': 12, 'Y': 13, 'D': 14, 'T': 15, 'L': 16, 'E': 17, 'A': 18, 'I': 19}

    peptide_onehot_list_padding,num_amino_acid = one_hot_encode(peptide_list,amino_acid_dict)
    return peptide_onehot_list_padding, num_amino_acid




# load batch label
def load_cell_type_labels():
    cell_type_file = pd.read_csv('Cells.csv')
    # print(cell_type_file)
    labels = list(cell_type_file.iloc[0,1:].values)
    batch_labels = list(cell_type_file.iloc[3,1:].values)
    return labels,batch_labels



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