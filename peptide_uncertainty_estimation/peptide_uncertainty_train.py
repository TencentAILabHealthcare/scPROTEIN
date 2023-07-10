import numpy as np
import pandas as pd
import os 
import argparse
import os.path as osp
import random
<<<<<<< HEAD
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
import torch.utils.data as Data

from multi_task_heteroscedastic_regression_loss import regression_loss
from multi_task_heteroscedastic_regression_model import peptide_CNN
from peptide_uncertainty_utils import *



parser = argparse.ArgumentParser()

=======
import torch
import numpy as np 
import sys
# from utils import *

from multi_task_heteroscedastic_regression_model import *
from peptide_uncertainty_utils import *


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

parser = argparse.ArgumentParser()
parser.add_argument("--use_trained_scPROTEIN", type=bool, default=True, help='if use trained scPROTEIN model')
parser.add_argument("--file_path", type=str, default='../data/Peptides-raw.csv', help='data path')
>>>>>>> 25c3e05 (update_scprotein)
parser.add_argument("--learning_rate", type=float, default=1e-3, help='learning rate.')
parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay.')
parser.add_argument("--batch_size", type=int, default=256, help='batch size.')
parser.add_argument("--kernel_nums", type=int, default=[300,200,100], help='kernel num of each conv block.')
parser.add_argument("--kernel_size", type=int, default=[2,2,2], help='kernel size of each conv block.')
parser.add_argument("--max_pool_size", type=int, default=1, help='max pooling size.')
parser.add_argument("--conv_layers", type=int, default=3, help='layer nums of conv.')
parser.add_argument("--hidden_dim", type=int, default=3000, help='hidden dim for fc layer.')
<<<<<<< HEAD
parser.add_argument("--num_cells", type=int, default=1490, help='input cell numbers.')
parser.add_argument("--num_epochs", type=int, default=100, help='number of epochs.')
parser.add_argument("--seed", type=int, default=3047, help='random seed.')
parser.add_argument("--patience", type=int, default=15, help='hidden dimension.')
=======
parser.add_argument("--num_epochs", type=int, default=90, help='number of epochs.')
parser.add_argument("--seed", type=int, default=3047, help='random seed.')
>>>>>>> 25c3e05 (update_scprotein)
parser.add_argument("--split_percentage", type=float, default=0.8, help='split.')
parser.add_argument("--dropout_rate", type=float, default=0.5, help='drop out rate.')

args = parser.parse_args()



<<<<<<< HEAD
        
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] =str(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



=======
>>>>>>> 25c3e05 (update_scprotein)
if __name__ == '__main__':

    setup_seed(args.seed)
    torch.cuda.empty_cache()

<<<<<<< HEAD
    peptides, proteins, Y_label, cell_list = load_peptide()
    peptide_list, peptide_list_charge = extract_peptide_seq(peptides)   
    amino_acid_set,max_length = extract_amino_acid_set(peptide_list)


    # amino_acid_dict = {}
    # for i in range(len(amino_acid_set)):
    #     amino_acid_dict[amino_acid_set[i]] = i
    
    # fixed dic
    amino_acid_dict = {'N': 0, 'M': 1, 'K': 2, 'W': 3, 'V': 4, 'C': 5, 'R': 6, 'Q': 7, 'G': 8, 'F': 9, 
                       'P': 10, 'S': 11, 'H': 12, 'Y': 13, 'D': 14, 'T': 15, 'L': 16, 'E': 17, 'A': 18, 'I': 19}

    peptide_onehot_list_padding,num_amino_acid = one_hot_encode(peptide_list,amino_acid_dict)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    peptide_onehot_list_padding = peptide_onehot_list_padding.to(device)
    Y_label = Y_label.to(device)
    


    indices = list(range(peptide_onehot_list_padding.shape[0]))
    random.shuffle(indices)
    train_size = int((args.split_percentage) * len(indices))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]
    x_train, x_valid = peptide_onehot_list_padding[train_indices], peptide_onehot_list_padding[valid_indices]
    y_train, y_valid = Y_label[train_indices], Y_label[valid_indices]


    train_dataset_split = Data.TensorDataset(x_train, y_train)

    
    loader = Data.DataLoader(
        dataset=train_dataset_split,
        batch_size=args.batch_size,
        shuffle=True)


    model = peptide_CNN(num_amino_acid, args.max_pool_size, args.hidden_dim, 2*args.num_cells, args.conv_layers, args.dropout_rate, args.kernel_nums, args.kernel_size).to(device)
    # criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    

    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(args.num_epochs):
        model.train()
        loss_all = 0
        model.train()

        for step, (batch_x, batch_y) in enumerate(loader):
            loss = 0.
            optimizer.zero_grad()
            y_predict = model(batch_x)


            for i in range(0,2*args.num_cells,2):
                loss_cell = regression_loss(batch_y[:,int(i/2)], y_predict[:,i:i+2])
                loss += loss_cell
            loss = loss/args.num_cells

            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        

        model.eval()
        y_valid_predict = model(x_valid)
        # valid_loss = criterion(y_valid_predict, y_valid)

        print('epoch {}, training_loss {}'.format(epoch, loss_all))

        # early stop
        if loss_all < best:
            best = loss_all
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_peptide_uncertainty.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    print('load best model')
    model.load_state_dict(torch.load('best_peptide_uncertainty.pkl'))
    model.eval()

    y_predict_all = model(peptide_onehot_list_padding)
    y_predict_all = y_predict_all.cpu().detach().numpy()

    log_uncertainty = y_predict_all[:,1::2]
    uncertainty = np.exp(log_uncertainty)
    np.save('peptide_uncertainty.npy',uncertainty)
=======
    peptides, proteins, Y_label, cell_list, num_cells = load_peptide(args.file_path)
    peptide_onehot_padding, num_amino_acid = peptide_encode(peptides)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    peptide_onehot_padding = peptide_onehot_padding.to(device)
    Y_label = Y_label.to(device)

    
    model = peptide_CNN(num_amino_acid, args.max_pool_size, args.hidden_dim, 2*num_cells, args.conv_layers, args.dropout_rate, args.kernel_nums, args.kernel_size).to(device)
    
    if args.use_trained_scPROTEIN:
        model.load_state_dict(torch.load('../trained_scPROTEIN/scPROTEIN_stage1.pkl'))
        scPROTEIN_stage1 = scPROTEIN_stage1_learning(model, peptide_onehot_padding, Y_label,args.learning_rate, args.weight_decay, args.split_percentage, args.num_epochs, args.batch_size)


    else:
        scPROTEIN_stage1 = scPROTEIN_stage1_learning(model, peptide_onehot_padding, Y_label,args.learning_rate, args.weight_decay, args.split_percentage, args.num_epochs, args.batch_size)
        scPROTEIN_stage1.train()

    
    

    uncertainty = scPROTEIN_stage1.uncertainty_generation()
    np.save('peptide_uncertainty_data.npy',uncertainty)
>>>>>>> 25c3e05 (update_scprotein)

    
    


    


