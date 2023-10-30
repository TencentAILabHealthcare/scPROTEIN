import numpy as np
import pandas as pd
import os 
import argparse
import os.path as osp
import random
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
parser.add_argument("--learning_rate", type=float, default=1e-3, help='learning rate.')
parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay.')
parser.add_argument("--batch_size", type=int, default=256, help='batch size.')
parser.add_argument("--kernel_nums", type=int, default=[300,200,100], help='kernel num of each conv block.')
parser.add_argument("--kernel_size", type=int, default=[2,2,2], help='kernel size of each conv block.')
parser.add_argument("--max_pool_size", type=int, default=1, help='max pooling size.')
parser.add_argument("--conv_layers", type=int, default=3, help='layer nums of conv.')
parser.add_argument("--hidden_dim", type=int, default=3000, help='hidden dim for fc layer.')
parser.add_argument("--num_epochs", type=int, default=90, help='number of epochs.')
parser.add_argument("--seed", type=int, default=3047, help='random seed.')
parser.add_argument("--split_percentage", type=float, default=0.8, help='split.')
parser.add_argument("--dropout_rate", type=float, default=0.5, help='drop out rate.')

args = parser.parse_args()



if __name__ == '__main__':

    setup_seed(args.seed)
    torch.cuda.empty_cache()

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

    
    
    scPROTEIN_stage1.uncertainty_generation()

    
    


    


