import argparse
import random
import sys
from scprotein import *
import numpy as np 
import scipy.sparse as sp
import os
import anndata as ad
from sklearn import metrics
from sklearn.metrics import silhouette_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
import warnings

warnings.filterwarnings('ignore')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

parser = argparse.ArgumentParser()

parser.add_argument("--use_trained_scPROTEIN", type=bool, default=False, help='if use trained scPROTEIN model')
parser.add_argument("--stage1", type=bool, default=False, help='if scPROTEIN starts from stage1')
parser.add_argument("--learning_rate", type=float, default=1e-3, help='learning rate')
parser.add_argument("--num_hidden", type=int, default=256, help='hidden dimension') 
parser.add_argument("--num_proj_hidden", type=int, default=256, help='dimension of projection head')
parser.add_argument("--activation", type=str, default='prelu', help='activation function') 
parser.add_argument("--num_layers", type=int, default=2, help='num of GCN layers')
parser.add_argument("--num_protos", type=int, default=4, help='num of prototypes')
parser.add_argument("--num_changed_edges", type=int, default=10, help='num of added/removed edges')
parser.add_argument("--topology_denoising", type=bool, default=False, help='if scPROTEIN uses topology denoising')
parser.add_argument("--drop_edge_rate_1", type=float, default=0.2, help='dropedge rate for view1')
parser.add_argument("--drop_edge_rate_2", type=float, default=0.4, help='dropedge rate for view2')
parser.add_argument("--drop_feature_rate_1", type=float, default=0.4, help='mask_feature rate for view1')
parser.add_argument("--drop_feature_rate_2", type=float, default=0.2, help='mask_feature rate for view2')
parser.add_argument("--alpha", type=float, default=0.0, help='balance factor')
parser.add_argument("--tau", type=float, default=0.4, help='temperature coefficient')
parser.add_argument("--weight_decay", type=float, default=0.00001, help='weight_decay')
parser.add_argument("--num_epochs", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=39788, help='Random seed.') 
parser.add_argument("--threshold", type=float, default=0.5, help='threshold of graph construct')
parser.add_argument("--feature_preprocess", type=bool, default=False, help='feature preprocess')
args = parser.parse_args()



if __name__ == '__main__':

    setup_seed(args.seed)
    activation = nn.PReLU() if args.activation == 'prelu' else F.relu
    
    adata = ad.read_h5ad('./data/T-SCP.h5ad')
    features = adata.X
    print('feature shape:',features.shape)
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = graph_generation(features, args.threshold, args.feature_preprocess).to(device)
    torch.cuda.empty_cache()
    encoder = Encoder(data.num_features, args.num_hidden, activation, k=args.num_layers).to(device)


    model = Model(encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(device)
    scPROTEIN = scPROTEIN_learning(model,device, data, args.drop_feature_rate_1,args.drop_feature_rate_2,args.drop_edge_rate_1,args.drop_edge_rate_2,
                args.learning_rate, args.weight_decay, args.num_protos, args.topology_denoising, args.num_epochs, args.alpha, args.num_changed_edges,args.seed)

    scPROTEIN.train()

    embedding = scPROTEIN.embedding_generation()
    np.save('embedding_T-SCP.npy', embedding)



    