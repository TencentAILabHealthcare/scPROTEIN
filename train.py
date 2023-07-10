import argparse
import random

from utils import *
from model import *
import numpy as np 
import scipy.sparse as sp
import os

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
parser.add_argument("--num_hidden", type=int, default=400, help='hidden dimension') 
parser.add_argument("--num_proj_hidden", type=int, default=256, help='dimension of projection head')
parser.add_argument("--activation", type=str, default='prelu', help='activation function') 
parser.add_argument("--num_layers", type=int, default=2, help='num of GCN layers')
parser.add_argument("--num_protos", type=int, default=2, help='num of prototypes')
parser.add_argument("--num_changed_edges", type=int, default=10, help='num of added/removed edges')
parser.add_argument("--topology_denoising", type=bool, default=False, help='if scPROTEIN uses topology denoising')
parser.add_argument("--drop_edge_rate_1", type=float, default=0.2, help='dropedge rate for view1')
parser.add_argument("--drop_edge_rate_2", type=float, default=0.4, help='dropedge rate for view2')
parser.add_argument("--drop_feature_rate_1", type=float, default=0.4, help='mask_feature rate for view1')
parser.add_argument("--drop_feature_rate_2", type=float, default=0.2, help='mask_feature rate for view2')
parser.add_argument("--alpha", type=float, default=0.05, help='balance factor')
parser.add_argument("--tau", type=float, default=0.4, help='temperature coefficient')
parser.add_argument("--weight_decay", type=float, default=0.00001, help='weight_decay')
parser.add_argument("--num_epochs", type=int, default=200, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=39788, help='Random seed.') 
parser.add_argument("--threshold", type=float, default=0.15, help='threshold of graph construct')
parser.add_argument("--feature_preprocess", type=bool, default=True, help='feature preprocess')
args = parser.parse_args()



if __name__ == '__main__':

    setup_seed(args.seed)
    activation = nn.PReLU() if args.activation == 'prelu' else F.relu

    _, _, features = load_sc_proteomic_features(args.stage1)   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = graph_generation(features, args.threshold, args.feature_preprocess).to(device)
    torch.cuda.empty_cache()
    encoder = Encoder(data.num_features, args.num_hidden, activation, k=args.num_layers).to(device)

    if args.use_trained_scPROTEIN:
        model = torch.load('./trained_scPROTEIN/scPROTEIN_stage2.pt').to(device)
        scPROTEIN = scPROTEIN_learning(model,device, data, args.drop_feature_rate_1,args.drop_feature_rate_2,args.drop_edge_rate_1,args.drop_edge_rate_2,
                    args.learning_rate, args.weight_decay, args.num_protos, args.topology_denoising, args.num_epochs, args.alpha, args.num_changed_edges,args.seed)

    else:
        model = Model(encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(device)
        scPROTEIN = scPROTEIN_learning(model,device, data, args.drop_feature_rate_1,args.drop_feature_rate_2,args.drop_edge_rate_1,args.drop_edge_rate_2,
                    args.learning_rate, args.weight_decay, args.num_protos, args.topology_denoising, args.num_epochs, args.alpha, args.num_changed_edges,args.seed)

        scPROTEIN.train()
        # print("=== saving learned cell embedding ===")
        # torch.save(model, './pretrained_scPROTEIN/scPROTEIN_stage2.pt')
    

    embedding = scPROTEIN.embedding_generation()
    np.save('scPROTEIN_embedding.npy', embedding)
    
