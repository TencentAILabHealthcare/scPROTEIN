import argparse
import random
from time import perf_counter as t

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from utils import *
from prototype_loss import *
from model import Encoder, Model, drop_feature

import numpy as np 
import scipy.sparse as sp

from sklearn.cluster import KMeans



parser = argparse.ArgumentParser()
parser.add_argument("--stage1", type=bool, default='True', help='if scPROTEIN starts from stage1')

parser.add_argument("--learning_rate", type=float, default=1e-3, help='learning rate')
parser.add_argument("--num_hidden", type=int, default=400, help='hidden dimension')
parser.add_argument("--num_proj_hidden", type=int, default=256, help='dimension of projection head')
parser.add_argument("--activation", type=str, default='prelu', help='activation function')
parser.add_argument("--base_model", type=str, default='GCNConv', help='base encoding model')
parser.add_argument("--num_layers", type=int, default=2, help='num of GCN layers')
parser.add_argument("--num_protos", type=int, default=2, help='num of prototypes')
parser.add_argument("--num_changed_edges", type=int, default=50, help='num of added/removed edges')
parser.add_argument("--topology_denoising", type=bool, default=False, help='if scPROTEIN uses topology denoising')

parser.add_argument("--drop_edge_rate_1", type=float, default=0.2, help='dropedge rate for view1')
parser.add_argument("--drop_edge_rate_2", type=float, default=0.4, help='dropedge rate for view2')
parser.add_argument("--drop_feature_rate_1", type=float, default=0.4, help='mask_feature rate for view1')
parser.add_argument("--drop_feature_rate_2", type=float, default=0.2, help='mask_feature rate for view2')

parser.add_argument("--alpha", type=float, default=0.05, help='balance factor')
parser.add_argument("--tau", type=float, default=0.4, help='temperature coefficient')
parser.add_argument("--weight_decay", type=float, default=0.00001, help='weight_decay')

parser.add_argument("--num_epochs", type=int, default=120, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=39788, help='Random seed.')
parser.add_argument("--threshold", type=float, default=0.15, help='threshold of graph construct')
parser.add_argument("--feature_preprocess", type=bool, default=True, help='feature preprocess')

parser.add_argument("--patience", type=int, default=30, help='Hidden dimension.')
args = parser.parse_args()





def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)
    return z


if __name__ == '__main__':

    torch.manual_seed(args.seed)
    learning_rate = args.learning_rate
    num_hidden = args.num_hidden
    num_proj_hidden = args.num_proj_hidden
    if args.activation == 'relu':
        activation = F.relu
    elif args.activation == 'prelu':
        activation = nn.PReLU()

    if args.base_model == 'GCNConv':
        base_model = GCNConv

    num_layers = args.num_layers
    drop_edge_rate_1 = args.drop_edge_rate_1
    drop_edge_rate_2 = args.drop_edge_rate_2
    drop_feature_rate_1 = args.drop_feature_rate_1
    drop_feature_rate_2 = args.drop_feature_rate_2
    tau = args.tau
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay


    _, _, features = load_sc_proteomic_features(args.stage1)   
    features = features.astype(float)  
    features_pd = pd.DataFrame(features)
    adj = features_pd.corr()
    adj_matrix = np.where(adj>args.threshold,1,0)
    features = features.T
    features = sp.coo_matrix(features)
    if args.feature_preprocess:
        features, _ = preprocess_features(features)


    adj_matrix_sp = sp.coo_matrix(adj_matrix)
    edge_index = torch.tensor(np.vstack((adj_matrix_sp.row, adj_matrix_sp.col)), dtype=torch.long)
    features = torch.tensor(features,dtype=torch.float32)
    data = Data(x=features, edge_index=edge_index)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    torch.cuda.empty_cache()

    encoder = Encoder(data.num_features, num_hidden, activation,base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    best = 1e9
    best_t = 0
    cnt_wait = 0
    for epoch in range(1, num_epochs + 1):
        loss_node = train(model, data.x, data.edge_index)

        # attribute denoising
        with torch.no_grad():
            embedding = test(model, data.x, data.edge_index, data.y, final=True)
            embedding = embedding.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=args.num_protos, n_init=20).fit(embedding)
            label_kmeans = kmeans.labels_
            centers = np.array([np.mean(embedding[label_kmeans == i,:], axis=0)
                              for i in range(args.num_protos)])
            label_kmeans = label_kmeans[:, np.newaxis]
            proto_norm = get_proto_norm(embedding, centers,label_kmeans,args.num_protos)
            embedding = torch.Tensor(embedding)
            centers = torch.Tensor(centers)
            label_kmeans = torch.Tensor(label_kmeans).long()
            proto_norm = torch.Tensor(proto_norm)
            loss_proto = get_proto_loss(embedding, centers, label_kmeans, proto_norm)

        # topology denoising
        if args.topology_denoising:
            with torch.no_grad():
                embedding = sp.coo_matrix(embedding)
                similarity_matrix = embedding.dot(embedding.transpose())
                similarity_matrix = similarity_matrix.tocoo()
                coords = list(np.vstack((similarity_matrix.row, similarity_matrix.col)).transpose())
                coords_tuple = []
                for i in coords:
                    coords_tuple.append(tuple(i))

                simi_data = list(similarity_matrix.data)
                coord_value_dict = {}
                for i in range(len(coords_tuple)):
                    coord_value_dict[coords_tuple[i]] = simi_data[i]

                coord_value_dict_wo_diag = {}
                cnt = 0
                for key,value in coord_value_dict.items():
                    if key[0] == key[1]:
                        cnt += 1
                    else:
                        coord_value_dict_wo_diag[key] = value

                coords_wo_diag = list(coord_value_dict_wo_diag.keys())
                simi_data_wo_diag = np.array(list(coord_value_dict_wo_diag.values()))
                simi_sort = simi_data_wo_diag.argsort()
                high_prob_indices = list(simi_sort[-args.num_changed_edges:])
                low_prob_indices = list(simi_sort[:args.num_changed_edges])
                high_prob_coords = []
                low_prob_coords = []
                for i in high_prob_indices:
                    high_prob_coords.append(coords_wo_diag[i])
                for i in low_prob_indices:
                    low_prob_coords.append(coords_wo_diag[i])

                edge_index_now = list(data.edge_index.cpu().detach().numpy().T)
                edge_index_now_list = []
                for i in edge_index_now:
                    edge_index_now_list.append(tuple(i))
                cnt_add = 0
                for i in high_prob_coords:
                    if i not in edge_index_now_list:
                        edge_index_now_list.append(i)
                        cnt_add += 1
                cnt_remove = 0
                for i in low_prob_coords:
                    if i in edge_index_now_list:
                        edge_index_now_list.remove(i)
                        cnt_remove += 1

                edge_index = torch.tensor(np.array(edge_index_now_list).T, dtype=torch.long).to(device)
                data.edge_index = edge_index


        loss = loss_node + args.alpha*loss_proto

        if loss < best:
            best = loss
            best_t = epoch + 1
            cnt_wait = 0
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f},'
              f'loss node={loss_node:.4f},'
              f'loss proto={loss_proto:.4f}')
        prev = now

    print("=== saving learned cell embedding ===")
    embedding = test(model, data.x, data.edge_index, data.y, final=True)
    print(embedding.shape)
    np.save('scPROTEIN_embedding.npy', embedding.cpu().detach().numpy())
    