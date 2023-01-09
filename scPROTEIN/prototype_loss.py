import numpy as np
import torch
from numpy import linalg as LA


def get_proto_norm(feature, centroid, ps_label, num_protos):
    num_data = feature.shape[0]
    each_cluster_num = np.zeros([num_protos])

    for i in range(num_protos):
        each_cluster_num[i] = np.sum(ps_label==i)
    
    proto_norm_term = np.zeros([num_protos])
    for i in range(num_protos):
        norm_sum = 0
        for j in range(num_data):
            if ps_label[j] == i:
                norm_sum = norm_sum + LA.norm(feature[j] - centroid[i], 2)
        proto_norm_term[i] = norm_sum / (each_cluster_num[i] * np.log2(each_cluster_num[i] + 10))
    proto_norm = torch.Tensor(proto_norm_term)
    return proto_norm



def get_proto_loss(feature, centroid, ps_label, proto_norm):

    feature_norm = torch.norm(feature, dim=-1)
    feature = torch.div(feature, feature_norm.unsqueeze(1)) 
    centroid_norm = torch.norm(centroid, dim=-1)
    centroid = torch.div(centroid, centroid_norm.unsqueeze(1))
    sim_zc = torch.matmul(feature, centroid.t())   
    sim_zc_normalized = torch.exp(sim_zc)
    sim_2centroid = torch.gather(sim_zc_normalized, -1, ps_label) 
    sim_sum = torch.sum(sim_zc_normalized, -1, keepdim=True)
    sim_2centroid = torch.div(sim_2centroid, sim_sum)
    loss = torch.mean(sim_2centroid.log())
    loss = -1 * loss
    return loss

