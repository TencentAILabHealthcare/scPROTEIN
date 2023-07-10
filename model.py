import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
<<<<<<< HEAD
=======
from torch_geometric.utils import dropout_adj
from prototype_loss import *
from sklearn.cluster import KMeans
from utils import *
>>>>>>> 25c3e05 (update_scprotein)

'''
part of code is borrowed from https://github.com/CRIPAC-DIG/GRACE
'''

<<<<<<< HEAD
=======

# mask feature function
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

>>>>>>> 25c3e05 (update_scprotein)
# GCN encoder
class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
<<<<<<< HEAD
            x = self.activation(self.conv[i](x, edge_index))
=======
            # print(self.conv[i](x, edge_index))
            x = self.activation(self.conv[i](x, edge_index))
            # print(x)
>>>>>>> 25c3e05 (update_scprotein)
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):

        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    # loss definition
    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


<<<<<<< HEAD
# mask feature function
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
=======

class scPROTEIN_learning(torch.nn.Module):

    def __init__(self, model,device, data, drop_feature_rate_1,drop_feature_rate_2,drop_edge_rate_1,drop_edge_rate_2,
                 learning_rate, weight_decay, num_protos, topology_denoising, num_epochs, alpha, num_changed_edges, seed):
        super(scPROTEIN_learning, self).__init__()
        self.model = model
        self.data = data
        self.device = device
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_protos = num_protos
        self.topology_denoising = topology_denoising
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.seed = seed
        self.num_changed_edges = num_changed_edges
        
    def train(self):
        setup_seed(self.seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            
            optimizer.zero_grad()
            edge_index_1 = dropout_adj(self.data.edge_index, p=self.drop_edge_rate_1)[0]
            edge_index_2 = dropout_adj(self.data.edge_index, p=self.drop_edge_rate_2)[0]
            x_1 = drop_feature(self.data.x, self.drop_feature_rate_1)
            x_2 = drop_feature(self.data.x, self.drop_feature_rate_2)
            z1 = self.model(x_1, edge_index_1)
            z2 = self.model(x_2, edge_index_2)
            loss_node = self.model.loss(z1, z2, batch_size=0)
            # return loss



            # embedding = test(model, data.x.to(device), data.edge_index.to(device))
            embedding = self.test()
            embedding_cpu = embedding.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=self.num_protos).fit(embedding_cpu)
            label_kmeans = kmeans.labels_
            centers = np.array([np.mean(embedding_cpu[label_kmeans == i,:], axis=0)
                                for i in range(self.num_protos)])
            label_kmeans = label_kmeans[:, np.newaxis]
            proto_norm = get_proto_norm(embedding_cpu, centers,label_kmeans,self.num_protos)
            centers = torch.Tensor(centers).to(self.device)
            label_kmeans = torch.Tensor(label_kmeans).long().to(self.device)
            proto_norm = torch.Tensor(proto_norm).to(self.device)
            loss_proto = get_proto_loss(embedding, centers, label_kmeans, proto_norm)       



            # topology denoising
            if self.topology_denoising:
                with torch.no_grad():
                    embedding_cpu = sp.coo_matrix(embedding_cpu)
                    similarity_matrix = embedding_cpu.dot(embedding_cpu.transpose())
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
                    high_prob_indices = list(simi_sort[-self.num_changed_edges:])
                    low_prob_indices = list(simi_sort[:self.num_changed_edges])
                    high_prob_coords = []
                    low_prob_coords = []
                    for i in high_prob_indices:
                        high_prob_coords.append(coords_wo_diag[i])
                    for i in low_prob_indices:
                        low_prob_coords.append(coords_wo_diag[i])

                    edge_index_now = list(self.data.edge_index.cpu().detach().numpy().T)
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

                    edge_index = torch.tensor(np.array(edge_index_now_list).T, dtype=torch.long).to(self.device)
                    self.data.edge_index = edge_index    
            
            
            loss = loss_node + self.alpha*loss_proto
            loss.backward()
            optimizer.step()

            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f} ')


    def test(self):
        self.model.eval()
        z = self.model(self.data.x, self.data.edge_index)
        return z

    def embedding_generation(self):
        self.model.eval()
        z = self.model(self.data.x, self.data.edge_index)
        return z.cpu().detach().numpy()







>>>>>>> 25c3e05 (update_scprotein)
