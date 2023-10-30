from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  
import sys
sys.path.append("../../") 
from utils import *
from operator import itemgetter
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn import metrics
from sklearn.metrics import silhouette_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

import warnings


warnings.filterwarnings("ignore")
seed = 0


def dimension_reduce(embedding):
    X_trans_PCA = PCA(n_components=50, random_state=seed).fit_transform(embedding)  
    X_trans = TSNE(n_components=2,random_state=seed).fit_transform(X_trans_PCA)
    return X_trans


adata = ad.read_h5ad('./data/T-SCP.h5ad')
Y_cell_type_label = list(adata.obs['cell_cycle'])
label_dict = {'G1':0,'G1-S':1,'G2':2,'G2-M':3}
target_names = ['G1','G1-S','G2','G2-M']

Y_label = np.array(itemgetter(*list(Y_cell_type_label))(label_dict))


# load learned cell embedding
X_fea = np.load('./data/embedding_T-SCP.npy')
print(X_fea.shape)

X_trans_learned = dimension_reduce(X_fea)

# plot
colors = [plt.cm.Set2(3), plt.cm.Set2(4), plt.cm.Set2(5), plt.cm.Set2(6)]

fig = plt.figure(figsize=(5,5))
for i in range(len(target_names)):
    plt.scatter(X_trans_learned[Y_label == i, 0]  
                , X_trans_learned[Y_label == i, 1] 
                , s = 10  
                , color=colors[i]  
                , label=target_names[i] 
                )
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
plt.xticks([])
plt.yticks([])
plt.title('scPROTEIN') 
plt.legend()
plt.savefig('T-SCP.jpg', bbox_inches='tight',dpi=300)   