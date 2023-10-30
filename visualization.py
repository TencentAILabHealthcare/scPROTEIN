from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  
from utils import *
from operator import itemgetter
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn import metrics
from sklearn.metrics import silhouette_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

import warnings


warnings.filterwarnings("ignore")
seed = 1

def purity_score(y_true, y_pred):
    contingency_matrix1 = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1) 


def dimension_reduce(embedding):
    X_trans_PCA = PCA(n_components=50, random_state=seed).fit_transform(embedding)  
    X_trans = TSNE(n_components=2,random_state=seed).fit_transform(X_trans_PCA)
    return X_trans



# load ground truth cell label
Y_cell_type_label = load_cell_type_labels()
label_dict = {'sc_m0':0, 'sc_u':1}
target_names = ['Macrophage','Monocyte']
Y_label = np.array(itemgetter(*list(Y_cell_type_label))(label_dict))



# load learned cell embedding
X_fea = np.load('scPROTEIN_embedding.npy')
print(X_fea.shape)


k_means = KMeans(n_clusters=len(target_names))
y_predict = k_means.fit_predict(X_fea)
df_result = pd.DataFrame()
df_result['ARI'] = [np.round(adjusted_rand_score(Y_label,y_predict),3)]
df_result['ASW'] = [np.round(silhouette_score(X_fea,y_predict),3)]
df_result['NMI'] = [np.round(normalized_mutual_info_score(Y_label,y_predict),3)]
df_result['PS'] = [np.round(purity_score(Y_label,y_predict),3)]
print(df_result)


X_trans_learned = dimension_reduce(X_fea)



# plot
colors = [plt.cm.Set2(2), plt.cm.Set2(1)]
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
plt.savefig('TSNE_result.jpg', bbox_inches='tight',dpi=300)   

