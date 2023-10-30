from sklearn.metrics import silhouette_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA  
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import warnings



seed = 1

def purity_score(y_true, y_pred):
    contingency_matrix1 = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1) 

def batch_clus(tsne,label,batch,overlap_cell_type_label):
    ncell = len(label)
    Kb = len(np.unique(batch))
    unique_celltype = np.unique(label) 
    warnings.filterwarnings("ignore")
    df = pd.DataFrame()
    tsnesil,tsneari,tsnenmi,tsneps,tsneji = 0,0,0,0,0
    ncell_overlap = 0
    for i in list(overlap_cell_type_label):
        kmeans_tsne = KMeans(n_clusters=Kb,random_state=seed).fit(
                (tsne[label==i,:]))
        clusterlabel_tsne = kmeans_tsne.labels_
        tsnesil = tsnesil + silhouette_score(StandardScaler(
                ).fit_transform(tsne[label==i,:]),batch[label==i])*np.sum(label==i)
        tsnenmi = tsnenmi + normalized_mutual_info_score(batch[label==i],clusterlabel_tsne)*np.sum(label==i)
        tsneari = tsneari + adjusted_rand_score(batch[label==i],clusterlabel_tsne)*np.sum(label==i)
        tsneps = tsneps + purity_score(batch[label==i],clusterlabel_tsne)*np.sum(label==i)
        ncell_overlap += tsne[label==i,:].shape[0]
    df['1-ASW'] = [np.round(1-tsnesil/ncell_overlap,3)]
    df['1-ARI']=[np.round(1-tsneari/ncell_overlap,3)]
    df['1-NMI'] = [np.round(1-tsnenmi/ncell_overlap,3)]
    df['1-PS'] = [np.round(1-tsneps/ncell_overlap,3)] 
    return df


def celltype_clus(tsne,label):
    unique_celltype = np.unique(label)  
    K = len(unique_celltype)
    warnings.filterwarnings("ignore")
    df = pd.DataFrame()
    kmeans = KMeans(n_clusters = K,random_state=seed).fit(tsne)
    cluster_label = kmeans.predict(tsne)
    df['ASW'] = [np.round(silhouette_score(tsne,label),3)]
    df['ARI'] = [np.round(adjusted_rand_score(label,cluster_label),3)]
    df['NMI'] = [np.round(normalized_mutual_info_score(label,cluster_label),3)]
    df['PS'] = [np.round(purity_score(label,cluster_label),3)]
    return df


def dimension_reduce(embedding):
    X_trans_PCA = PCA(n_components=50, random_state=seed).fit_transform(embedding)  
    X_trans = TSNE(n_components=2, random_state=seed).fit_transform(X_trans_PCA)
    return X_trans


def integration_visualization(cell_type_with_dataname,embedding):
    warnings.filterwarnings("ignore")
    color_dic = {'C10(nanoPOTS)':'violet',
                 'RAW(nanoPOTS)':'dodgerblue',
                 'SVEC(nanoPOTS)':'coral',
                 'C10(N2)':'darkviolet',
                 'RAW(N2)':'cyan',
                 'SVEC(N2)':'orangered',
                 'Hela(SCoPE2_Leduc)':plt.cm.Set3(0),
                 'U-937(SCoPE2_Leduc)':plt.cm.tab20(4),
                 'U-937(plexDIA)':plt.cm.tab20(5),
                 'Melanoma(plexDIA)':plt.cm.tab20(10),
                 'HPAFII(plexDIA)':plt.cm.tab20b(14) ,
                 'BxPC3(pSCoPE_Huffman)':plt.cm.Set3(2) ,
                 'HPAFII(pSCoPE_Huffman)':plt.cm.tab20b(15),
                 'CFPACI(pSCoPE_Huffman)':plt.cm.tab20c(12), 
                 'U-937(pSCoPE_Leduc)':plt.cm.Set3(6),
                 'Melanoma(pSCoPE_Leduc)':plt.cm.tab20(11)
                 }
    marker_dic = {'C10':'o',
                 'RAW':'^',
                 'SVEC':'h',
                 'Hela':"s" ,
                 'U-937':"p",
                 'Melanoma':"D",
                 'HPAFII':"*",
                 'CFPACI':"P",
                 'BxPC3':"H"
                 }
    embedding = dimension_reduce(embedding)
    cell_type_with_dataname = np.array(cell_type_with_dataname)
    for cell_type in list(set(cell_type_with_dataname)):
        plt.scatter(
            embedding[cell_type_with_dataname==cell_type,0],
            embedding[cell_type_with_dataname==cell_type,1],
            label = cell_type,
            s = 10,
            color = color_dic[cell_type],
            marker = marker_dic[cell_type.split('(')[0]]
        )
    plt.legend()  
    plt.xlabel('tsne 1')
    plt.ylabel('tsne 2') 
    plt.xticks([])
    plt.yticks([])
    plt.show()


