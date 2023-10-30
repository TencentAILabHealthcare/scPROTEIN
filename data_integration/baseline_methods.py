import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import scipy as sp
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA  
import magic
from harmony import harmonize
import scanorama
import sys
sys.path.append("../") 
from utils import *
from batch_visualization import *



baseline = 'KNN-ComBat'  

# load data
batch_label,cell_type_with_dataname,cell_type_label,overlap_cell_type_label, features = integrate_sc_proteomic_features('SCoPE2_Leduc','plexDIA')


## KNN-ComBat
if baseline == 'KNN-ComBat':

    imputer = KNNImputer(n_neighbors=5) 
    features = imputer.fit_transform(features)

    # construct anndata data file
    cell_nums, protein_nums = features.shape[0], features.shape[1]
    cell_name = list(range(cell_nums))
    protein_name = list(range(protein_nums))
    var = pd.DataFrame(index = protein_name)
    obs = pd.DataFrame(index = cell_name)
    adata = ad.AnnData(features,obs=obs,var=var)
    adata.obs['batch'] = list(batch_label)

    combat_data = adata.copy()
    sc.pp.combat(combat_data)
    features_combat = combat_data.X
    print(features_combat.shape)
    print(features_combat)

    np.save('{}_feature.npy'.format(baseline),features_combat)



## MAGIC
elif baseline == 'MAGIC':
    
    magic_operator = magic.MAGIC()
    features_magic = magic_operator.fit_transform(features)
    print(features_magic.shape)
    np.save('{}_feature.npy'.format(baseline),features_magic)



## Harmony
elif baseline == 'Harmony':
    batch_labels_pd = pd.DataFrame(batch_label,columns=['batch_labels'])
    # print(batch_labels_pd)
    pca = PCA(n_components = 50)
    features_pca = pca.fit_transform(features)
    features_harmony = harmonize(features_pca, batch_labels_pd, batch_key = 'batch_labels')
    print(features_harmony.shape)
    np.save('{}_feature.npy'.format(baseline),features_harmony)



## AutoClass
elif baseline == 'AutoClass':
    from AutoClass.AutoClass import AutoClassImpute, take_norm 
    # need to extract this script AutoClass.AutoClass from AutoClass 
    res = AutoClassImpute(X,cellwise_norm=False,log1p=False,num_cluster = [3,4,5])
    # num_cluster [2,3,4] for SCoPE2_Specht dataset, [3,4,5] for other experiments
    features_autoclass = res['imp'] 
    np.save('{}_feature.npy'.format(baseline),features_autoclass)



## Scanorama
elif baseline == 'Scanorama':
    adata_raw = ad.AnnData(features)
    adata_raw.obs['batch_label'] = pd.Categorical(batch_label)
    batches = adata_raw.obs['batch_label'].cat.categories.tolist()
    alldata = []
    for batch in batches:
        alldata.append(adata_raw[adata_raw.obs['batch_label'] == batch,])

    # Integration.
    scanorama.integrate_scanpy(alldata)

    # Batch correction.
    corrected = scanorama.correct_scanpy(alldata, dimred = 50, sigma = 15)
    scanorama_int = [ad.obsm['X_scanorama'] for ad in corrected]

    # make into one matrix.
    all_s = np.concatenate(scanorama_int)
    print(all_s.shape)
    print(all_s)
    np.save('{}_feature.npy'.format(baseline),all_s)


## Liger
## Liger is R package, and we provided its code here which users can run in R.
# files <- list.files(pattern = "batch[1-2].csv")
# total_data <- list()
# for(file in files){
#   tmp <-
#     Matrix(t(as.matrix(
#       read.csv(file, header = T, row.names = 1)
#     )),sparse = T)
#   colnames(tmp) <- paste0(str_remove(file, ".csv"), 1 : ncol(tmp))
#   total_data[[str_remove(file, ".csv")]] <- tmp
# }
# ifnb_liger <- createLiger(total_data)

# ifnb_liger <- normalize(ifnb_liger)
# ifnb_liger <- selectGenes(ifnb_liger)
# ifnb_liger <- scaleNotCenter(ifnb_liger)
# ifnb_liger <- optimizeALS(ifnb_liger, k = 20)
# ifnb_liger <- quantile_norm(ifnb_liger)


