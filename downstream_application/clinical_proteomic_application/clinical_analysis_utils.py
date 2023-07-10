import matplotlib.pyplot as plt
import seaborn as sns  
from bioinfokit import analys, visuz
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import warnings


# output 5 upregulated proteins and plot the volcano plot
def rank_proteins_and_volcano_plot(adata):
        
    names = list(adata.uns['rank_genes_groups']['names'])
    logfoldchanges = list(adata.uns['rank_genes_groups']['logfoldchanges'])
    pvals_adj = list(adata.uns['rank_genes_groups']['pvals_adj'])

    names_list = []
    logfoldchanges_list = []
    pvals_adj_list = []

    for i in range(len(names)):
        names_list.append(names[i][1])
        logfoldchanges_list.append(logfoldchanges[i][1])
        pvals_adj_list.append(pvals_adj[i][1])

    print('top 5 upregulated proteins:',names_list[:5])

    dic = {'GeneNames':names_list,
        'log2FC' : logfoldchanges_list,
        'p-value' : pvals_adj_list}
    data = pd.DataFrame(dic) 
    visuz.GeneExpression.volcano(df=data, lfc="log2FC", pv="p-value", geneid="GeneNames", lfc_thr = (np.log2(1.5), np.log2(1.5)),
        gstyle=2, sign_line=True,xlm=(-2,2,0.5),genenames = (names_list[0]),color = ("#fd625e", "grey", "#01b8aa"))
    