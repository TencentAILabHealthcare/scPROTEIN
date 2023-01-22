# scPROTEIN


[![python >3.6.8](https://img.shields.io/badge/python-3.6.8-brightgreen)](https://www.python.org/) 


### A Versatile Deep Graph Contrastive Learning Framework for Single-cell Proteomics Embedding
scPROTEIN (**s**ingle-**c**ell **PROT**eomics **E**mbedd**IN**g) is a deep contrastive learning framework for Single-cell Proteomics Embedding.


The advance of single-cell proteomics sequencing technology sheds light on the research in revealing the protein-protein interactions, the post-translational modifications, and the proteoform dynamics of proteins in a cell. However, the uncertainty estimation for peptide quantification, data missingness, severe batch effects and high noise hinder the analysis of single-cell proteomic data. It is a significant challenge to solve this set of tangled problems together, where existing methods tailored for single-cell transcriptome do not address. Here, we proposed a novel versatile framework scPROTEIN, composed of peptide uncertainty estimation based on a multi-task heteroscedastic regression model and cell embedding learning based on graph contrastive learning designed for single-cell proteomic data analysis. scPROTEIN estimated the uncertainty of peptide quantification, denoised the protein data, removed batch effects and encoded single-cell proteomic-specific embeddings in a unified framework. We demonstrate that our method is efficient for cell clustering, batch correction, cell-type annotation and clinical analysis. Furthermore, our method can be easily plugged into single-cell resolved spatial proteomic data, laying the foundation for encoding spatial proteomic data for tumor microenvironment analysis.

For more information, please refer to [https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1](https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1)



![](https://github.com/TencentAILabHealthcare/scPROTEIN/blob/main/framework.jpg)




## Usage

Recomended usage procedure is as follows. 


1. Installation
(This usually takes 5 seconds on a normal desktop computer)
```
git clone https://github.com/TencentAILabHealthcare/scPROTEIN.git
cd scPROTEIN/
```


2. For datasets provided with raw peptide-level profile, scPROTEIN starts from stage 1 to learn the peptide uncertainty and obtain the protein-level expression profile in an uncertainty-guided manner. 

```
cd peptide_uncertainty_estimation/
python peptide_uncertainty_train.py
```

After stage 1, the learned estimated peptide uncertainty array "peptide_uncertainty.npy" will be saved in folder './scPROTEIN/peptide_uncertainty_estimation'


3. Run stage 2 to obtain the learned cell embeddings.

```
cd ..
python train.py --stage1 True
```

For datasets provided directly with the reconstructed protein-level profile, scPROTEIN will start from stage2.

```
python train.py --stage1 False
```

Afger stage 2, the learned cell embedding "scPROTEIN_embedding.npy" will be saved in folder './scPROTEIN/'


4. Evaluate the learned cell embeddings.
```
python visualization.py
```



## Expected output

After running the "visualization.py", a TSNE plot showing the cluster result will be saved in folder './scPROTEIN/' and an evaluation metric table will be displayed. Taking the demo SCoPE2_Specht dataset as a example, the expected TSNE plot output is:

and the expected metric table result is:

ARI  |ASW  |NMI  |PS 
-----|-----|-----|-----
0.387|0.625|0.389|0.811

## Hyperparameters


Hyperparameter       |Description                     | Default 
---------------------|--------------------------------| -------
stage1               |If scPROTEIN starts from stage1 | True
num_hidden           |Hidden dimension                | 256  
num_proj_hidden      |Dimension of projection head    | 256
num_layers           |Number of GCN layers            | 2
num_protos           |Number of prototypes            | 2
num_changed_edges    |Number of added/removed edges   | 50
drop_edge_rate_1     |Dropedge rate for view1         | 0.2
drop_edge_rate_2     |Dropedge rate for view2         | 0.4
drop_feature_rate_1  |Mask_feature rate for view1     | 0.4
drop_feature_rate_2  |Mask_feature rate for view1     | 0.2
alpha                |Balance factor                  | 0.05
tau                  |Temperature coefficient         | 0.4
threshold            |Threshold of graph construct    | 0.15


## Software Requirements

- Python >= 3.6.8
- torch >= 1.6.1
- scanpy >= 1.7.1
- pandas >= 1.3.5
- numpy >= 1.20.1
- torch_geometric >= 2.0.4
- scikit-learn >= 1.1.1
- scipy >= 1.8.1



## Disclaimer
This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.


## Coypright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.



## Citation
Li, W., Yang, F. et al. A Versatile Deep Graph Contrastive Learning Framework for Single-cell Proteomics Embedding. https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1


