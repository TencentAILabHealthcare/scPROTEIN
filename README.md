# scPROTEIN


[![python >3.6.8](https://img.shields.io/badge/python-3.6.8-brightgreen)](https://www.python.org/) 


### A Versatile Deep Graph Contrastive Learning Framework for Single-cell Proteomics Embedding
scPROTEIN (**s**ingle-**c**ell **PROT**eomics **E**mbedd**IN**g) is a deep contrastive learning framework for Single-cell Proteomics Embedding.


The advance of single-cell proteomics sequencing technology sheds light on the research in revealing the protein-protein interactions, the post-translational modifications, and the proteoform dynamics of proteins in a cell. However, the uncertainty estimation for peptide quantification, data missingness, severe batch effects and high noise hinder the analysis of single-cell proteomic data. It is a significant challenge to solve this set of tangled problems together, where existing methods tailored for single-cell transcriptome do not address. Here, we proposed a novel versatile framework scPROTEIN, composed of peptide uncertainty estimation based on a multi-task heteroscedastic regression model and cell embedding learning based on graph contrastive learning designed for single-cell proteomic data analysis. scPROTEIN estimated the uncertainty of peptide quantification, denoised the protein data, removed batch effects and encoded single-cell proteomic-specific embeddings in a unified framework. We demonstrate that our method is efficient for cell clustering, batch correction, cell-type annotation and clinical analysis. Furthermore, our method can be easily plugged into single-cell resolved spatial proteomic data, laying the foundation for encoding spatial proteomic data for tumor microenvironment analysis.

For more information, please refer to [https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1](https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1)



![](https://github.com/TencentAILabHealthcare/scPROTEIN/blob/main/framework.jpg)



## Input single-cell proteomic data format

A csv file in the following format is needed for scPROTEIN learning from stage 1:

Protein  |Peptide                |Cell 0          |Cell 1          |Cell 2          |Cell 3          |Cell 4          |Cell 5          |Cell 6
---------|-----------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------
P08865	 |LLVVTDPR_2	           |0.215943903	    |1.825849332	   |0.17106779	    |0.090752671	   |0.633329732	    |-0.044091136    |NA
P26447	 |RTDEAAFQK_3	           |1.873431237	    |1.425136257	   |2.354956659    	|1.373487482	   |1.724188343	    |0.828024968	   |0.511722654	
P26447	 |LNKSELK_3	             |NA	            |NA	             |NA	            |NA	             |NA	            |-0.164518259	   |-0.765802428
Q00610	 |LLYNNVSNFGR_2	         |-0.452033525	  |NA	             |NA	            |-0.211513228	   |-0.573607252	  |-0.593867542    |NA
P05120	 |LNGLYPFR_2	           |NA	            |NA	             |0.245379509	    |0.923845132	   |0.300612918	    |NA	             |NA



"Protein" represents the protein name and "Peptide" denotes the corresponding constituting peptide sequence(s). The columns "Cell 0","Cell 1"... are the protein data in each cell. NA is the missing value. If datasets are provided directly from protein-level (without "Peptide" column), scPROTEIN can start from stage 2.


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

![](https://github.com/TencentAILabHealthcare/scPROTEIN/blob/main/TSNE_result.jpg)


and the expected metric table result is like:

ARI  |ASW  |NMI  |PS 
-----|-----|-----|-----
0.387|0.625|0.389|0.811

## Hyperparameters

Hyperparameters for stage 1:

Hyperparameter       |Description                     | Default 
---------------------|--------------------------------| -------
batch_size           |Batch_size                      |256  
kernel_nums          |Kernel num of each conv block   |[300,200,100]
kernel_size          |Kernel size of each conv block  |[2,2,2]
max_pool_size        |Max pooling size                |1
conv_layers          |Nums of conv layers             |3
hidden_dim           |Hidden dim for fc layer         |3000


Hyperparameters for stage 2:

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


## Time cost

Taking demo SCoPE2_Specht data (1490 cells, 3042 proteins) as an example, typical running time on a "normal" desktop computer is about 40 minutes for stage 1 and about 10 minutes for stage 2.


## Disclaimer
This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.


## Coypright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.



## Citation
Li, W., Yang, F. et al. A Versatile Deep Graph Contrastive Learning Framework for Single-cell Proteomics Embedding. https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1


