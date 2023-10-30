# scPROTEIN


[![python >3.8.12](https://img.shields.io/badge/python-3.8.12-brightgreen)](https://www.python.org/) 



### A Versatile Deep Graph Contrastive Learning Framework for Single-cell Proteomics Embedding
scPROTEIN (**s**ingle-**c**ell **PROT**eomics **E**mbedd**IN**g) is a deep contrastive learning framework for Single-cell Proteomics Embedding.


The advance of single-cell proteomics sequencing technology sheds light on the research in revealing the protein-protein interactions, the post-translational modifications, and the proteoform dynamics of proteins in a cell. However, the uncertainty estimation for peptide quantification, data missingness, severe batch effects and high noise hinder the analysis of single-cell proteomic data. It is important to solve this set of tangled problems together, which existing methods tailored for single-cell transcriptome cannot fully address. Here, we proposed a novel versatile framework scPROTEIN, composed of peptide uncertainty estimation based on a multi-task heteroscedastic regression model and cell embedding learning based on graph contrastive learning designed for single-cell proteomic data analysis. scPROTEIN estimated the uncertainty of peptide quantification, denoised the protein data, removed batch effects and encoded single-cell proteomic-specific embeddings in a unified framework. We demonstrate that our method is efficient for cell clustering, batch correction, cell-type annotation and clinical analysis. Furthermore, our method can also be plugged into single-cell resolved spatial proteomic data, laying the foundation for encoding spatial proteomic data for tumor microenvironment analysis.

For more information, please refer to [https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1](https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1)



<p align="center">
  <img width="80%" src=./image/framework.jpg>
</p>

## Dependences

[![torch-1.10.0](https://img.shields.io/badge/torch-1.10.0-red)](https://github.com/pytorch/pytorch) 
[![scanpy-1.8.2](https://img.shields.io/badge/scanpy-1.8.2-blue)](https://github.com/theislab/scanpy) 
[![scikit__learn-1.1.1](https://img.shields.io/badge/scikit__learn-1.1.1-green)](https://github.com/scikit-learn/scikit-learn)
[![numpy-1.22.3](https://img.shields.io/badge/numpy-1.22.3-orange)](https://github.com/numpy/numpy) 
[![pandas-1.3.5](https://img.shields.io/badge/pandas-1.3.5-lightgrey)](https://github.com/pandas-dev/pandas) 
[![scipy-1.8.1](https://img.shields.io/badge/scipy-1.8.1-yellowgreen)](https://github.com/scipy/scipy) 
[![torch__geometric-2.0.4](https://img.shields.io/badge/torch__geometric-2.0.4-blueviolet)](https://github.com/pyg-team/pytorch_geometric/)



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



## Documentation
The [documentation](./docs/documentaion.md) which elucidates the functions of scPROTEIN is provided.



## Usage

Recomended usage procedure is as follows. 


1.Installation

The running environment of scPROTEIN can be installed from docker-hub repository: 

- Pull the docker image from docker-hub

```
docker pull nkuweili/scprotein:latest
```

- Run the docker image (GPU is needed)


```
docker run --name scprotein --gpus all -it --rm nkuweili/scprotein:latest /bin/bash
```


- Download this repository
(This usually takes 10 seconds on a normal desktop computer)
```
git clone https://github.com/TencentAILabHealthcare/scPROTEIN.git
cd scPROTEIN/
```


2.For datasets provided with raw peptide-level profile, scPROTEIN starts from stage 1 to learn the peptide uncertainty and obtain the protein-level abundance in an uncertainty-guided manner. 

```
cd peptide_uncertainty_estimation/
python3 peptide_uncertainty_train.py
```

After stage 1, the learned estimated peptide uncertainty array will be saved in folder './scPROTEIN/peptide_uncertainty_estimation'


3.Run stage 2 to obtain the learned cell embeddings.

```
cd ..
python3 train.py --stage1 True
```

For datasets provided directly with the reconstructed protein-level profile, scPROTEIN will start from stage2.

```
python3 train.py
```

Afger stage 2, the learned cell embedding will be saved in folder './scPROTEIN/'


4.Evaluate the learned cell embeddings.
```
python3 visualization.py
```



## Expected output

After running the "visualization.py", a TSNE plot showing the cluster result will be saved in folder './scPROTEIN/', and a corresponding evaluation metric table will be displayed.


## Use trained scPROTEIN model for evaluation

For loading [checkpoints](./trained_scPROTEIN/) for scPROTEIN stage1 and stage2 on SCoPE2_Specht dataset for generating uncertainty and cell embedding, respectively:

```
cd peptide_uncertainty_estimation/
python3 peptide_uncertainty_train.py --use_trained_scPROTEIN True
cd ..
python3 train.py --stage1 True --use_trained_scPROTEIN True
python3 visualization.py
```


## Tutorial

The following notebooks are provided to show how to run scPROTEIN model

1. [tutorial_scPROTEIN_stage1](tutorial_scPROTEIN_stage1.ipynb) gives a detailed description for uncertainty estimation for scPROTEIN stage1.
2. [tutorial_scPROTEIN_stage2](tutorial_scPROTEIN_stage2.ipynb) provides an example using protein-level data from stage1 to learn cell embedding in stage2.
3. [data_integration](./data_integration/) shows the running process for data integration and batch correction across various MS acquisitions.
4. [downstream_application](./downstream_application/) displays the analysis for clinical proteomic data, spatial proteomic data and cell cycle.


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
num_hidden           |Hidden dimension                | 400  
num_proj_hidden      |Dimension of projection head    | 256
num_layers           |Number of GCN layers            | 2
num_protos           |Number of prototypes            | 2
num_changed_edges    |Number of added/removed edges   | 10
drop_edge_rate_1     |Dropedge rate for view1         | 0.2
drop_edge_rate_2     |Dropedge rate for view2         | 0.4
drop_feature_rate_1  |Mask_feature rate for view1     | 0.4
drop_feature_rate_2  |Mask_feature rate for view1     | 0.2
alpha                |Balance factor                  | 0.05
tau                  |Temperature coefficient         | 0.4


## Time cost

Taking demo SCoPE2_Specht dataset (1490 cells, 3042 proteins) as an example, typical running time on a "normal" desktop computer is about 40 minutes for stage 1 and about 10 minutes for stage 2.



## Disclaimer
This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.


## Questions
If you have a question about using scPROTEIN, you can post an [issue](https://github.com/TencentAILabHealthcare/scPROTEIN/issues) or reach us by email(nkuweili@mail.nankai.edu.cn, fionafyang@tencent.com).



## Citation
Li, W., Yang, F., et al. A Versatile Deep Graph Contrastive Learning Framework for Single-cell Proteomics Embedding. https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1


