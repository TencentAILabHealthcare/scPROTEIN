# scPROTEIN

**scPROTEIN** is a deep contrastive learning framework for single-cell proteomics embedding.

## Overview

- For the datasets provided with raw peptide intensities, scPROTEIN *stage1* estimates the uncertainty of peptide quantification and aggregates the peptide content to the protein level in an uncertainty-guided manner.
- Taking the protein-level abundance matrix as input, scPROTEIN *stage2* aims to alleviate data missingness, denoise the protein data, remove batch effects in a unified framework, and encode single-cell proteomic-specific embeddings. These embeddings can be applied to a variety of downstream tasks.

## Key Functions

### For *stage 1*


<br/>




**load_peptide(data_path)**

**- Function:**

Load the input peptide-level file, and then extract the peptide sequences, peptide-level data along with other meta information.

**- Parameters:**
- `data_path` (str): Data path to load the peptide-level file.

**- Returns:**
- `peptides` (list): Peptide sequences.
- `proteins` (list): Protein names.
- `Y_label` (array): Peptide-level abundance matrix (peptide*cell).
- `cell_list` (list): The list containing the index of each cell.
- `num_cells` (int): Number of total cells.



<br/>
<br/>




**peptide_encode(peptides)**


**- Function:**

This function takes as input peptide sequences composed of amino acids. It returns the corresponding one-hot encoding data matrix and the total number of different amino acid types.

**- Parameters:**

- `peptides` (list): The input peptide sequences.

**- Returns:**

- `peptide_onehot_padding` (array): One-hot encoding matrix for peptide sequences.
- `num_amino_acid` (int): The number of different amino acid types.



<br/>
<br/>



**peptide_CNN(num_amino_acid, max_pool_size, hidden_dim, output_dim, conv_layers, dropout_rate, kernel_nums, kernel_size)**

**- Function:**

This function defines the Heteroscedastic regression model of scPROTEIN *stage1* for peptide uncertainty estimation.

**- Parameters:**

- `num_amino_acid` (int): The number of different amino acid types.
- `max_pool_size` (int): The size of the sliding window in the max-pooling operation.
- `hidden_dim` (int): The hidden dimension in the fully-connected layer.
- `output_dim` (int): Output dimension of the Heteroscedastic regression model, which is twice the number of cells (each cell has a $\mu$ and a $\sigma$).
- `conv_layers` (int): Number of convolutional layers.
- `dropout_rate` (float): Dropout rate.
- `kernel_nums` (int): Number of kernels in each convolutional block.
- `kernel_size` (int): Kernel size of each convolutional block.

**- Returns:**

- `model` (object): The defined Heteroscedastic regression model object.



<br/>
<br/>


**scPROTEIN_stage1_learning(model, peptide_onehot_padding, Y_label, learning_rate, weight_decay, split_percentage, num_epochs, batch_size)**

**- Function:**

This function constructs the framework for scPROTEIN *stage1* training and prediction.

**- Parameters:**

- `model` (object): Defined Heteroscedastic regression model object of *stage1*.
- `peptide_onehot_padding` (array): One-hot encoding matrix for the input peptide sequences.
- `Y_label` (array): Peptide-level abundance matrix (peptide*cell).
- `split_percentage` (float): Split percentage of data.
- `learning_rate` (float): Learning rate for the Adam optimizer.
- `weight_decay` (float): Weight decay for the Adam optimizer.
- `num_epochs` (int): Number of epochs for training *stage1*. We empirically set 90 to strike a balance between achieving convergence and reducing training time. 
- `batch_size` (int): Batch size for mini-batch training.

**- Returns:**

- `scPROTEIN_stage1` (object): The scPROTEIN *stage1* object. The functions of `scPROTEIN_stage1` are as follows:



    - `scPROTEIN_stage1.train()`: Perform scPROTEIN *stage1* training.
    - `scPROTEIN_stage1.uncertainty_generation()`: Generate the estimated peptide uncertainty based on the trained *stage1* model.


<br/>
<br/>


**load_sc_proteomic_features(stage1)**

**- Function:**

This function specifies whether to use *stage1* and loads the single-cell protein-level data matrix.

**- Parameters:**

- `stage1` (bool): This parameter indicates if scPROTEIN starts from *stage1*. `True` represents generating protein-level data using *stage1* in the uncertainty-guided manner, and `False` denotes directly learning from protein-level data.

**- Returns:**

- `proteins` (list): Protein names.
- `cells` (list): The list containing the index of each cell.
- `features` (array): Single-cell proteomics data matrix.



<br/>
<br/>



### For *stage 2*

**graph_generation(features, threshold, feature_preprocess)**

**- Function:**

This function constructs the cell graph based on the protein feature matrix.

**- Parameters:**

- `features` (array): Single-cell proteomics data matrix.
- `threshold` (float): Threshold for graph construction.
- `feature_preprocess` (bool): Feature preprocessing.

**- Returns:**

- `graph_data` (torch_geometric data object): The graph data in torch_geometric format, consisting of edges and node features.


<br/>
<br/>


**Encoder(input_features, num_hidden, activation, num_layers)**

**- Function:**

Construct the graph encoder for embedding learning.

**- Parameters:**

- `input_features` (int): Dimension of the input feature matrix (usually the number of proteins).
- `num_hidden` (int): Hidden dimension in the graph encoder.
- `activation` (str): The type of non-linear activation function.
- `num_layers` (int): Number of layers in the graph encoder.

**- Returns:**

- `encoder` (PyTorch module): The defined graph encoder module.


<br/>
<br/>

**Model(encoder, num_hidden, num_proj_hidden, tau)**

**- Function:**

This function establishes the scPROTEIN *stage2* model, consisting of a graph encoder, projection head, and loss calculation.

**- Parameters:**

- `encoder` (PyTorch module): Defined graph encoder.
- `num_hidden` (int): Hidden dimension in the graph encoder.
- `num_proj_hidden` (int): Hidden dimension of the projection head.
- `tau` (float): Temperature coefficient.

**- Returns:**

- `model` (PyTorch module): The defined scPROTEIN *stage2* model.



<br/>
<br/>


**scPROTEIN_learning(model, device, data, drop_feature_rate_1, drop_feature_rate_2, drop_edge_rate_1, drop_edge_rate_2, learning_rate, weight_decay, num_protos, topology_denoising, num_epochs, alpha, num_changed_edges, seed)**

**- Function:**

This function constructs the framework of scPROTEIN *stage2* training and prediction.

**- Parameters:**

- `model` (PyTorch module): Defined scPROTEIN *stage2* model.
- `device` (str): Running device.
- `data` (torch_geometric data): The defined graph data in torch_geometric format, consisting of edges and node features.
- `drop_feature_rate_1` (float): Dropedge rate for augmentation view1.
- `drop_feature_rate_2` (float): Dropedge rate for augmentation view2.
- `drop_edge_rate_1` (float): Feature masking rate for augmentation view1.
- `drop_edge_rate_2` (float): Feature masking rate for augmentation view2.
- `learning_rate` (float): Learning rate for Adam optimizer.
- `weight_decay` (float): Weight decay for Adam optimizer.
- `num_protos` (int): Number of prototypes.
- `topology_denoising` (bool): Indicator of if using the topology denoising.
- `num_epochs` (int): Number of epochs for training *stage2*. We empirically set 200 to strike a balance between achieving convergence and reducing training time.
- `alpha` (float): Balance factor in the loss function.
- `num_changed_edges` (int): Number of added/removed edges in topology denoising.
- `seed` (int): Random seed.

**- Returns:**

- `scPROTEIN` object for *stage2*. The functions of `scPROTEIN` are as follows:

    - `scPROTEIN.train()`: Conduct training of scPROTEIN *stage2*.
    - `scPROTEIN.embedding_generation()`: Generate the cell representation matrix based on the trained *stage2* model.



<br/>
<br/>


**integrate_sc_proteomic_features(dataset1, dataset2)**

**- Function:**

This function prepares for integrating different single-cell proteomics datasets.

**- Parameters:**

- `dataset1` (h5ad format): First dataset for integration.
- `dataset2` (h5ad format): Second dataset for integration.

**- Returns:**

- `batch_label` (array): The batch label which indicates the source of each cell.
- `cell_type_with_dataname` (list): Cell type of each cell, along with the dataset name.
- `cell_type_label` (array): Discrete cell type labels.
- `overlap_cell_type_label` (list): The overlap cell type(s) of both integrated datasets.
- `features_concat` (array): The combination of single-cell proteomics data from both integrated datasets, using the overlap proteins.


<br/>
<br/>


**integration_visualization(cell_type_with_dataname, embedding)**

**- Function:**

This function generates a 2D visualization plot of the data integration result. Users can customize the default cell type names and colors based on the used datasets.

**- Parameters:**

- `cell_type_with_dataname` (list): Cell type of each cell, along with the dataset name. This can be obtained using the `integrate_sc_proteomic_features` function.
- `embedding` (array): Learned cell representation matrix (cell * embedding).

**- Returns:**

- A 2D visualization plot of the integration result, colored by cell types.



<br/>
<br/>

**rank_proteins_and_volcano_plot(adata)**

**- Function:**

This function identifies the top 5 upregulated proteins and generates a volcano plot for clinical proteomics data analysis.

**- Parameters:**

- `adata` (object): The anndata object of the single-cell proteomics dataset. It should include protein rank information used for characterizing groups. This information can be generated in advance, e.g., using the `sc.tl.rank_genes_groups` function in Scanpy.

**- Returns:**

- The top 5 upregulated proteins.
- A volcano plot for differential protein analysis.


## Questions
If you have a question about using scPROTEIN, you can post an [issue](https://github.com/TencentAILabHealthcare/scPROTEIN/issues) or reach us by email(nkuweili@mail.nankai.edu.cn).