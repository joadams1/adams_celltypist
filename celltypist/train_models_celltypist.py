#making cell typing models using CellTypist & variations - from Making New Models.ipynb

import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import itertools
from anndata import AnnData
from scipy.sparse import spmatrix
from datetime import datetime
from typing import Optional, Union
from sklearn import __version__ as skv

import cytopus as cp

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

#import celltypist as ct #if its throwing an error with sklearn, install scikit-learn version 1.1.0 & that should fix
#from celltypist import logger 
#from celltypist.models import Model

import train

#fxn to make individual models 
def make_model(model_ver: int = 0, 
              X = None,
              labels: Optional[Union[str, list, tuple, np.ndarray, pd.Series, pd.Index]] = None,
              genes: Optional[Union[str, list, tuple, np.ndarray, pd.Series, pd.Index]] = None,
              check_expression: bool = False,
              cyto_genes: Optional[np.ndarray] = None,
              write_loc: str = 'New Models'): 
    """
    mode_ver 
        Which type of model to make 
        (Default: 0)
    X
        Path to the input count matrix (supported types are csv, txt, tsv, tab and mtx) or AnnData (h5ad).
        Also accepts the input as an :class:`~anndata.AnnData` object, or any array-like objects already loaded in memory.
        See `check_expression` for detailed format requirements.
        A cell-by-gene format is desirable (see `transpose_input` for more information).
    labels
        Path to the file containing cell type label per line corresponding to the cells in `X`.
        Also accepts any list-like objects already loaded in memory (such as an array).
        If `X` is specified as an AnnData, this argument can also be set as a column name from cell metadata.
    genes
        Path to the file containing one gene per line corresponding to the genes in `X`.
        Also accepts any list-like objects already loaded in memory (such as an array).
        Note `genes` will be extracted from `X` where possible (e.g., `X` is an AnnData or data frame).
    check_expression
        Check whether the expression matrix in the input data is supplied as required by celltypist.
        `X` should be in log1p normalized expression to 10000 counts per cell.
        (Default: `False`)
    cyto_genes
        (For model 4) A list of genes to make sure are included in feature selection
    write_loc
        Where to save the newly made model 
        (Default: 'New Models' - directory in GitHub)
    """
    if model_ver == 0 or model_ver == 6:
        #vanilla celltypist
        model, genes = train.train_modified(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True, balance_cell_type = True, feature_selection = True)
    
    if model_ver == 1:
        #no fs
        model, genes = train.train_modified(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True)
    
    if model_ver == 2:
        #L1 reg
        model, genes = train.train_modified(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True, feature_selection = True, penalty = "l1")
    
    if model_ver == 3: 
        #cytopus genes only
        model, genes = train.train_modified(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True)
    
    if model_ver == 4: 
        #fs with cytopus genes
        model, genes = train.train_modified(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True, feature_selection = True, balance_cell_type = True, use_cytopus = True, cyto_genes = cyto_genes)
    
    if model_ver == 5: 
        #merge of 2 & 4 
        model, genes = train.train_modified(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True, feature_selection = True, penalty = "l1", switch_penalty = True, balance_cell_type = True, use_cytopus = True, cyto_genes = cyto_genes)
    #model.write(write_loc)
    write_loc_g = write_loc+'_genes.csv'
    print(write_loc_g)
    genes.to_csv(write_loc_g)  
    
    return model, genes 

def make_all_models(adata_loc, annot_col, abrev, check_expression: bool = False, data_dir: str = '/data/peer/adamsj5/cell_typing/train_test_data/', write_loc: str = '/home/adamsj5/auto-cell-typing/celltypist/New Models/Glasner Models/'):
    """
    This function makes all of the models defined for this cell typing benchmark for one particular dataset. 
    adata - AnnData dataset of interest. 
        Make sure the data has gone through desired preprocessing and is in the format required (only gene counts (no protein), normalized to median library size (prefered) or 10,000 counts per cell, log transformed with pseudeocount of 1, etc)
    annot_col - name of the column in adata that holds the groundtruth cell type annotationg
    abrev - an abreviation for this dataset to identify later 
    precent_train - what percent of the dataset you want to section of for training (the rest will be set aside as test data)
        (Default: 0.7)
    check_expression - whether or not to use CellTypist's check expression function 
        It will throw an error if the data is not normalized to 10,000 counts per cell and log transformed with a pseudeo count of 1 
        (Default: False)
    data_dir - the directory to save the split train & test data 
        (Default: '/data/peer/adamsj5/cell_typing/train_test_data/' - Jo Adams' directory on lilac/calcifer)
    write_loc - where to save the newly made model 
        (Default: 'New Models/')
    train_data - Location of AnnData of interest that has already been split into train & test data. This is optional; it is to allow for the same dataset to be used for training if one already exists.
    """
    print(abrev)
    
    train = ad.read(adata_loc)  
    indata = train.X
    labels = train.obs[annot_col]
    genes = train.var_names
    
    cp_and_ct_genes = [x for x in cp_genes if x in train.var_names]
    cp_and_ct_genes = np.unique(cp_and_ct_genes)
    
    train_cp = train[:, cp_and_ct_genes]
    indata_cp = train_cp.X
    labels_cp = train_cp.obs[annot_col]
    genes_cp = train_cp.var_names
    
   
    #acutually make all the models
    write_loc_0 = write_loc+abrev+'_model_0'
    print(write_loc_0)
    model_0, genes_0 = make_model(X = indata, labels = labels, genes = genes, check_expression = check_expression, write_loc = write_loc_0)
    print("Model 0 Done")
    
    write_loc_2 = write_loc+abrev+'_model_2'
    print(write_loc_2)
    model_2, genes_2 = make_model(model_ver = 2, X = indata, labels = labels, genes = genes, check_expression = check_expression, write_loc = write_loc_2)
    print("Model 2 Done")
    '''
    write_loc_3 = write_loc+abrev+'_model_3'
    print(write_loc_3)
    model_3, genes_3 = make_model(model_ver = 3, X = indata_cp, labels = labels_cp, genes = genes_cp, check_expression = check_expression, write_loc = write_loc_3)
    print("Model 3 Done")
    
    write_loc_4 = write_loc+abrev+'_model_4'
    print(write_loc_4)
    model_4, genes_4 = make_model(model_ver = 4, X = indata, labels = labels, genes = genes, check_expression = check_expression, cyto_genes = cp_and_ct_genes, write_loc = write_loc_4)
    print("Model 4 Done")
    '''
    write_loc_5 = write_loc+abrev+'_model_5'
    print(write_loc_5)
    model_5, genes_5 = make_model(model_ver = 5, X = indata, labels = labels, genes = genes, check_expression = check_expression, cyto_genes = cp_and_ct_genes, write_loc = write_loc_5)
    print("Model 5 Done")
    
    
    #filter for 10000 most variable 
    sc.pp.highly_variable_genes(train, n_top_genes=10000)

    all_genes = train.var['highly_variable']
    highly_var = []

    for i in all_genes.index: 
        if all_genes[i] == True: 
            highly_var.append(i)
    for i in cp_and_ct_genes:
        if i not in highly_var:
            highly_var.append(i)
    train_hv = train[:, highly_var]
    indata_hv = train_hv.X
    labels_hv = train_hv.obs[annot_col]
    genes_hv = train_hv.var_names
    
    write_loc_6 = write_loc+abrev+'_model_6'
    model_6, genes_6 = make_model(model_ver = 6, X = indata_hv, labels = labels_hv, genes = genes_hv, check_expression = check_expression, write_loc = write_loc_6)
    print("Model 6 Done")
    
    '''
    #filter for 4000 most variable 
    
    sc.pp.highly_variable_genes(train, n_top_genes=4000)

    all_genes = train.var['highly_variable']
    highly_var = []

    for i in all_genes.index: 
        if all_genes[i] == True: 
            highly_var.append(i)
    for i in cp_and_ct_genes:
        if i not in highly_var:
            highly_var.append(i)
    train_hv = train[:, highly_var]
    indata_hv = train_hv.X
    labels_hv = train_hv.obs[annot_col]
    genes_hv = train_hv.var_names
    
    write_loc_6_4000 = write_loc+abrev+'_model_6_4000'
    model_6_4000, genes_6_4000 = make_model(model_ver = 6, X = indata_hv, labels = labels_hv, genes = genes_hv, check_expression = check_expression, write_loc = write_loc_6_4000)
    print("Model 6_4000 Done")
    '''
    print("All Models Done")
    
    
## DATA
G = cp.kb.KnowledgeBase()
cell_dict = G.identities

#make a list of genes from cytopus dict & remove NaNs
#the celltype information doesn't need to be retained since we're applying this gene list to all celltypes
cp_genes = []
for i in cell_dict.values():
    cp_genes.append(i)
cp_genes = list(itertools.chain(*cp_genes)) #make flatlist out of LoL
cp_genes = [x for x in cp_genes if str(x) != 'nan']


#THINGS TO CHANGE:
"""
adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/LuCA_Train.h5ad'
#sc.pp.filter_genes(adata_read, min_cells = 1)
annot_col = 'cell_type'
abrev = 'LuCA_f'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/LuCA Models/' #change above in fxn def also for it to work properly
"""
"""
adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/HBCA_Train.h5ad'
#sc.pp.filter_genes(adata_read, min_cells = 1)
annot_col = 'cell_type'
abrev = 'HBCA_f'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/HBCA Models/'
"""

adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/train_glas.h5ad'
#sc.pp.filter_genes(adata_read, min_cells = 1)
annot_col = 'finer_cell_types'
abrev = 'g_alt'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/Glasner Models/'

"""
adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/COV_Train.h5ad'
#sc.pp.filter_genes(adata_read, min_cells = 1)
annot_col = 'full_clustering'
abrev = 'COV_f'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/COV_PBMC Models/'
"""
"""
adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/CT_98_Train.h5ad'
#sc.pp.filter_genes(adata_read, min_cells = 1)
annot_col = 'Harmonised_detailed_type'
abrev = '98_f'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/CT_98 Models/'
"""
"""
adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/CT_45_Train.h5ad'
#sc.pp.filter_genes(adata_read, min_cells = 1)
annot_col = 'Manually_curated_celltype'
abrev = 'ct_f'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/CT_45 Models/'
"""
"""
adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/HuBMAP_Train.h5ad'
annot_col = 'cell_type'
abrev = 'HuBMAP_f'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/HuBMAP Models/'
"""
"""
adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/Niec_SI_Train.h5ad'
annot_col = 'cell_state'
abrev = 'Niec_SI'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/Niec Models/'

make_all_models(adata_loc, annot_col , abrev ,  write_loc )

adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/Niec_LI_Train.h5ad'
annot_col = 'cell_state'
abrev = 'Niec_LI'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/Niec Models/'
"""
make_all_models(adata_loc, annot_col , abrev ,  write_loc )