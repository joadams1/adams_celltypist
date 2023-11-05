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

import train
    
def make_final_model(adata_loc, annot_col, abrev, write_loc):
    train_adata = ad.read(adata_loc)  
    indata = train_adata.X
    labels = train_adata.obs[annot_col]
    genes = train_adata.var_names
    
    cp_and_ct_genes = [x for x in cp_genes if x in train_adata.var_names]
    cp_and_ct_genes = np.unique(cp_and_ct_genes)

    sc.pp.highly_variable_genes(train_adata, n_top_genes=10000)

    all_genes = train_adata.var['highly_variable']
    highly_var = []

    for i in all_genes.index: 
        if all_genes[i] == True: 
            highly_var.append(i)
    for i in cp_and_ct_genes:
        if i not in highly_var:
            highly_var.append(i)
    train_hv = train_adata[:, highly_var]
    indata_hv = train_hv.X
    labels_hv = train_hv.obs[annot_col]
    genes_hv = train_hv.var_names

    model, genes = train.train_modified(X = indata_hv, labels = labels_hv, genes = genes_hv, check_expression = False, use_SGD = True, mini_batch = True, balance_cell_type = True, feature_selection = True)
    
    write_loc_full = write_loc+abrev+'_full'

    model.write(write_loc_full)
    write_loc_g = write_loc_full+'_genes.csv'
    print(write_loc_g)
    genes.to_csv(write_loc_g)  


    
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
    

adata_loc = '/data/peer/adamsj5/cell_typing/glasner_fine_annot.h5ad'
annot_col = 'finer_cell_types'
abrev = 'g'
write_loc = '/home/adamsj5/auto_cell_typing/celltypist/Final Models/'
make_final_model(adata_loc, annot_col , abrev ,  write_loc )

