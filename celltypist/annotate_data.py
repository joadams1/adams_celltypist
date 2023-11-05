#using models to annotate - from 'Benchmarking Models.ipynb'
import scanpy as sc
import pandas as pd
import anndata as ad
from anndata import AnnData
import numpy as np

from datetime import datetime
import itertools

from typing import Optional

from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.sparse import spmatrix

from annotate import annotate

def get_predictions(adata_loc, model_loc, write_loc):
    """
    Annotates a dataset using a CellTypist, transforms the result into AnnData, and then saves the results 
    
    adata_loc - where to find the training data 
    model_loc - where to find the model 
    write_loc - where to save the AnnData with predictions 
    """
    test_data = ad.read(adata_loc) 
    #print(model_loc)
    
    predictions = annotate(test_data, model = model_loc,  majority_voting = True)
    
    pred_adata = predictions.to_adata() #writes with conf_score for predicted_labels
    #pred_adata_mv = predictions.to_adata(insert_conf_by = 'majority_voting') #writes with conf_score for majority_voting
    
    #pred_adata.obs["conf_score_mv"] = pred_adata_mv.obs["conf_score"]
    
    pred_adata.write_h5ad(write_loc)
    return pred_adata


get_predictions('/data/peer/adamsj5/cell_typing/train_test_data/Cheong_Test.h5ad', '/home/adamsj5/auto_cell_typing/celltypist/Final Models/Cheong_full.pkl', '/data/peer/adamsj5/cell_typing/Data with Predictions/pred_Cheong_model.h5ad')
