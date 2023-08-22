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
    test = ad.read(adata_loc) 
    print(model_loc)
    predictions = annotate(test, model = model_loc, majority_voting = True)
    pred_adata = predictions.to_adata()
    pred_adata.write_h5ad(write_loc)
    return pred_adata


def get_all_predictions(adata_loc, models_loc, write_loc):
    
    model_loc_0 = models_loc+ "0.pkl"
    write_loc_0 = write_loc + "0.h5ad"
    pred_0 = get_predictions(adata_loc, model_loc_0, write_loc_0)
    print("model 0 done")
    
    model_loc_2 = models_loc+ "2.pkl"
    write_loc_2 = write_loc + "2.h5ad"
    pred_2 = get_predictions(adata_loc, model_loc_2, write_loc_2)
    print("model 2 done")
    
    if pred_0.obs["predicted_labels"].equals(pred_2.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 0 predicted labels == model 2 predicted labels ")
    elif pred_0.obs["majority_voting"].equals(pred_2.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 0 maj voting == model 2 maj voting ")
        
    model_loc_3 = models_loc+ "3.pkl"
    write_loc_3 = write_loc + "3.h5ad"
    pred_3 = get_predictions(adata_loc, model_loc_3, write_loc_3)
    print("model 3 done")
    
    if pred_3.obs["predicted_labels"].equals(pred_2.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 3 predicted labels == model 2 predicted labels ")
    elif pred_3.obs["majority_voting"].equals(pred_2.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 3 maj voting == model 2 maj voting ")
    
    model_loc_4 = models_loc+ "4.pkl"
    write_loc_4 = write_loc + "4.h5ad"
    pred_4 = get_predictions(adata_loc, model_loc_4, write_loc_4)
    print("model 4 done")
    
    if pred_3.obs["predicted_labels"].equals(pred_4.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 3 predicted labels == model 4 predicted labels ")
    elif pred_3.obs["majority_voting"].equals(pred_4.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 3 maj voting == model 4 maj voting ")
        
    model_loc_5 = models_loc+ "5.pkl"
    write_loc_5 = write_loc + "5.h5ad"
    pred_5 = get_predictions(adata_loc, model_loc_5, write_loc_5)
    print("model 5 done")
    
    if pred_5.obs["predicted_labels"].equals(pred_4.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 5 predicted labels == model 4 predicted labels ")
    elif pred_5.obs["majority_voting"].equals(pred_4.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 5 maj voting == model 4 maj voting ")
  
    model_loc_6 = models_loc+ "6.pkl"
    write_loc_6 = write_loc + "6.h5ad"
    pred_6 = get_predictions(adata_loc, model_loc_6, write_loc_6)
    print("model 6 done")
    
    if pred_5.obs["predicted_labels"].equals(pred_6.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 5 predicted labels == model 6 predicted labels ")
    elif pred_5.obs["majority_voting"].equals(pred_6.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 5 maj voting == model 6 maj voting ")
    


    
get_all_predictions('/data/peer/adamsj5/cell_typing/train_test_data/CT_98_Test.h5ad', '/home/adamsj5/auto-cell-typing/celltypist/New Models/CT_98 Models/98_f_model_', '/data/peer/adamsj5/cell_typing/Data with Predictions/pred_98_model')