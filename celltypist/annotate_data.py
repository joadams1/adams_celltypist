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
    
    predictions = annotate(test_data, model = model_loc, majority_voting = True)
    
    pred_adata = predictions.to_adata() #writes with conf_score for predicted_labels
    pred_adata_mv = predictions.to_adata(insert_conf_by = 'majority_voting') #writes with conf_score for majority_voting
    
    pred_adata.obs["conf_score_mv"] = pred_adata_mv.obs["conf_score"]
    
    pred_adata.write_h5ad(write_loc)
    return pred_adata


def get_all_predictions(adata_loc, models_loc, write_loc):
    print(adata_loc)
    
    #model 0
    model_loc_0 = models_loc+ "0.pkl"
    write_loc_0 = write_loc + "0.h5ad"
    pred_0 = get_predictions(adata_loc, model_loc_0, write_loc_0)
    pred_0.obs["model"] = 0
    print("model 0 done")
    
    #model 2
    model_loc_2 = models_loc+ "2.pkl"
    write_loc_2 = write_loc + "2.h5ad"
    pred_2 = get_predictions(adata_loc, model_loc_2, write_loc_2)
    pred_2.obs["model"] = 2
    print("model 2 done")
    
    #making sure the labels are different - had some super stranger issues with this when running this in notebooks 
    if pred_0.obs["predicted_labels"].equals(pred_2.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 0 predicted labels == model 2 predicted labels ")
    elif pred_0.obs["majority_voting"].equals(pred_2.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 0 maj voting == model 2 maj voting ")
    #model 3    
    model_loc_3 = models_loc+ "3.pkl"
    write_loc_3 = write_loc + "3.h5ad"
    pred_3 = get_predictions(adata_loc, model_loc_3, write_loc_3)
    pred_3.obs["model"] =3
    print("model 3 done")
    
    if pred_3.obs["predicted_labels"].equals(pred_2.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 3 predicted labels == model 2 predicted labels ")
    elif pred_3.obs["majority_voting"].equals(pred_2.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 3 maj voting == model 2 maj voting ")
    
    #model 4
    model_loc_4 = models_loc+ "4.pkl"
    write_loc_4 = write_loc + "4.h5ad"
    pred_4 = get_predictions(adata_loc, model_loc_4, write_loc_4)
    pred_4.obs["model"] = 4
    print("model 4 done")
    
    if pred_3.obs["predicted_labels"].equals(pred_4.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 3 predicted labels == model 4 predicted labels ")
    elif pred_3.obs["majority_voting"].equals(pred_4.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 3 maj voting == model 4 maj voting ")
    
    #model 5
    model_loc_5 = models_loc+ "5.pkl"
    write_loc_5 = write_loc + "5.h5ad"
    pred_5 = get_predictions(adata_loc, model_loc_5, write_loc_5)
    pred_5.obs["model"] = 5
    print("model 5 done")
    
    if pred_5.obs["predicted_labels"].equals(pred_4.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 5 predicted labels == model 4 predicted labels ")
    elif pred_5.obs["majority_voting"].equals(pred_4.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 5 maj voting == model 4 maj voting ")
    
    #model 6
    model_loc_6 = models_loc+ "6.pkl"
    write_loc_6 = write_loc + "6.h5ad"
    pred_6 = get_predictions(adata_loc, model_loc_6, write_loc_6)
    pred_6.obs["model"] = 6
    print("model 6 done")
    
    if pred_5.obs["predicted_labels"].equals(pred_6.obs["predicted_labels"]):
        raise ValueError(
                    f"🛑 Model 5 predicted labels == model 6 predicted labels ")
    elif pred_5.obs["majority_voting"].equals(pred_6.obs["majority_voting"]):
        raise ValueError(
                    f"🛑 Model 5 maj voting == model 6 maj voting ")
        
    """
    #if there is no X_umap in obsm    
    sc.pp.neighbors(pred_0)
    sc.tl.umap(pred_0)
    sc.pp.neighbors(pred_2)
    sc.tl.umap(pred_2)
    sc.pp.neighbors(pred_3)
    sc.tl.umap(pred_3)
    sc.pp.neighbors(pred_4)
    sc.tl.umap(pred_4)
    sc.pp.neighbors(pred_5)
    sc.tl.umap(pred_5)
    sc.pp.neighbors(pred_6)
    sc.tl.umap(pred_6)
    """
    
    #make one large h5ad file with all 6 models in one 
    pred_all = ad.concat([pred_0, pred_2, pred_3, pred_4, pred_5, pred_6]) 
    pred_all.write_h5ad(write_loc + "all.h5ad")

    
#get_all_predictions('/data/peer/adamsj5/cell_typing/train_test_data/CT_98_Test.h5ad', '/home/adamsj5/auto-cell-typing/celltypist/New Models/CT_98 Models/98_f_model_', '/data/peer/adamsj5/cell_typing/Data with Predictions/pred_98_model')

#get_all_predictions('/data/peer/adamsj5/cell_typing/train_test_data/COV_Test.h5ad', '/home/adamsj5/auto-cell-typing/celltypist/New Models/COV_PBMC Models/COV_f_model_', '/data/peer/adamsj5/cell_typing/Data with Predictions/pred_COV_model')
    
#get_all_predictions('/data/peer/adamsj5/cell_typing/train_test_data/train_glas.h5ad', '/home/adamsj5/auto-cell-typing/celltypist/New Models/Glasner Models/g_model_', '/data/peer/adamsj5/cell_typing/Data with Predictions/pred_g_model')

#get_all_predictions('/data/peer/adamsj5/cell_typing/train_test_data/HBCA_Test.h5ad', '/home/adamsj5/auto-cell-typing/celltypist/New Models/HBCA Models/HBCA_f_model_', '/data/peer/adamsj5/cell_typing/Data with Predictions/pred_HBCA_model')

#get_all_predictions('/data/peer/adamsj5/cell_typing/train_test_data/LuCA_Test.h5ad', '/home/adamsj5/auto-cell-typing/celltypist/New Models/LuCA Models/LuCA_f_model_', '/data/peer/adamsj5/cell_typing/Data with Predictions/pred_LuCA_model')

get_all_predictions('/data/peer/adamsj5/cell_typing/train_test_data/HuBMAP_Test.h5ad', '/home/adamsj5/auto-cell-typing/celltypist/New Models/HuBMAP Models/HuBMAP_f_model_', '/data/peer/adamsj5/cell_typing/Data with Predictions/pred_HuBMAP_model')