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

import celltypist as ct #if its throwing an error with sklearn, install scikit-learn version 1.1.0 & that should fix
from celltypist import logger 
from celltypist.models import Model

def _to_vector(_vector_or_file):
    """
    For internal use. Turn a file into an array.
    """
    if isinstance(_vector_or_file, str):
        try:
            return pd.read_csv(_vector_or_file, header=None)[0].values
        except Exception as e:
            raise Exception(
                    f"🛑 {e}")
    else:
        return _vector_or_file

def _to_array(_array_like) -> np.ndarray:
    """
    For internal use. Turn an array-like object into an array.
    """
    if isinstance(_array_like, pd.DataFrame):
        return _array_like.values
    elif isinstance(_array_like, spmatrix):
        return _array_like.toarray()
    elif isinstance(_array_like, np.matrix):
        return np.array(_array_like)
    elif isinstance(_array_like, np.ndarray):
        return _array_like
    else:
        raise TypeError(
                f"🛑 Please provide a valid array-like object as input")

def _prepare_data(X, labels, genes, transpose) -> tuple:
    """
    For internal use. Prepare data for celltypist training.
    """
    if (X is None) or (labels is None):
        raise Exception(
                "🛑 Missing training data and/or training labels. Please provide both arguments")
    if isinstance(X, AnnData) or (isinstance(X, str) and X.endswith('.h5ad')):
        adata = sc.read(X) if isinstance(X, str) else X
        adata.var_names_make_unique()
        if adata.X.min() < 0:
            logger.info("👀 Detected scaled expression in the input data, will try the .raw attribute")
            try:
                indata = adata.raw.X
                genes = adata.raw.var_names
            except Exception as e:
                raise Exception(
                        f"🛑 Fail to use the .raw attribute in the input object. {e}")
        else:
            indata = adata.X
            genes = adata.var_names
        if isinstance(labels, str) and (labels in adata.obs):
            labels = adata.obs[labels]
        else:
            labels = _to_vector(labels)
    elif isinstance(X, str) and X.endswith(('.csv', '.txt', '.tsv', '.tab', '.mtx', '.mtx.gz')):
        adata = sc.read(X)
        if transpose:
            adata = adata.transpose()
        if X.endswith(('.mtx', '.mtx.gz')):
            if genes is None:
                raise Exception(
                        "🛑 Missing `genes`. Please provide this argument together with the input mtx file")
            genes = _to_vector(genes)
            if len(genes) != adata.n_vars:
                raise ValueError(
                        f"🛑 The number of genes provided does not match the number of genes in {X}")
            adata.var_names = np.array(genes)
        adata.var_names_make_unique()
        if not float(adata.X.max()).is_integer():
            logger.warn(f"⚠️ Warning: the input file seems not a raw count matrix. The trained model may be biased")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        indata = adata.X
        genes = adata.var_names
        labels = _to_vector(labels)
    elif isinstance(X, str):
        raise ValueError(
                "🛑 Invalid input. Supported types: .csv, .txt, .tsv, .tab, .mtx, .mtx.gz and .h5ad")
    else:
        logger.info("👀 The input training data is processed as an array-like object")
        indata = X
        if transpose:
            indata = indata.transpose()
        if isinstance(indata, pd.DataFrame):
            genes = indata.columns
        else:
            if genes is None:
                raise Exception(
                        "🛑 Missing `genes`. Please provide this argument together with the input training data")
            genes = _to_vector(genes)
        labels = _to_vector(labels)
    return indata, labels, genes

def _SGDClassifier(indata, labels,
                   alpha, max_iter, n_jobs,
                   mini_batch, batch_number, batch_size, epochs, balance_cell_type, penalty , **kwargs) -> SGDClassifier:
    """
    For internal use 
    
    ONE NEW ARG
    penalty
        allows to user specify what type of regularization
    """
    loss_mode = 'log_loss' if float(skv[:3]) >= 1.1 else 'log'
    classifier = SGDClassifier(loss = loss_mode, penalty = penalty, alpha = alpha, max_iter = max_iter, n_jobs = n_jobs, **kwargs)
    if not mini_batch:
        logger.info(f"🏋️ Training data using SGD logistic regression")
        if (len(labels) > 100000) and (indata.shape[1] > 10000):
            logger.warn(f"⚠️ Warning: it may take a long time to train this dataset with {len(labels)} cells and {indata.shape[1]} genes, try to downsample cells and/or restrict genes to a subset (e.g., hvgs)")
        classifier.fit(indata, labels)
    else:
        logger.info(f"🏋️ Training data using mini-batch SGD logistic regression")
        no_cells = len(labels)
        if no_cells < 10000:
            logger.warn(f"⚠️ Warning: the number of cells ({no_cells}) is not big enough to conduct a proper mini-batch training. You may consider using traditional SGD classifier (mini_batch = False)")
        if no_cells <= batch_size:
            raise ValueError(
                    f"🛑 Number of cells ({no_cells}) is fewer than the batch size ({batch_size}). Decrease `batch_size`, or use SGD directly (mini_batch = False)")
        no_cells_sample = min([batch_number*batch_size, no_cells])
        starts = np.arange(0, no_cells_sample, batch_size)
        if balance_cell_type:
            celltype_freq = np.unique(labels, return_counts = True)
            len_celltype = len(celltype_freq[0])
            mapping = pd.Series(1 / (celltype_freq[1]*len_celltype), index = celltype_freq[0])
            p = mapping[labels].values
        for epoch in range(1, (epochs+1)):
            logger.info(f"⏳ Epochs: [{epoch}/{epochs}]")
            if not balance_cell_type:
                sampled_cell_index = np.random.choice(no_cells, no_cells_sample, replace = False)
            else:
                sampled_cell_index = np.random.choice(no_cells, no_cells_sample, replace = False, p = p)
            for start in starts:
                classifier.partial_fit(indata[sampled_cell_index[start:start+batch_size]], labels[sampled_cell_index[start:start+batch_size]], classes = np.unique(labels))
    return classifier

def train_1(X = None,
          labels: Optional[Union[str, list, tuple, np.ndarray, pd.Series, pd.Index]] = None,
          genes: Optional[Union[str, list, tuple, np.ndarray, pd.Series, pd.Index]] = None,
          transpose_input: bool = False,
          with_mean: bool = True,
          check_expression: bool = False,
          #LR param
          C: float = 1.0, solver: Optional[str] = None, max_iter: Optional[int] = None, n_jobs: Optional[int] = None,
          #SGD param
          use_SGD: bool = False, alpha: float = 0.0001,
          #mini-batch
          mini_batch: bool = False, batch_number: int = 100, batch_size: int = 1000, epochs: int = 10, balance_cell_type: bool = False,
          #feature selection
          feature_selection: bool = False, top_genes: int = 300, use_cytopus: bool = False, cyto_genes: Optional[np.ndarray] = None,
          #description
          date: str = '', details: str = '', url: str = '', source: str = '', version: str = '',
          #penalty
          penalty: str = 'l2', switch_penalty: bool = False,
          #other param
          **kwargs): 
    """
    Train a celltypist model using mini-batch (optional) logistic classifier with a global solver or stochastic gradient descent (SGD) learning. 
    A version of the celltypist fxn train that adds some additional choices/functions 

    Parameters
    ----------
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
    transpose_input
        Whether to transpose the input matrix. Set to `True` if `X` is provided in a gene-by-cell format.
        (Default: `False`)
    with_mean
        Whether to subtract the mean values during data scaling. Setting to `False` can lower the memory usage when the input is a sparse matrix but may slightly reduce the model performance.
        (Default: `True`)
    check_expression
        Check whether the expression matrix in the input data is supplied as required.
        Except the case where a path to the raw count table file is specified, all other inputs for `X` should be in log1p normalized expression to 10000 counts per cell.
        Set to `False` if you want to train the data regardless of the expression formats.
        (Default: `True`)
    C
        Inverse of L2 regularization strength for traditional logistic classifier. A smaller value can possibly improve model generalization while at the cost of decreased accuracy.
        This argument is ignored if SGD learning is enabled (`use_SGD = True`).
        (Default: 1.0)
    solver
        Algorithm to use in the optimization problem for traditional logistic classifier.
        The default behavior is to choose the solver according to the size of the input data.
        This argument is ignored if SGD learning is enabled (`use_SGD = True`).
    max_iter
        Maximum number of iterations before reaching the minimum of the cost function.
        Try to decrease `max_iter` if the cost function does not converge for a long time.
        This argument is for both traditional and SGD logistic classifiers, and will be ignored if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
        Default to 200, 500, and 1000 for large (>500k cells), medium (50-500k), and small (<50k) datasets, respectively.
    n_jobs
        Number of CPUs used. Default to one CPU. `-1` means all CPUs are used.
        This argument is for both traditional and SGD logistic classifiers.
    use_SGD
        Whether to implement SGD learning for the logistic classifier.
        (Default: `False`)
    alpha
        L2 regularization strength for SGD logistic classifier. A larger value can possibly improve model generalization while at the cost of decreased accuracy.
        This argument is ignored if SGD learning is disabled (`use_SGD = False`).
        (Default: 0.0001)
    mini_batch
        Whether to implement mini-batch training for the SGD logistic classifier.
        Setting to `True` may improve the training efficiency for large datasets (for example, >100k cells).
        This argument is ignored if SGD learning is disabled (`use_SGD = False`).
        (Default: `False`)
    batch_number
        The number of batches used for training in each epoch. Each batch contains `batch_size` cells.
        For datasets which cannot be binned into `batch_number` batches, all batches will be used.
        This argument is relevant only if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
        (Default: 100)
    batch_size
        The number of cells within each batch.
        This argument is relevant only if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
        (Default: 1000)
    epochs
        The number of epochs for the mini-batch training procedure.
        The default values of `batch_number`, `batch_size`, and `epochs` together allow observing ~10^6 training cells.
        This argument is relevant only if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
        (Default: 10)
    balance_cell_type
        Whether to balance the cell type frequencies in mini-batches during each epoch.
        Setting to `True` will sample rare cell types with a higher probability, ensuring close-to-even cell type distributions in mini-batches.
        This argument is relevant only if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
        (Default: `False`)
    feature_selection
        Whether to perform two-pass data training where the first round is used for selecting important features/genes using SGD learning.
        If `True`, the training time will be longer.
        (Default: `False`)
    top_genes
        The number of top genes selected from each class/cell-type based on their absolute regression coefficients.
        The final feature set is combined across all classes (i.e., union).
        (Default: 300)
    date
        Free text of the date of the model. Default to the time when the training is completed.
    details
        Free text of the description of the model.
    url
        Free text of the (possible) download url of the model.
    source
        Free text of the source (publication, database, etc.) of the model.
    version
        Free text of the version of the model.
    **kwargs
        Other keyword arguments passed to :class:`~sklearn.linear_model.LogisticRegression` (`use_SGD = False`) or :class:`~sklearn.linear_model.SGDClassifier` (`use_SGD = True`).
        
    FOUR NEW PARAMETERS: 
    penalty
        Which regularization method to use 
        (Default: "l2")
    switch_penalty
        Whether to switch the type of regualarization for the second round of model training 
        (Default: False)
    use_cytopus 
        Whether to confirm if cytopus genes are included in feature_selection (they are added if not)
        This argument is relevant only if feature selection happens (`feature_selection = True`) 
        (Default: False)
    cyto_genes
        List of gene names from ctyopus cell identities dictionary
        This argument is relevant only if feature selection with cytopus genes happens (`feature_selection = True` and `use_cytopus = True`) 

    Returns
    ----------
    :class:`~celltypist.models.Model`
        An instance of the :class:`~celltypist.models.Model` trained by celltypist.
    """
    #prepare
    logger.info("🍳 Preparing data before training")
    indata, labels, genes = _prepare_data(X, labels, genes, transpose_input)
    if isinstance(indata, pd.DataFrame):
        indata = indata.values
    elif with_mean and isinstance(indata, spmatrix):
        indata = indata.toarray()
    labels = np.array(labels)
    genes = np.array(genes)
    #check
    ##NEED TO CHANGE 10000 TO MEDIAN AMOUNT 
    print("Check expression: ", check_expression)
    print("sum: ", (np.abs(np.expm1(indata[0]).sum()-10000) > 1))
    if check_expression and (np.abs(np.expm1(indata[0]).sum()-10000) > 1):
        raise ValueError(
                "🛑 Invalid expression matrix, expect log1p normalized expression to 10000 counts per cell")
    if len(labels) != indata.shape[0]:
        raise ValueError(
                f"🛑 Length of training labels ({len(labels)}) does not match the number of input cells ({indata.shape[0]})")
    if len(genes) != indata.shape[1]:
        raise ValueError(
                f"🛑 The number of genes ({len(genes)}) provided does not match the number of genes in the training data ({indata.shape[1]})")
    #filter
    flag = indata.sum(axis = 0) == 0
    if isinstance(flag, np.matrix):
        flag = flag.A1
    if flag.sum() > 0:
        logger.info(f"✂️ {flag.sum()} non-expressed genes are filtered out")
        #indata = indata[:, ~flag]
        genes = genes[~flag]
    #report data stats
    logger.info(f"🔬 Input data has {indata.shape[0]} cells and {(~flag).sum()} genes")
    #scaler
    logger.info(f"⚖️ Scaling input data")
    
    scaler = StandardScaler(with_mean = with_mean)
    print('Scale transform done1')
    indata = scaler.fit_transform(indata[:, ~flag] if flag.sum() > 0 else indata)
    print('Scale transform done')
    indata[indata > 10] = 10
    #sklearn (Cython) does not support very large sparse matrices for the time being
    if isinstance(indata, spmatrix) and ((indata.indices.dtype == 'int64') or (indata.indptr.dtype == 'int64')):
        indata = indata.toarray()
    print('to array done')
    #max_iter
    if max_iter is None:
        if indata.shape[0] < 50000:
            max_iter = 1000
        elif indata.shape[0] < 500000:
            max_iter = 500
        else:
            max_iter = 200
    #classifier
    if use_SGD or feature_selection:
        classifier = _SGDClassifier(indata = indata, labels = labels, alpha = alpha, max_iter = max_iter, n_jobs = n_jobs, mini_batch = mini_batch, batch_number = batch_number, batch_size = batch_size, epochs = epochs, balance_cell_type = balance_cell_type, penalty = penalty, **kwargs)
    else:
        classifier = _LRClassifier(indata = indata, labels = labels, C = C, solver = solver, max_iter = max_iter, n_jobs = n_jobs, **kwargs)
    #feature selection -> new classifier and scaler
    if feature_selection:
        logger.info(f"🔎 Selecting features")
        if len(genes) <= top_genes:
            raise ValueError(
                    f"🛑 The number of genes ({len(genes)}) is fewer than the `top_genes` ({top_genes}). Unable to perform feature selection")
        gene_index = np.argpartition(np.abs(classifier.coef_), -top_genes, axis = 1)[:, -top_genes:]
        gene_index = np.unique(gene_index)
        if use_cytopus: 
            logger.info(f"🧬 {len(gene_index)} features are selected pre cytopus")
            #confirming that all cytopus genes are in the top genes used in feature selection
            #first get a list of all the indexs of cyto_genes 
            ct_gene_index = []
            for x in cyto_genes:
                if x in genes: 
                    idx = np.where(genes==x)[0][0]
                    ct_gene_index.append(idx)
            for x in ct_gene_index: 
                if x not in gene_index: 
                    gene_index = np.append(gene_index, x)
            gene_index = np.unique(gene_index)
            logger.info(f"🧬 {len(gene_index)} features are selected after cytopus")
        else:
            logger.info(f"🧬 {len(gene_index)} features are selected")
        genes = genes[gene_index]
        #indata = indata[:, gene_index]
        logger.info(f"🏋️ Starting the second round of training")
        if switch_penalty: 
            if penalty == "l2":
                penalty = "l1"
            else: 
                penalty = "l2"
        if use_SGD:
            classifier = _SGDClassifier(indata = indata[:, gene_index], labels = labels, alpha = alpha, max_iter = max_iter, n_jobs = n_jobs, mini_batch = mini_batch, batch_number = batch_number, batch_size = batch_size, epochs = epochs, balance_cell_type = balance_cell_type, penalty = penalty, **kwargs)
        else:
            classifier = _LRClassifier(indata = indata[:, gene_index], labels = labels, C = C, solver = solver, max_iter = max_iter, n_jobs = n_jobs, **kwargs)
        scaler.mean_ = scaler.mean_[gene_index]
        scaler.var_ = scaler.var_[gene_index]
        scaler.scale_ = scaler.scale_[gene_index]
        scaler.n_features_in_ = len(gene_index)
    #model finalization
    classifier.features = genes
    classifier.n_features_in_ = len(genes)
    if not date:
        date = str(datetime.now())
    description = {'date': date, 'details': details, 'url': url, 'source': source, 'version': version, 'number_celltypes': len(classifier.classes_)}
    logger.info(f"✅ Model training done!")
    return Model(classifier, scaler, description), genes

#fxn to make individual models 
def make_model(model_ver: int = 0, 
              X = None,
              labels: Optional[Union[str, list, tuple, np.ndarray, pd.Series, pd.Index]] = None,
              genes: Optional[Union[str, list, tuple, np.ndarray, pd.Series, pd.Index]] = None,
              check_expression: bool = False,
              cyto_genes: Optional[np.ndarray] = None,
              write_loc: str = 'New Models') -> Model: 
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
        model, genes = train_1(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True, balance_cell_type = True, feature_selection = True)
    
    if model_ver == 1:
        #no fs
        model, genes = train_1(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True)
    
    if model_ver == 2:
        #L1 reg
        model, genes = train_1(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True, feature_selection = True, penalty = "l1")
    
    if model_ver == 3: 
        #cytopus genes only
        model, genes = train_1(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True)
    
    if model_ver == 4: 
        #fs with cytopus genes
        model, genes = train_1(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True, feature_selection = True, balance_cell_type = True, use_cytopus = True, cyto_genes = cyto_genes)
    
    if model_ver == 5: 
        #merge of 2 & 4 
        model, genes = train_1(X = X, labels = labels, genes = genes, check_expression = False, use_SGD = True, mini_batch = True, feature_selection = True, penalty = "l1", switch_penalty = True, balance_cell_type = True, use_cytopus = True, cyto_genes = cyto_genes)
    model.write(write_loc)
    write_loc_g = write_loc+'_genes.csv'
    pd.DataFrame(genes).to_csv(write_loc_g)  
    
    return model, genes 

def make_all_models(adata_loc, annot_col, abrev, check_expression: bool = False, data_dir: str = '/data/peer/adamsj5/cell_typing/train_test_data/', write_loc: str = '/home/adamsj5/auto-cell-typing/celltypist/New Models/HuBMAP Models/'):
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
    
    write_loc_3 = write_loc+abrev+'_model_3'
    print(write_loc_3)
    model_3, genes_3 = make_model(model_ver = 3, X = indata_cp, labels = labels_cp, genes = genes_cp, check_expression = check_expression, write_loc = write_loc_3)
    print("Model 3 Done")
    
    write_loc_4 = write_loc+abrev+'_model_4'
    print(write_loc_4)
    model_4, genes_4 = make_model(model_ver = 4, X = indata, labels = labels, genes = genes, check_expression = check_expression, cyto_genes = cp_and_ct_genes, write_loc = write_loc_4)
    print("Model 4 Done")
    
    write_loc_5 = write_loc+abrev+'_model_5'
    print(write_loc_5)
    model_5, genes_5 = make_model(model_ver = 5, X = indata, labels = labels, genes = genes, check_expression = check_expression, cyto_genes = cp_and_ct_genes, write_loc = write_loc_5)
    print("Model 5 Done")
    
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
adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/LUCA_Train.h5ad'
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
"""
adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/train_glas.h5ad'
#sc.pp.filter_genes(adata_read, min_cells = 1)
annot_col = 'finer_cell_types'
abrev = 'g'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/Glasner Models/'
"""
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

adata_loc = '/data/peer/adamsj5/cell_typing/train_test_data/HuBMAP_Train.h5ad'
annot_col = 'cell_type'
abrev = 'HuBMAP_f'
write_loc = '/home/adamsj5/auto-cell-typing/celltypist/New Models/HuBMAP Models/'

make_all_models(adata_loc, annot_col , abrev ,  write_loc )