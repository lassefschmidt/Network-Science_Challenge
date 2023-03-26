# import own scripts
import util.load_Data as loadData

# import external packages
import os
import json
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder, AverageEmbedder, WeightedL1Embedder, WeightedL2Embedder
from ray import tune, air
from ray.tune import JupyterNotebookReporter
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import PredefinedSplit
import pandas as pd

# ignore warnings that show in every raytune run
import warnings
warnings.simplefilter(action = "ignore", category = np.VisibleDeprecationWarning)

def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # same for pytorch
    random_seed = 1 # or any of your favorite number 

def train_validate(config):
    # ensure reproduction
    set_reproducible()

    # load data
    G = config["graph"]

    # node2vec walk generations
    node2vec = Node2Vec(G, 
                        dimensions=config["dimensions"], 
                        walk_length=config["walk_length"], 
                        num_walks=config["num_walks"], 
                        workers=1, 
                        p=config["p"],
                        q=config["q"]
                        )

    # embed nodes
    model = node2vec.fit(window=10)

    # embed edges
    edge_embs = config["operator"](keyed_vectors=model.wv)
    
    # enrich edgelists
    trainval_tf = pd.concat([config["trainval_tf"], pd.DataFrame([edge_embs[(str(u), str(v))] for u, v in zip(config["trainval_tf"].node1, config["trainval_tf"].node2)], index=config["trainval_idx"])], axis=1)
    val_tf = pd.concat([config["val_tf"], pd.DataFrame([edge_embs[(str(u), str(v))] for u, v in zip(config["val_tf"].node1, config["val_tf"].node2)], index=config["val_idx"])], axis=1)
    X_trainval, y_trainval = loadData.split_frame(trainval_tf)
    X_val, y_val = loadData.split_frame(val_tf)
    X_trainval.columns = X_trainval.columns.astype(str)
    X_val.columns = X_val.columns.astype(str)

    # prepare predefined split for cv
    val_fold = [0 if i in X_val.index else -1 for i in X_trainval.index]
    ps = PredefinedSplit(val_fold)

    # fit logistic 
    clf = LogisticRegressionCV(max_iter = 10000, cv = ps)
    clf.fit(X_trainval, y_trainval)

    # predict on train set
    y_train_hat = clf.predict(X_trainval)
    # compute accuracy
    acc_train = accuracy_score(y_trainval, y_train_hat)

    # predict on validation set
    y_val_hat = clf.predict(X_val)
    # compute accuracy
    acc_val = accuracy_score(y_val, y_val_hat)

    ##REPORT##
    if config["ray"]:
        tune.report(acc_train = acc_train, acc_val = acc_val)

def trial_str_creator(trial):
    """
    Trial name creator for ray tune logging.
    """
    model = trial.config["model"]
    dim   = trial.config["dimensions"]
    lr    = trial.config["p"]
    wd    = trial.config["q"]
    return f"{model}_{dim}_{lr}_{wd}_{trial.trial_id}"

def run_ray_experiment(train_func, config, ray_path, num_samples, metric_columns, parameter_columns):

    reporter = JupyterNotebookReporter(
        metric_columns = metric_columns,
        parameter_columns= parameter_columns,
        max_column_length = 15,
        max_progress_rows = 20,
        max_report_frequency = 30, # refresh output table every ten seconds
        print_intermediate_tables = True
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"CPU": 4, "GPU": 0}
        ),
        tune_config = tune.TuneConfig(
            metric = "acc_val",
            mode = "max",
            num_samples = num_samples,
            trial_name_creator = trial_str_creator,
            trial_dirname_creator = trial_str_creator,
            ),
        run_config = air.RunConfig(
            local_dir = ray_path,
            progress_reporter = reporter,
            verbose = 1),
        param_space = config
    )

    result_grid = tuner.fit()
    
    return result_grid

def open_validate_ray_experiment(experiment_path, trainable):
    # open & read experiment folder
    print(f"Loading results from {experiment_path}...")
    restored_tuner = tune.Tuner.restore(experiment_path, trainable = trainable, resume_unfinished = False)
    result_grid = restored_tuner.get_results()
    print("Done!\n")

    # Check if there have been errors
    if result_grid.errors:
        print(f"At least one of the {len(result_grid)} trials failed!")
    else:
        print(f"No errors! Number of terminated trials: {len(result_grid)}")
        
    return restored_tuner, result_grid