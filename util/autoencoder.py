# import own scripts
import util.load_Data as loadData

# import external packages
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GAE, VGAE, APPNP
import torch_geometric.transforms as T
from ray import tune
from ray import tune, air
from ray.tune import JupyterNotebookReporter
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

# ignore warnings that show in every raytune run
import warnings
warnings.simplefilter(action = "ignore", category = np.VisibleDeprecationWarning)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scaling_factor, num_propagations, teleport_probability, dropout):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.scaling_factor = scaling_factor
        self.propagate = APPNP(K = num_propagations, alpha = teleport_probability, dropout = dropout)

    def forward(self, x, edge_index):
        x_ = self.linear1(x)
        x_ = self.propagate(x_, edge_index)

        x = self.linear2(x)
        x = F.normalize(x,p=2,dim=1) * self.scaling_factor
        x = self.propagate(x, edge_index)
        return x, x_

def load(testing_ratio = 0.3):
    # load data
    (G, G_train, node_info, train_tf, val_tf, trainval_tf, test, test_tf) = loadData.load(testing_ratio)

    # get train and validation masks
    trainval_tf = (trainval_tf
        .assign(train_mask = lambda df_: [True if idx in train_tf.index else False for idx in df_.index])
        .assign(val_mask = lambda df_: ~df_.train_mask)
    )

    # initialise PyTorch Geometric Dataset
    data = Data(x = torch.tensor(node_info.values, dtype = torch.float32),
                train_pos_edges = torch.tensor(
                    trainval_tf.loc[(trainval_tf.y == 1) & (trainval_tf.train_mask == 1)][["node1", "node2"]].values
                ).T,
                train_neg_edges = torch.tensor(
                    trainval_tf.loc[(trainval_tf.y == 0) & (trainval_tf.train_mask == 1)][["node1", "node2"]].values
                ).T,
                val_pos_edges = torch.tensor(
                    trainval_tf.loc[(trainval_tf.y == 1) & (trainval_tf.val_mask == 1)][["node1", "node2"]].values
                ).T,
                val_neg_edges = torch.tensor(
                    trainval_tf.loc[(trainval_tf.y == 0) & (trainval_tf.val_mask == 1)][["node1", "node2"]].values
                ).T,
                train_edges = torch.tensor(
                    trainval_tf.loc[trainval_tf.train_mask == 1][["node1", "node2"]].values
                ).T,
                val_edges = torch.tensor(
                    trainval_tf.loc[trainval_tf.val_mask == 1][["node1", "node2"]].values
                ).T,
                trainval_edges = torch.tensor(
                    trainval_tf[["node1", "node2"]].values
                ).T,
                trainval_pos_edges = torch.tensor(
                    trainval_tf.loc[trainval_tf.y == 1][["node1", "node2"]].values
                ).T,
                test_edges = torch.tensor(
                    test_tf.values
                ).T)

    # preprocess data
    data = T.NormalizeFeatures()(data)

    return data, (G, G_train, node_info, train_tf, val_tf, trainval_tf, test, test_tf)

def get_device(model = None):
    # where we want to run the model (so this code can run on cpu, gpu, multiple gpus depending on system)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1 and model is not None:
            model = nn.DataParallel(model)
    if model is not None:
        return device, model.to(device)
    return device

def train_validate(config):
    # how many epochs we want to train for (at maximum)
    max_epochs = int(config["max_epochs"])

    # load data
    data = config["data"]

    # model initialisation
    if config["model"] == "VGNAE":
        model = VGAE(Encoder(data.x.size()[1], config["enc_channels"], config["scaling"],
                             config["num_prop"], config["teleport"], config["dropout"]))

    # initialise device
    device, model = get_device(model)

    # move data to device
    data.to(device)

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = config["wd"])

    # metrics
    max_val_acc = 0

    # helper function
    def validate(pos_edges, neg_edges):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.train_pos_edges)

        # get preds
        pos_y = z.new_ones(pos_edges.size(1))
        neg_y = z.new_zeros(neg_edges.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred_probas = model.decoder(z, pos_edges, sigmoid=True)
        neg_pred_probas = model.decoder(z, neg_edges, sigmoid=True)
        pred_probas = torch.cat([pos_pred_probas, neg_pred_probas], dim=0)

        y, pred_probas = y.detach().cpu().numpy(), pred_probas.detach().cpu().numpy()
        pred = (pd.DataFrame(pred_probas)
            .rename(columns = {0: "pred_probas"})
            .assign(pred = lambda df_: (df_.pred_probas > df_.pred_probas.median()).astype(int))
        )
        # compute scores
        auc = roc_auc_score(y, pred_probas)
        acc = accuracy_score(y, pred.pred)

        return auc, acc
    
    for epoch in range(1, max_epochs + 1):
        ##TRAINING##
        model.train()
        optimizer.zero_grad()

        # forward + backward + optimize
        z = model.encode(data.x, data.train_pos_edges)
        loss = model.recon_loss(z, data.train_pos_edges)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

        # compute stats
        trn_loss = loss.item()
        trn_auc, trn_acc = validate(data.train_pos_edges, data.train_neg_edges)

        ##VALIDATION##
        model.eval()
        with torch.no_grad():
            # forward
            z = model.encode(data.x, data.train_pos_edges)
            loss = model.recon_loss(z, data.val_pos_edges, data.val_neg_edges)
            loss = loss + (1 / data.num_nodes) * model.kl_loss()

            # compute stats
            val_loss = loss.item()
            val_auc, val_acc = validate(data.val_pos_edges, data.val_neg_edges)
        
        ##SAVE current best models##
        if config["save"]:
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                path = os.path.abspath("")+"\\autoencoder.pt"
                torch.save(model.state_dict(), path)

        ##REPORT##
        if config["verbose"]:          
            print('Epoch: [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train ACC: {:.4f}, Val ACC: {:.4f}'.format(epoch, max_epochs,
                                                                                                                    trn_loss, val_loss,
                                                                                                                    trn_acc, val_acc))

        if config["ray"]:
            tune.report(trn_loss = trn_loss, val_loss = val_loss,
                        trn_auc = trn_auc, val_auc = val_auc, max_val_acc = max_val_acc,
                        trn_acc = trn_acc, val_acc = val_acc)
            
def get_embeddings(model, x, pos_edges):
    embeddings = model.encode(x, pos_edges)
    return embeddings.detach().cpu().numpy()

def get_similarity(model, x, pos_edges, pred_edges):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, pos_edges)
    pred = model.decoder(z, pred_edges, sigmoid = True)
    return pred.detach().cpu().numpy()
            
def trial_str_creator(trial):
    """
    Trial name creator for ray tune logging.
    """
    model = trial.config["model"]
    lr    = trial.config["lr"]
    wd    = trial.config["wd"]
    return f"{model}_{lr}_{wd}_{trial.trial_id}"

def run_ray_experiment(train_func, config, ray_path, num_samples, metric_columns, parameter_columns):

    reporter = JupyterNotebookReporter(
        metric_columns = metric_columns,
        parameter_columns= parameter_columns,
        max_column_length = 15,
        max_progress_rows = 20,
        max_report_frequency = 1, # refresh output table every second
        print_intermediate_tables = True
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"CPU": 16, "GPU": 1}
        ),
        tune_config = tune.TuneConfig(
            metric = "trn_loss",
            mode = "min",
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