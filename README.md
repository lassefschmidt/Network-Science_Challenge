# Network-Science_Challenge
Group project in ML for Network Science Class at CentraleSupelec 

Link to challenge:
- https://www.kaggle.com/competitions/mlns-2023/data

Modeling:
- https://github.com/dadacheng/walkpooling as end-to-end model for link prediction (state of the art on the CORA dataset which we will use for our challenge, so any implementation here will also help us for the final project)
- use Logistic Regression
- finetune XGBoost
- use MLP (same input as XGBOOST) to predict links
- use Graph Attention Network (GATConv Layers in Pytorch Geometric) --> maybe use VGNAE similarities as edge weights (?), but probably bad idea and just use GATConv directly (see https://github.com/PetarV-/GAT)

Great papers and stuff to cite:
- https://graph-neural-networks.github.io/static/file/chapter10.pdf
- https://mdpi-res.com/d_attachment/make/make-02-00036/article_deploy/make-02-00036-v2.pdf?version=1608552901
