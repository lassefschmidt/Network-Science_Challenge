# Network-Science_Challenge
Group project in ML for Network Science Class at CentraleSupelec 

Link to challenge:
- https://www.kaggle.com/competitions/mlns-2023/data

Modeling:
- https://github.com/dadacheng/walkpooling as end-to-end model for link prediction (state of the art on the CORA dataset which we will use for our challenge, so any implementation here will also help us for the final project)
- use MLP (same input as XGBOOST) to predict links
- use Graph Attention Network (GATConv Layers in Pytorch Geometric) --> maybe use VGNAE similarities as edge weights (?), but probably bad idea and just use GATConv directly
