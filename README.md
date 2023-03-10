# Network-Science_Challenge
Group project in ML for Network Science Class at CentraleSupelec 

Useful links:

- https://www.kaggle.com/competitions/mlns-2023/data
- https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
- https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970
- https://github.com/VHRanger/nodevectors/blob/master/examples/link%20prediction%20-%20FULL%20RUN.ipynb (great package for node embeddings based on Random Walks!)


Feature engineering:
- decision rules --> only keep those that have predictive power (instead of removing the negative decision rules)
- how to incorporate node features into encoding of each node? e.g. for each edge compute node embedding similarity and set it as edge weight --> then run encoding (node2vec or deep graph encoder) on the weighted graph
