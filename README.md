# Network-Science_Challenge
Group project in ML for Network Science Class at CentraleSupelec 

## About the Project

This repository contains our work during the Kaggle challenge within the Machine Learning in Network Science class at CentraleSupélec. The challenge was about predicting missing links in an actor co-occurence network and we reached a score of 78.17 \% on the public test set (ranked 4th).

Link to challenge:
- https://www.kaggle.com/competitions/mlns-2023/data

## Repository organization
```
.
├── data 
│   *folder containing data files from original repostiory as well as generated files during feature extraction*
├── models
│   *folder containing VGNAE models (current, best and fintuned)*
│    ├── best so far
│    ├── finetuned
├── resources
│   *folder containing research papers referenced as well as chart/plot pictures for report*
├── util
│   *folder containing python scripts*
│    ├── analyse_Data.py *Data analysis helper functions*
│    ├── autoencoder.py *VGNAE Autoencoder model script and training/tuning pipeline*
│    ├── load_Data.py *Data loading script*
│    ├── modeling.py *Modeling helper functions*
│    ├── node2vec.py *Node2vec model script and training/tuning pipeline*
│    ├── preprocess_Data.py *Data preprocessing script*
└── 0_Exploratory Data Analysis and Preprocessing.ipynb
│   *python notebook*
└── 1_Global_Feature_Extraction.ipynb
│   *python notebook for Rooted Pagerank and Simrank finetuning and extraction*
└── 1_Node Embedding_Node2vec.ipynb
│   *python notebook for finetuning of Node2vec algorithm*
└── 1_Node Embedding_VGNAE.ipynb
│   *python notebook for training and finetuning of VGNAE*
└── Link_Prediction_Supervised.ipynb
│   *python notebook for supervised link prediction model training and finetuning*
└── Link_Prediction_Unsupervised.ipynb*
    *python notebook for unsupervised link prediction model finetuning*
```

## References
1. E. C. Mutlu, T. Oghaz, A. Rajabi, and I. Garibay, “Review on learning
and extracting graph features for link prediction,” Mach. Learn. Knowl.
Extr. 2, 672–704 (2020).
2. S. J. Ahn and M. Kim, “Variational graph normalized AutoEncoders,” in
Proceedings of the 30th ACM International Conference on Information
&amp Knowledge Management, (ACM, 2021).
3. Y.-L. Lee and T. Zhou, “Collaborative filtering approach to link predic-
tion,” Phys. A: Stat. Mech. its Appl. 578, 126107 (2021).
4. A. Papadimitriou, P. Symeonidis, and Y. Manolopoulos, “Fast and
accurate link prediction in social networking systems,” J. Syst. Softw.85, 2119–2132 (2012). Selected papers from the 2011 Joint Working
IEEE/IFIP Conference on Software Architecture (WICSA 2011).
5. A. Grover and J. Leskovec, “node2vec: Scalable feature learning for
networks,” CoRR. abs/1607.00653 (2016)
