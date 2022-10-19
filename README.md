# TH-GNN

Implementation of the KDD2022 paper, Company-as-Tribe: Company Risk Assessment On Tribe-Style Graph With Hierarchical Graph Neural Networks.

TH-GNN (our proposed model) is a hierarchical graph neural network model for company financial risk assessment tasks. Each listed company with its investment graph is viewed as a tribe, which is more informative than  a single company. 

![model_overview](model_structure.png)

## Installation

### Requirements

* Linux with Python >= 3.7

* Pytorch >= 1.9.0

* DGL >= 0.7.0

* Scikit-learn >= 0.24.2

* Network >= 2.6.2

* Numpy >= 1.19.2

  

## Quick Start

### Dataset

Due to company privacy and copyright issues in the design of the datasets that we used, we do not disclose the company attributes and risk labels used in the paper.

### Training

We provide an example for training the TH-GNN model in ./run.sh

```shell
python3 main.py \
 --gpu 0 \
 --local_num_layer 2 \
 --hidden 64 \
 --num_epoch 100 \
 --tribe_encoder_gnn gin \
 --lr 3e-3 \
 --weight_decay 5e-3 \
 --fusion_mode mlp \
 --path_x {path to node attribute file} \
 --path_y {path to node label file} \
 --path_tribe_files {path to tribe-graph files} \
 --path_tribe_order {path to tribe_graph order file}
```

