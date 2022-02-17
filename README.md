# MM-GNN

Original implementation for the submission paper: Towards Modeling Of Neighbors' Feature Distribution:  Mix-Moment GNN With Adaptive Attention Mechanism.

Mix-Moment GNN (MM-GNN) is a Graph Neural Network Model that introduce Moment Methodes to GNN and use neighbors' feature distribution to enhance GNN-based methods. 

![model_structure](/Users/biwendong/Desktop/research/msra/KDD22_code/MM-GNN-main/model_structure.png)

## Installation

### Requirements

* Linux with Python $\geq$ 3.7
* PyTorch $\geq$ 1.7.1
* Torch-geometric $\geq$ 2.0.2
* Torch-scatter $\geq$ 2.0.7
* Torch-sparse $\geq$ 0.6.9

## Quick Start

## Datasets

The datasets used in this paper include 9 social graph datasets from Facebook Social Network: UGA50, GWU54, Northeastern19, Hamilton46, Caltech36, Howard90, UF21, Simmons81, Tulane29. All these 9 datasets are provided in this repository at path: `./dataset/facebook100`. 

For other datasets used in the appendix, `torch_geometric.datasets` provides the dataset API and data spilts in the corresponding papers. 

### Training

#### MM-GNN

We provide an example for running the script in MMGNN/run.sh

```shell
python3 main.py \
    --model MM_GNN \
    --num_layer 2 \
    --repeat 9 \
    --num_epoch 200 \
    --gpu 0 \
    --data_dir ../data/Facebook100_pyg \
    --dataset GWU54 \
    --moment 3 \
    --hidden 16 \
    --mode attention \
    --auto_fixed_seed \
    --use_adj_norm \
```

Please feel free to E-mail me [biwendong20@mails.ucas.ac.cn](biwendong20@mails.ucas.ac.cn) if you have any questions on running the code.

