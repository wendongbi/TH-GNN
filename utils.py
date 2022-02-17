import os
import os.path as osp
from collections import defaultdict, namedtuple
# train/test function of gnn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import dgl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from dgl.data.tu import TUDataset
import random
import time
import sys
from dataset import ourGraphClassificationDataset
from scipy.sparse import linalg

# construct model
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)



def train_gnn(g, x, y, model, optimizer, train_idx, weight_node=None, Alpha=0.1, local_graph_loader=None,  epoch=None):
    # print(g.device, x.device, y.device, train_idx.device)
    model.train()
    probs, contrast_loss, logits = model(g, x, local_graph_loader)
    optimizer.zero_grad()
    if weight_node is not None:
        loss = torch.nn.BCELoss(reduction='none')(probs[train_idx], y[train_idx])
        loss = (loss * weight_node).sum()
    else:
        loss = torch.nn.BCELoss(reduction='none')(probs[train_idx], y[train_idx]).mean()
    assert epoch is not None
 

    loss_final = loss + contrast_loss * Alpha
    loss_final.backward()
    optimizer.step()
    return np.around(loss.detach().cpu(), 3), np.around(contrast_loss.detach().cpu(), 3)


# @torch.no_grad()
def test_gnn(g, x, y, model, train_val_test_idx, local_graph_loader=None, K=100):
    with torch.no_grad():
        res_bucket = {
            'loss':[],
            'f1':[],
            'precision':[],
            'recall':[]
        }
        model.eval()
        probs, _, logits = model(g, x, local_graph_loader)

        criteria = torch.nn.BCELoss()
        res_bucket = {}
        for split in ['train', 'val', 'test']:
            split_idx = train_val_test_idx[split]
            loss = criteria(probs[split_idx], y[split_idx])
            pred = (probs > 0.5).long()[split_idx]
            target = y.long()[split_idx]
            correct_all = ((pred == target).sum(1) == 1).sum(0).long()
            acc_all = correct_all / len(split_idx)

            auc = roc_auc_score(target.cpu().numpy(), probs[split_idx].cpu().numpy())
            f1 = f1_score(target.cpu().numpy(), pred.cpu().numpy(), average='binary')
            precision= precision_score(target.cpu().numpy(), pred.cpu().numpy(), average='binary')
            recall = recall_score(target.cpu().numpy(), pred.cpu().numpy(), average='binary')
            res_bucket[split] = {
                'loss': loss.cpu().item(),
                'acc_all': acc_all.cpu().numpy(),
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
            
            for key in res_bucket[split]:
                res_bucket[split][key] = np.around(res_bucket[split][key], 3)
    return res_bucket


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed(worker_info.seed % (2 ** 32))


def batcher():
    def batcher_dev(batch):
        graph_q, graph_k, graph_idx = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k, graph_idx

    return batcher_dev


def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)

    return batcher_dev


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x

def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    n = g.number_of_nodes()
    # adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    adj = g.adjacency_matrix(transpose=False, scipy_fmt="csr").astype(float)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g


def build_dataset(
    path_X='financial_statement.npy',\
    path_Y='risk_label.npy',\
    path_tribes='/home/biwendong/code/data_financial/graph_files/listed_graph_raw',
    path_tribes_order='/home/biwendong/code/data_financial/graph_files/order_listed_graph_files_by_node_id.json'):
    # prepare node attributes (financial statements) and labels (company financial risks) .
    data_mat = np.load(path_X)
    label_mat = np.load(path_Y)
    if len(label_mat.shape) == 2:
        lbl_dim=label_mat.shape[1]
    else:
        lbl_dim = 1
    print('label_dim:', lbl_dim)



    # prepare tribe sub-graph dataloader
    print('preparing subgraph dataloader...')
    t0 = time.time()
    local_graph_dataset = ourGraphClassificationDataset(
            graph_file_dir=path_tribes,
            rw_hops=30,
            subgraph_size=256,
            restart_prob=0.8,
            positional_embedding_size=32,
            entire_graph=True,
            order_graph_files_path=path_tribes_order,
        )
    local_graph_loader = torch.utils.data.DataLoader(
            dataset=local_graph_dataset,
            batch_size=4040,
            collate_fn=batcher(),
            shuffle=False,
            num_workers=0,
            worker_init_fn=worker_init_fn,
        )
    print('Subgraph dataloader completed, using {:.3f}s'.format(time.time() - t0))
    # # valid check
    # for graph_idx in local_graph_dataset.mapper_idx2name:
    #     company_name = local_graph_dataset.mapper_idx2name[graph_idx].split('.json')[0]
    #     assert graph_idx == mapper_name2node_id[company_name]

    # prepare 
    g = dgl.load_graphs('./news_graph.bin')[0][0]
    return data_mat, label_mat, local_graph_loader, g