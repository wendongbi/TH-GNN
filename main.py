import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import json
import scipy
import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv
import numpy as np
import scipy
import time
import sys
sys.path.append('./models')
from model import THGNN
from dataset import ourGraphClassificationDataset
from utils import  worker_init_fn, batcher
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# from dgl.nn.pytorch.utils import JumpingKnowledge

# prepare dataset
with open('../data_financial/processed_company_data/business_risk_2020_frist3seasons_multi-hot.json', 'r') as fin:
    business_label_dict = json.load(fin)
with open('../data_financial/processed_company_data/finance_risk_2020_frist3seasons_multi-hot.json', 'r') as fin:
    finance_label_dict = json.load(fin)
with open('../data_financial/processed_company_data/company_expert_index_merge_2019final.json', 'r') as fin:
# with open('processed_company_data/company_expert_index_merge.json', 'r') as fin: 
    company_data_bucket = json.load(fin)
############################ remove possible risk companys current in wind_label_table ###############################
with open('../data_financial/possible_non_risk_companys.json', 'r') as fin:
    possbile_non_risk_companys = json.load(fin)
print('../data_financial/possible_non_risk_companys num:', len(possbile_non_risk_companys))


with open('../data_financial/graph_files/nodes/listed_company_id2nameANDidx.json', 'r') as fin:
    listed_company_id2nameANDidx = json.load(fin)
mapper_name2node_id = {}
for _id in listed_company_id2nameANDidx:
    company_name, node_id = listed_company_id2nameANDidx[_id]
    mapper_name2node_id[company_name] = node_id
print(len(mapper_name2node_id))

node_num = len(mapper_name2node_id)
node_dim = len(company_data_bucket[list(company_data_bucket.keys())[0]]) - 1
data_mat = np.zeros((node_num, node_dim))
label_mat = np.ones(node_num).astype(int) * -1 # complete missing lbl as -1
# label_mat = np.zeros(node_num).astype(int) # complete missing lbl as non-risks
print(len(company_data_bucket))
for company_id in company_data_bucket:
    company_name = company_data_bucket[company_id][0]
    if company_name in mapper_name2node_id:
        node_id = mapper_name2node_id[company_name]
        node_feat = company_data_bucket[company_id][1:]
        assert len(node_feat) == data_mat.shape[1]
        data_mat[node_id] = node_feat
        if company_id in business_label_dict or company_id in finance_label_dict:
            label_mat[node_id] = 1
        # elif company_id in possbile_non_risk_companys:
            # label_mat[node_id] = 0
        # elif (label_mat == 0).sum() < 1698:
        else:
            label_mat[node_id] = 0
print('feature-missing node num:', (data_mat.sum(1) == 0).sum())
print('label-missing node num:', (label_mat == -1).sum())
print('pos_num:{}, neg_num:{}'.format((label_mat == 1).sum(), (label_mat == 0).sum()))
print(data_mat.shape, label_mat.shape)

# process nan case
mask_nan = np.isnan(data_mat)
print('original nan rate:', np.sum(mask_nan) / (len(mask_nan) * len(mask_nan[0]))) # nan rate
thres_nan = 0.6
dim_select = []

for idx in range(data_mat.shape[1]):
    nan_rate = np.isnan(data_mat[:, idx]).sum() / (len(data_mat))
    if nan_rate < thres_nan:
        dim_select.append(idx)
    print('dim:{} | nan_rate:{:.2f}%'.format(idx, nan_rate * 100))
print('dim select:', dim_select, len(dim_select))


data_mat = data_mat[:, dim_select]
mask_nan = np.isnan(data_mat)
print('[del_nan_col] nan rate:', np.sum(mask_nan) / (len(mask_nan) * len(mask_nan[0]))) # nan rate

mask_nan = np.isnan(data_mat)
data_mat[mask_nan] = 0

mask_nan = np.isnan(data_mat)
print('processed nan rate:', np.sum(mask_nan) / (len(mask_nan) * len(mask_nan[0]))) # nan rate
mask_inf = np.isinf(data_mat)
print('inf num:', mask_inf.sum())
data_mat[mask_inf] = 0

# # row normalization
def print_dim_statistic(x, thres_range=50):
    dim_select = []
    for idx in range(x.shape[1]):
        x_dim = x[:, idx]
        range_dim = np.abs(x_dim).max() - np.abs(x_dim).mean()
        print('dim:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | range:{:.3f}'.format(idx, x_dim.min(), x_dim.max(), x_dim.mean(), range_dim))
        if range_dim < thres_range:
            print(f'dim-{idx} < {thres_range}, so continue.')
            continue
        dim_select.append(idx)
    return dim_select
dim_need_to_norm = print_dim_statistic(data_mat)
from sklearn.preprocessing import Normalizer, KBinsDiscretizer
# # row normalization
# data_mat = Normalizer(norm='l1').fit_transform(data_mat)
# print(data_mat)
# print_dim_statistic(data_mat)

# # bins strategy 1
# bins_encoder = KBinsDiscretizer(n_bins=50, encode='onehot-dense', strategy='quantile')
# data_mat = bins_encoder.fit_transform(data_mat)

# bins strategy 2
bins_encoder = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile')  # uniform, quantile, kmeans
mask_sign = data_mat < 0
data_mat[mask_sign] *= -1
# data_mat[dim_need_to_norm] = bins_encoder.fit_transform(data_mat[dim_need_to_norm])
data_mat = bins_encoder.fit_transform(data_mat)
data_mat[mask_sign] *= -1
print('data_mat.shape:', data_mat.shape)

if len(label_mat.shape) == 2:
    lbl_dim=label_mat.shape[1]
else:
    lbl_dim = 1
print('label_dim:', lbl_dim)

# construct model
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# train/test function of gnn
from sklearn.metrics import f1_score, precision_score, recall_score

def train_gnn(g, x, y, model, optimizer, train_idx, weight_node=None, Alpha=0.1, Beta=0.1, local_graph_loader=None, \
    use_retrieval_loss=False, K=100, epoch_start_compute_retrieval_loss=10, epoch=None):
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
    if use_retrieval_loss and epoch >= epoch_start_compute_retrieval_loss:
        # for ranking retrieval loss
        # sim_memory = torch.mm(logits, model.memory[train_val_test_idx['train']].T) # [logits_memory]only use training set memory, N * N_train

        probs_dist = torch.cat([1 - probs.detach(), probs.detach()], dim=-1) # [2-dim probs_memory]
        sim_memory = torch.mm(probs_dist, model.memory[train_val_test_idx['train']].T) # [current-memory sim]only use training set memory, N * N_train

        # sim_memory = torch.mm(model.memory, model.memory[train_idx].T) # [memory-memory sim]only use training set memory, N * N_train

        print('#' *20, probs_dist.shape, model.memory[train_val_test_idx['train']].T.shape, sim_memory.shape)

        mask_y = y[train_idx].view(1, -1).repeat(x.shape[0], 1) # N * N_train
        mask_y[mask_y == 0] = -1 # replace label 0 as -1
        # remove the node itself to compute similarity
        sim_memory[train_idx, torch.arange(len(train_idx))] = -100
        # balance sample weight by propotion of positive and negative samples
        pos_train_num = (y[train_idx] == 1).sum().item()
        neg_train_num = (y[train_idx] == 0).sum().item()
        assert pos_train_num + neg_train_num == len(train_idx)
        pos_train_weight = neg_train_num / (neg_train_num + pos_train_num)
        neg_train_weight = pos_train_num / (neg_train_num + pos_train_num)
        mask_y[mask_y == 1] *= pos_train_weight
        mask_y[mask_y == -1] *= neg_train_weight
        # topK ranking
        topk_value, topk_col_idx = sim_memory.topk(k=K, dim=1)
        topk_value = F.softmax(topk_value, dim=1)
        topk_col_idx = topk_col_idx.view(-1)
        topk_row_idx = torch.arange(x.shape[0]).view(-1, 1).repeat(1, K).view(-1)
        topk_y_mat = mask_y[topk_row_idx, topk_col_idx].view(x.shape[0], K)
        ranking_logits = (topk_y_mat * topk_value).sum(1).view(x.shape[0], 1)
        # ranking_logits = topk_y_mat.sum(1).view(x.shape[0], 1) / 100
        ranking_probs = F.sigmoid(ranking_logits)
        epsilon = 1e-8
        retrieval_loss = F.kl_div(
            input=torch.cat([1 - ranking_probs[train_idx], ranking_probs[train_idx]], dim=-1).add(epsilon).log(), \
            target=torch.cat([1 - probs[train_idx], probs[train_idx]], dim=-1), \
            reduction='batchmean') # batchmean, mean
        retrieval_loss = F.kl_div(
            input=torch.cat([1 - probs[train_idx], probs[train_idx]], dim=-1).add(epsilon).log(), \
            target=torch.cat([1 - ranking_probs[train_idx], ranking_probs[train_idx]], dim=-1), \
            reduction='batchmean') # batchmean, mean
        # retrieval_loss *= 0.5
        print('epoch:{} retrieval loss:{:.4f}'.format(epoch, retrieval_loss.detach().item()))
        loss += Beta * retrieval_loss




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

        # for ranking res eval
        probs_dist = torch.cat([1 - probs.detach(), probs.detach()], dim=-1)
        sim_memory = torch.mm(probs_dist, model.memory[train_val_test_idx['train']].T) # only use training set memory, N * N_train
        # sim_memory = torch.mm(model.memory, model.memory[train_val_test_idx['train']].T)
        mask_y = y[train_val_test_idx['train']].view(1, -1).repeat(x.shape[0], 1) # N * N_train
        mask_y[mask_y == 0] = -1 # replace label 0 as -1
        # remove the node itself to compute similarity
        sim_memory[train_val_test_idx['train'], torch.arange(len(train_val_test_idx['train']))] = -100
        # balance sample weight by propotion of positive and negative samples
        pos_train_num = (y[train_val_test_idx['train']] == 1).sum().item()
        neg_train_num = (y[train_val_test_idx['train']] == 0).sum().item()
        assert pos_train_num + neg_train_num == len(train_val_test_idx['train'])
        pos_train_weight = neg_train_num / (neg_train_num + pos_train_num)
        neg_train_weight = pos_train_num / (neg_train_num + pos_train_num)
        mask_y[mask_y == 1] *= pos_train_weight
        mask_y[mask_y == -1] *= neg_train_weight
        # topK ranking
        topk_value, topk_col_idx = sim_memory.topk(k=K, dim=1)
        topk_value = F.softmax(topk_value, dim=1)
        topk_col_idx = topk_col_idx.view(-1)
        topk_row_idx = torch.arange(x.shape[0]).view(-1, 1).repeat(1, K).view(-1)
        topk_y_mat = mask_y[topk_row_idx, topk_col_idx].view(x.shape[0], K)
        ranking_logits_weight = (topk_y_mat * topk_value).sum(1).view(x.shape[0], 1)
        ranking_pred_weight = (ranking_logits_weight > 0).long()
        ranking_logits_vote = topk_y_mat.sum(1).view(x.shape[0], 1) / 100
        ranking_pred_vote = (ranking_logits_vote > 0).long()
        # print(topk_value[0, :10].cpu().numpy(), topk_y_mat[0, :10].cpu().numpy())
        # print(probs[0].item(), ranking_logits_weight[0].item(), ranking_logits_vote[0].item(), ranking_pred_vote[0].item(), ranking_pred_weight[0].item(), y[0].item())


        criteria = torch.nn.BCELoss()
        res_bucket = {}
        for split in ['train', 'val', 'test']:
            split_idx = train_val_test_idx[split]
            loss = criteria(probs[split_idx], y[split_idx])
            pred = (probs > 0.5).long()[split_idx]
            target = y.long()[split_idx]
            correct_all = ((pred == target).sum(1) == lbl_dim).sum(0).long()
            acc_all = correct_all / len(split_idx)

            # ranking res scores
            f1_ranking_weight = f1_score(target.cpu().numpy(), ranking_pred_weight[split_idx].cpu().numpy(), average='binary')
            f1_ranking_vote = f1_score(target.cpu().numpy(), ranking_pred_vote[split_idx].cpu().numpy(), average='binary')
            
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
                'f1_rank_weight': f1_ranking_weight,
                'f1_rank_vote': f1_ranking_vote,
                'auc': auc
            }
            
            for key in res_bucket[split]:
                res_bucket[split][key] = np.around(res_bucket[split][key], 3)
    return res_bucket

# prepare subgraph dataloader
print('preparing subgraph dataloader...')
t0 = time.time()
local_graph_dataset = ourGraphClassificationDataset(
        graph_file_dir='/home/biwendong/code/data_financial/graph_files/listed_graph_raw',
        rw_hops=30,
        subgraph_size=256,
        restart_prob=0.8,
        positional_embedding_size=32,
        entire_graph=True,
        order_graph_files_path='/home/biwendong/code/data_financial/graph_files/order_listed_graph_files_by_node_id.json',
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
# valid check
for graph_idx in local_graph_dataset.mapper_idx2name:
    company_name = local_graph_dataset.mapper_idx2name[graph_idx].split('.json')[0]
    assert graph_idx == mapper_name2node_id[company_name]

# construct global graph
print('constructing global graph')
hg = dgl.load_graphs('../data_financial/graph_files/HG_company_6.bin')[0][0]

g_news_2020 = hg['listed_company-news_2020-listed_company']
print('l-news_2020-l', g_news_2020.num_edges())

g_news_2019 = hg['listed_company-news_2019-listed_company']
print('l-news_2019-l', g_news_2019.num_edges())

adj_ensemble = g_news_2020.adj() + g_news_2019.adj()
adj_ensemble = adj_ensemble.to_dense()
adj_ensemble = adj_ensemble + adj_ensemble.T
adj_ensemble = (adj_ensemble > 0).long()
adj_ensemble = scipy.sparse.csr_matrix(adj_ensemble)
g = dgl.from_scipy(adj_ensemble)
print('0-degree node num:', (g.in_degrees() == 0).sum().item())
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
print('global graph:', g)




###################################################### main ###########################################
# Training
x = torch.from_numpy(data_mat).float()
print(x.shape, g)
class_idx_bucket = {}


test_scores_bucket = []
data_seed = 0
# model_seed = 1
for data_seed in range(1):
    set_random_seed(data_seed)
    data_idx_select = np.where(label_mat != -1)[0]

    print(data_idx_select, label_mat)
    label_mat_select = label_mat[data_idx_select]
    train_idx,val_idx,train_label,val_label = train_test_split(data_idx_select,label_mat_select,test_size=0.4, stratify=label_mat_select)
  
    val_idx,test_idx,val_label,test_label = train_test_split(val_idx,val_label,test_size=0.5, stratify=val_label)
        
    print(len(train_idx), len(val_idx), len(test_idx))
    y = torch.from_numpy(label_mat.reshape(-1, 1)).float()

    import warnings
    warnings.filterwarnings('ignore')
    # set_random_seed(0)
    GPU_ID = 1
    device = torch.device('cuda:{}'.format(GPU_ID)) if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print('device', device)
    print('begin training model...')

    for model_seed in range(10):
        print('train model with data_seed:{} and model_seed:{}'.format(data_seed, model_seed))
        set_random_seed(model_seed)

        model = THGNN(in_size=x.shape[1], out_size=1, hidden=64, norm=False)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=5e-3)
        num_epoch = 100
        x = x.to(device)
        y = y.to(device)
        g = g.to(device)
        # train_idx = torch.Tensor(train_idx).long().to(device)
        # val_idx = torch.Tensor(val_idx).long().to(device)

        train_val_test_idx = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        # weight_node = get_weight_node(label_mat=label_mat)
        # weight_node = weight_node.to(device).view(-1, 1)[train_idx]
        history_train = {
            'loss': [],
            'acc_all': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'f1_rank_weight': [],
            'f1_rank_vote': [],
            'auc': []
        }
        history_val = {
            'loss': [],
            'acc_all': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'f1_rank_weight': [],
            'f1_rank_vote': [],
            'auc': []
        }
        history_loss_train = []
        best_score = {
            'val_score': 0,
            'test_res': None,
            'val_res': None
        }
        choose_by = 'f1' # acc_all, f1
        for epoch in range(num_epoch):
            t0 = time.time()
            # alpha = 0.01, Beta=0. [best]
            loss_clf, loss_contrast = train_gnn(g, x, y, model, optimizer, train_idx, weight_node=None, Alpha=0.01, Beta=0., local_graph_loader=local_graph_loader, epoch=epoch)
            history_loss_train.append(loss_clf.item())
            t1 = time.time()
            print('[Train:{}] loss:{:.3f}, loss_contrast:{:.3f}, training time:{}'.format(epoch, loss_clf, loss_contrast, t1 - t0))

            res_bucket = test_gnn(g, x, y, model, train_val_test_idx, local_graph_loader=local_graph_loader)
            # print('[Train:{}] loss_clf:{}, loss_contrast:{},  eval: {}'.format(epoch, loss_clf, loss_contrast, res_bucket['train']))
            print('Val[{}]: {}'.format(epoch, res_bucket['val']))
            print('Test[{}]: {}, test time:{}'.format(epoch, res_bucket['test'], time.time() - t1))
            if res_bucket['val'][choose_by] > best_score['val_score'] and epoch >= 10:
                best_score['val_score'] = res_bucket['val'][choose_by]
                best_score['test_res'] = res_bucket['test']
                best_score['val_res'] = res_bucket['val']
                torch.save(model.state_dict(), 'ckpt/best_model.ckpt')
            for key in history_val:
                history_train[key].append(res_bucket['train'][key])
                history_val[key].append(res_bucket['val'][key])
        print('Final val score:', best_score['val_res'])
        print('Final test score:', best_score['test_res'])
        test_scores_bucket.append(best_score['test_res'])
print('Average Test Scores:')
for key in test_scores_bucket[0]:
    avg_score = 0
    list_score = []
    for idx in range(len(test_scores_bucket)):
        avg_score += test_scores_bucket[idx][key]
        list_score.append(test_scores_bucket[idx][key])
    print(key, avg_score / len(test_scores_bucket))
    print('[{}] score_history:'.format(key), list_score)

