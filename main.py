from email.policy import default
from matplotlib import use
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
from utils import  worker_init_fn, batcher, train_gnn, test_gnn, set_random_seed, build_dataset
from sklearn.model_selection import train_test_split

def main(args):
    # load datasets
    data_mat, label_mat, local_graph_loader, g = build_dataset(
        path_X=args.path_x,\
        path_Y=args.path_y,\
        path_tribes=args.path_tribe_files,
        path_tribes_order=args.path_tribe_order)

    # Training
    x = torch.from_numpy(data_mat).float()
    print(x.shape, g)
    class_idx_bucket = {}
    test_scores_bucket = []
    # model_seed = 1
    for data_seed in range(3):
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
        GPU_ID = args.gpu
        device = torch.device('cuda:{}'.format(GPU_ID)) if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')
        print('device', device)
        print('begin training model...')

        for model_seed in range(10):
            print('train model with data_seed:{} and model_seed:{}'.format(data_seed, model_seed))
            set_random_seed(model_seed)

            use_attnFusioner = False
            if args.fusion_mode == 'attention':
                use_attnFusioner = True
            model = THGNN(in_size=x.shape[1], out_size=1, hidden=args.hidden, norm=False,\
                 use_attnFusioner=use_attnFusioner, tribe_encoder_type=args.tribe_gnn,\
                      local_layer_num=args.local_num_layer)
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                'auc': []
            }
            history_val = {
                'loss': [],
                'acc_all': [],
                'f1': [],
                'precision': [],
                'recall': [],
                'auc': []
            }
            history_loss_train = []
            best_score = {
                'val_score': 0,
                'test_res': None,
                'val_res': None
            }
            choose_by = 'f1' # acc_all, f1
            for epoch in range(args.num_epoch):
                t0 = time.time()
                # alpha = 0.01, Beta=0. [best]
                loss_clf, loss_contrast = train_gnn(g, x, y, model, optimizer, train_idx, weight_node=None, Alpha=args.alpha, local_graph_loader=local_graph_loader, epoch=epoch)
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



if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Training scripts for TH-GNN')
    ap.add_argument('--gpu', type=int, default=0, help='GPU ID to train the model.')
    ap.add_argument('--local_num_layer', type=int, default=2, help='layer num of the tribe_graph_encoding module (a GNN) in TH-GNN')
    ap.add_argument('--global_num_layer', type=int, default=2, help='layer num of the global GNN in TH-GNN')
    ap.add_argument('--hidden', type=int, default=64, help='hidden unit num of the global GNN in TH-GNN')
    ap.add_argument('--num_epoch', type=int, default=100, help='training epoch num')
    ap.add_argument('-path_x', type=str, default='./financial_statement.npy', help='node attribute files (.npy, financial statements for each listed company)')
    ap.add_argument('--path_y', type=str, default='./risk_label.npy', help='node classification label file path, risk or not.')
    ap.add_argument('--path_tribe_files', type=str, default='/home/biwendong/code/data_financial/graph_files/listed_graph_raw')
    ap.add_argument('--path_tribe_order', type=str, default='/home/biwendong/code/data_financial/graph_files/order_listed_graph_files_by_node_id.json')
    ap.add_argument('--tribe_encoder_gnn', type=str, default='gin')
    ap.add_argument('--tribe_gnn', type=str, default='gin', choices=['gin'], help='gnn model used in tribe encoder.')
    ap.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate for the optimizer.')
    ap.add_argument('--alpha', type=float, default=0.01, help='weight for the contrastive loss term.')
    ap.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay for the optimizer.')
    ap.add_argument('--fusion_mode', type=str, default='mlp', choices=['mlp', 'attention'], help='mode used to fuse the information from node attributes and tribe structure.')
    args = ap.parse_args()
    print(args)
    main(args)

