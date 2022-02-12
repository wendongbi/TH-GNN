import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./models')
from TribeEncoder import GraphEncoder
from criterions import NCESoftmaxLoss, NCESoftmaxLossNS
class JumpingKnowledge(nn.Module):
    def __init__(self, mode='cat', in_feats=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        assert mode in ['cat', 'max', 'lstm'], \
            "Expect mode to be 'cat', or 'max' or 'lstm', got {}".format(mode)
        self.mode = mode

        if mode == 'lstm':
            assert in_feats is not None, 'in_feats is required for lstm mode'
            assert num_layers is not None, 'num_layers is required for lstm mode'
            hidden_size = (num_layers * in_feats) // 2
            self.lstm = nn.LSTM(in_feats, hidden_size, bidirectional=True, batch_first=True)
            self.att = nn.Linear(2 * hidden_size, 1)

    def reset_parameters(self):

        if self.mode == 'lstm':
            self.lstm.reset_parameters()
            self.att.reset_parameters()

    def forward(self, feat_list):

        if self.mode == 'cat':
            return torch.cat(feat_list, dim=-1)
        elif self.mode == 'max':
            return torch.stack(feat_list, dim=-1).max(dim=-1)[0]
        else:
            # LSTM
            stacked_feat_list = torch.stack(feat_list, dim=1) # (N, num_layers, in_feats)
            alpha, _ = self.lstm(stacked_feat_list)
            alpha = self.att(alpha).squeeze(-1)            # (N, num_layers)
            alpha = torch.softmax(alpha, dim=-1)
            return (stacked_feat_list * alpha.unsqueeze(-1)).sum(dim=1)

class AttnFusioner(nn.Module):
    def __init__(self, input_num=2, in_size=64, hidden=64):
        super(AttnFusioner, self).__init__()
        self.encoder_q = nn.Linear(in_size * input_num, hidden, bias=False)
        self.encoder_k = nn.Linear(in_size, hidden, bias=False)
        self.w_att = nn.Parameter(torch.FloatTensor(2 * hidden, hidden))
        self.input_num = input_num
        self.in_size = in_size
        self.hidden = hidden
        self.reset_parameters()
    
    def reset_parameters(self):
        self.encoder_q.reset_parameters()
        self.encoder_k.reset_parameters()
        nn.init.xavier_uniform_(self.w_att.data, gain=1.414)
    
    def forward(self, input_list):
        assert len(input_list) == self.input_num
        q = self.encoder_q(torch.cat(input_list, dim=1)) # N, hidden
        q = q.repeat(self.input_num, 1) # input_num*N, hidden
        k_list = []
        for i in range(len(input_list)):
            k_list.append(self.encoder_k(input_list[i])) # N, hidden
        k = torch.cat(k_list, dim=0) # input_num*N, hidden
        attn_input = torch.cat([q, k], dim=1)
        attn_input = F.dropout(attn_input, 0.5, training=self.training)
        # print(self.training)
        e = F.elu(torch.matmul(attn_input, self.w_att)) # N*input_num, 1
        attention = F.softmax(e.view(self.input_num, -1, self.hidden).transpose(0, 1), dim=1) # N, input_num, hidden
        out = torch.stack(input_list, dim=1).mul(attention) # N, input_num, hidden
        out = torch.cat([out[:, i, :] for i in range(self.input_num)], dim=-1) # N, input_num * hidden
        # out = torch.cat(input_list, dim=1).mul(attention)
        return out


class THGNN(nn.Module):
    def __init__(self, in_size, out_size, hidden, mode='cat',\
         num_layer=2, N=4040, gamma=0.1, use_memory=True, \
             use_jk = True, use_attnFusioner = False,  norm=False,
             ):
        super(THGNN, self).__init__()
        self.adj_cache = None
        self.use_jk = use_jk
        self.norm = norm
        self.tribe_encoder = GraphEncoder(
            positional_embedding_size=32,
            max_node_freq=16,
            max_edge_freq=16,
            max_degree=512,
            freq_embedding_size=16,
            degree_embedding_size=16,
            output_dim=hidden,
            node_hidden_dim=64,
            edge_hidden_dim=64,
            num_layers=5,
            num_step_set2set=6,
            num_layer_set2set=3,
            norm=True,
            gnn_model='gin',
            degree_input=True,
            shortest_path_input=True,
            max_path_length=6,
            sp_embedding_size=16,
            directional_subgraph=True,
            ntype_embedding_size=32,
            ntype_num = 2,
        )
        self.criterion = NCESoftmaxLossNS()
        self.nce_t = 0.07
        self.gamma = gamma
        self.hidden = hidden
        self.mode = mode
        self.input_layer = nn.Linear(in_size, hidden)
        # attention mode 2
        self.attn_1 = nn.Linear(hidden * 2, hidden, bias=False)
        self.attn_2 = nn.Linear(hidden * 2, hidden, bias=False)

        self.use_attnFusioner = use_attnFusioner

        if use_attnFusioner:
            self.fusion_layer = AttnFusioner(input_num=2, in_size=hidden, hidden=64)
        else:
            self.fusion_layer = nn.Linear(hidden + hidden, hidden * 2)
        self.out_layer = nn.Linear(2 * hidden, out_size)
        if self.use_jk:
            self.hidden_layer = nn.Linear(hidden * 2, hidden * 2)
            if self.mode == 'lstm':
                self.jump = JumpingKnowledge(mode, hidden * 2, num_layer)
            else:
                self.jump = JumpingKnowledge(mode)

            if mode == 'cat':
                self.out_layer = nn.Linear(2 * hidden * (num_layer + 1), out_size)
            else:
                self.out_layer = nn.Linear(2 * hidden, out_size)
        
        self.use_memory = use_memory
        if use_memory:
            self.memory = torch.zeros((N, out_size), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        self.input_layer.reset_parameters()
        self.fusion_layer.reset_parameters()
        self.out_layer.reset_parameters()
        if self.use_jk:
            # for model using jumpy knowledge
            self.hidden_layer.reset_parameters()
            self.jump.reset_parameters()
        self.attn_1.reset_parameters()
        self.attn_2.reset_parameters()

    def update_memory(self, logits_detach, gamma=0.1):
        # logits need to detach from the computation graph
        assert logits_detach.requires_grad == False
        if self.memory.device != logits_detach.device:
            self.memory = self.memory.to(logits_detach.device)
        self.memory = (1 - gamma) * self.memory + gamma * logits_detach

    def get_memory_similarity(self, logits_detach):
        # logits should not detach from the computation graph
        assert logits_detach.requires_grad == False
        return torch.mm(logits_detach, self.memory.T())
    

    def forward(self, g, x, local_graphs_loader=None):
        # generate graph embedding for local graphs
        emb_list = []
        for idx, batch in enumerate(local_graphs_loader):
            assert idx == 0 # full batch
            graph_q, graph_k, graph_idx = batch
            graph_q = graph_q.to(x.device)
            graph_k = graph_k.to(x.device)
            bsz = graph_q.batch_size
            feat_q = self.tribe_encoder(graph_q)
            feat_k = self.tribe_encoder(graph_k)
            emb = (feat_q + feat_k) / 2
            emb_list.append(emb)
            contrast_loss = self.criterion(torch.matmul(feat_k, feat_q.t()) / self.nce_t)
            # prob = out[range(graph_q.batch_size), range(graph_q.batch_size)].mean()
            assert feat_q.shape == (graph_q.batch_size, self.hidden)
        emb_list = torch.cat(emb_list, dim=0)
        
        hidden_list = []
        x = self.input_layer(x)

        if self.norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-5)
        a1 = self.attn_1(torch.cat([x, emb_list], dim=1))
        a2 = self.attn_2(torch.cat([x, emb_list], dim=1))
        a = torch.stack([a1, a2], dim=-1)
        a = F.leaky_relu(a, negative_slope=0.2)
        a = F.softmax(a, dim=-1)
        x *= a[:, :, 0]
        emb_list *= a[:, :, 1]
        # print(x.mean(0).mean(0).detach(), emb_list.mean(0).mean(0).detach(), a.mean(0).mean(0).detach())
        if self.use_attnFusioner:
            x = self.fusion_layer([x, emb_list])
        else:
            x = self.fusion_layer(torch.cat([x, emb_list], dim=1))

        # x = emb_list + x
        # x = self.fusion_layer(torch.stack((x, emb_list), dim=1))

        hidden_list.append(x) # h_0, N * 2d
        if self.adj_cache is None:
            self.adj_cache = g.adj().to(x.device)
            degs = g.in_degrees().float().clamp(min=1)
            # norm = torch.pow(degs, -0.5)
            norm = 1.0 / degs
            shp = norm.shape + (1,) * (x.dim() - 1)
            norm = torch.reshape(norm, shp).repeat(1, g.num_nodes())
            self.adj_cache = (self.adj_cache.to_dense() * norm).to_sparse()
        x = torch.sparse.mm(self.adj_cache, x)
        x = F.relu(F.dropout(x, p=0.5, training=self.training))
        
        if self.use_jk:
            # use jumpknowledge
            hidden_list.append(x) # h_1, N * 2d
            x = self.hidden_layer(x)
            x = torch.sparse.mm(self.adj_cache, x)
            hidden_list.append(x) # h_2, N*2d
            x = self.jump(hidden_list)
            logits = self.out_layer(x)
        else:
            x = self.out_layer(x)
            logits = torch.sparse.mm(self.adj_cache, x)

        probs = torch.sigmoid(logits)
        if self.use_memory:
            mem = torch.cat([1-probs.detach(), probs.detach()], dim=-1)
            self.update_memory(mem, gamma=self.gamma)
            # self.update_memory(logits.detach(), gamma=self.gamma)
            # sim_memory = self.get_memory_similarity(x.detach())
        
        return probs, contrast_loss, logits

    def get_emb(self, g, x, local_graphs_loader=None):
        # generate graph embedding for local graphs
        emb_list = []
        for idx, batch in enumerate(local_graphs_loader):
            assert idx == 0 # full batch
            graph_q, graph_k, graph_idx = batch
            graph_q = graph_q.to(x.device)
            graph_k = graph_k.to(x.device)
            bsz = graph_q.batch_size
            feat_q = self.tribe_encoder(graph_q)
            feat_k = self.tribe_encoder(graph_k)
            emb = (feat_q + feat_k) / 2
            emb_list.append(emb)
            contrast_loss = self.criterion(torch.matmul(feat_k, feat_q.t()) / self.nce_t)
            # prob = out[range(graph_q.batch_size), range(graph_q.batch_size)].mean()
            assert feat_q.shape == (graph_q.batch_size, self.hidden)
        emb_list = torch.cat(emb_list, dim=0)
        
        hidden_list = []
        x = self.input_layer(x)
        
        if self.use_attnFusioner:
            x = self.fusion_layer([x, emb_list])
        else:
            x = self.fusion_layer(torch.cat([x, emb_list], dim=1))

        # x = emb_list + x
        # x = self.fusion_layer(torch.stack((x, emb_list), dim=1))

        hidden_list.append(x) # h_0, N * 2d
        if self.adj_cache is None:
            self.adj_cache = g.adj().to(x.device)
            degs = g.in_degrees().float().clamp(min=1)
            # norm = torch.pow(degs, -0.5)
            norm = 1.0 / degs
            shp = norm.shape + (1,) * (x.dim() - 1)
            norm = torch.reshape(norm, shp).repeat(1, g.num_nodes())
            self.adj_cache = (self.adj_cache.to_dense() * norm).to_sparse()
        x = torch.sparse.mm(self.adj_cache, x)
        x = F.relu(F.dropout(x, p=0.5, training=self.training))
        
        if self.use_jk:
            # use jumpknowledge
            hidden_list.append(x) # h_1, N * 2d
            x = self.hidden_layer(x)
            x = torch.sparse.mm(self.adj_cache, x)
            hidden_list.append(x) # h_2, N*2d
            x = self.jump(hidden_list)
        return x