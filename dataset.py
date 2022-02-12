from builtins import NotImplementedError
import math
import operator

import dgl
import dgl.data
from graph_tool import Graph
# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import os, json
import copy
from tqdm import tqdm
from dgl.data import AmazonCoBuy, Coauthor
from networkx.algorithms.shortest_paths.generic import shortest_path_length
import networkx as nx
import utils

class GraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        super(GraphDataset).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        #  graphs = []
        graphs, _ = dgl.data.utils.load_graphs(
            "data_bin/dgl/lscc_graphs.bin", [0, 1, 2]
        )
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])

    def __len__(self):
        return self.length

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                    # self.graphs[graph_idx].out_degree(node_idx)
                    self.graphs[graph_idx].out_degrees()[node_idx]
                    * math.e
                    / (math.e - 1)
                    / self.restart_prob
                )
                + 0.5
            ),
        )
        traces = dgl.sampling.random_walk(
            self.graphs[graph_idx],
            nodes=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            length=max_nodes_per_seed,
        )

        graph_q = utils._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        graph_k = utils._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        return graph_q, graph_k, graph_idx

class ourGraphClassificationDataset(GraphDataset):
    def __init__(
        self,
        graph_file_dir='/home/biwendong/code/data_financial/graph_files/listed_graph_raw',
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
        entire_graph=True,
        order_graph_files_path=None,
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.entire_graph = entire_graph
        self.order_graph_files_path = order_graph_files_path
        assert positional_embedding_size > 1

        self.graphs, self.center_idxs, self.mapper_idx2name = self.prepare_graphs(raw_dir=graph_file_dir)

        self.length = len(self.graphs)
        self.total = self.length

    def _convert_idx(self, idx):
        graph_idx = idx
        node_idx = self.center_idxs[idx]
        return graph_idx, node_idx
    def _shortest_path_embedding(self, g, center_idx):
        g_networkx = g.to_networkx()
        sp_in_dict = shortest_path_length(g_networkx, target=center_idx)
        # assert len(sp_in_dict) == g.num_nodes()
        sp_in_vec = torch.ones(g.num_nodes()) * -1
        for key in sp_in_dict:
            sp_in_vec[key] = sp_in_dict[key]

        sp_out_dict = shortest_path_length(g_networkx, source=center_idx)
        # assert len(sp_out_dict) == g.num_nodes()
        sp_out_vec = torch.ones(g.num_nodes()) * -1
        for key in sp_out_dict:
            sp_out_vec[key] = sp_out_dict[key]
        return sp_in_vec.long(), sp_out_vec.long()

    def _create_dgl_graph(self, path):
        company_name = path.split('/')[-1].split('.json')[0]
        with open(path, 'r') as fin:
            info = json.load(fin)
        nodes = info['result']['nodes']
        relations = info['result']['relationships']
        node_lbl_map = {}
        for node in nodes:
            node_id = node['id']
            node_name = node['properties']['name']
            node_lbl_map[node_id] = node['labels'][0]
            if node_name == company_name:
                center_node_id = node_id
        node_id_map = {}
        edges = [[], []]
        for rel in relations:
            startNode = rel['startNode']
            endNode = rel['endNode']
            if startNode not in node_id_map:
                node_id_map[startNode] = len(node_id_map)
            if endNode not in node_id_map:
                node_id_map[endNode] = len(node_id_map)
            edges[0].append(node_id_map[startNode])
            edges[1].append(node_id_map[endNode])
        assert len(node_id_map) == len(node_lbl_map)
        # ntypes = torch.zeros((len(node_id_map), 2), dtype=torch.long) # 1-0-Company, 0-1-Human
        # for node_id in node_lbl_map:
        #     if node_lbl_map[node_id] == 'Company':
        #         ntypes[node_id_map[node_id]][0] = 1
        #     elif node_lbl_map[node_id] == 'Human': 
        #         ntypes[node_id_map[node_id]][1] = 1
        #     else:
        #         raise NotImplementedError('Unknown label name:{}'.format(node_lbl_map[node_id]))
        ntypes = torch.zeros(len(node_id_map), dtype=torch.long) # 0-Company, 1-Human
        for node_id in node_lbl_map:
            if node_lbl_map[node_id] == 'Company':
                ntypes[node_id_map[node_id]] = 0
            elif node_lbl_map[node_id] == 'Human': 
                ntypes[node_id_map[node_id]] = 1
            else:
                raise NotImplementedError('Unknown label name:{}'.format(node_lbl_map[node_id]))

        g = dgl.graph((torch.Tensor(edges[0]).long(), torch.Tensor(edges[1]).long()), num_nodes=len(node_id_map)) # directional graph
        # g.add_edges(edges[1], edges[0]) # convert to undirectional graph
        g.ndata['ntype'] = ntypes

        # g = dgl.DGLGraph()
        # src, dst = edges[0], edges[1]
        # num_nodes = len(node_id_map)
        # g.add_nodes(num_nodes)
        # g.add_edges(src, dst)
        # g.add_edges(dst, src)
        # g.ndata['ntype'] = ntypes
        # # g.readonly()
        return g, node_id_map[center_node_id]

    def prepare_graphs(self, raw_dir='/home/biwendong/code/data_financial/jsj', add_statistical_encoding_on_raw_graph=True):
        if self.order_graph_files_path is not None:
            with open(self.order_graph_files_path, 'r') as fin:
                f_list = json.load(fin)
            assert len(f_list) == 4040
        else:
            f_list = os.listdir(raw_dir)
        graphs = []
        center_idxs = []
        mapper_idx2name = {}
        for idx, f_name in tqdm(enumerate(f_list)):
            if idx > 10:
                # for debug
                graphs.append(g)
                center_idxs.append(center_idx)
                mapper_idx2name[idx] = f_name
                continue
            f_path = os.path.join(raw_dir, f_name)
            g, center_idx = self._create_dgl_graph(f_path)
            if add_statistical_encoding_on_raw_graph:
                g = utils._add_undirected_graph_positional_embedding(g, self.positional_embedding_size)
                g.ndata["seed"] = torch.zeros(g.number_of_nodes(), dtype=torch.long)
                g.ndata["seed"][center_idx] = 1
                g.ndata["sp_in_emb"], g.ndata["sp_out_emb"] = self._shortest_path_embedding(g, center_idx)

            graphs.append(g)
            center_idxs.append(center_idx)
            mapper_idx2name[idx] = f_name
        return graphs, center_idxs, mapper_idx2name

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)
        graph_q = self.graphs[graph_idx]
        graph_k = copy.deepcopy(graph_q)
        return graph_q, graph_k, graph_idx