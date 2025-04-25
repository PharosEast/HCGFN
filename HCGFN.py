import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch_sparse import spmm
from config import *

class ContrastiveHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, device=torch.device('cuda')):
        super(ContrastiveHead, self).__init__()
        self.device = device
        layers = []
        in_dense = nn.Linear(in_dim, hidden_dim)
        layers.append(in_dense)
        layers.append(nn.ReLU())
        for n in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        out_dense = nn.Linear(hidden_dim, out_dim)
        layers.append(out_dense)
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

def cal_infonce_loss(h1, h2):
    cos_dist = torch.exp(torch.matmul(h1, h2.T) / 0.5)
    neg_sim = torch.sum(cos_dist, dim=1)
    pos_sim = torch.diag(cos_dist)
    return -torch.mean(torch.log(pos_sim / neg_sim))

def cal_label_contrastive_loss(h1, h2, intra_mask, inter_mask, tr_num):
    similarity = torch.exp(torch.matmul(h1, h2.T) / 0.5)
    pos_sim = torch.sum(similarity * intra_mask, dim=1)
    pos_sim = pos_sim / torch.sum(intra_mask, dim=1)
    neg_sim = (torch.sum(similarity * inter_mask, dim=1) + pos_sim) / (tr_num - 1)
    return -torch.mean(torch.log(pos_sim / neg_sim))

def get_intra_inter_mask(y):
    num_samples = y.shape[0]
    num_classes = torch.max(y) + 1
    intra_mask = torch.zeros((num_samples, num_samples))
    inter_mask = torch.ones((num_samples, num_samples))
    for class_idx in range(num_classes):
        class_indices = np.argwhere(y.cpu().numpy() == class_idx).flatten()
        for idx in class_indices:
            intra_mask[idx, class_indices] = 1
    inter_mask -= intra_mask
    return intra_mask, inter_mask

class HCGFN(nn.Module):
    def __init__(self, config: BiGATConfig, d_config: DHGSLMConfig, device=torch.device('cuda')):
        super(HCGFN, self).__init__()
        dropout = config.dropout
        alpha = config.alpha
        self.device = device
        self.dropout = config.dropout
        self.attentions_hsi = nn.ModuleList(
            [SparseGATLayer(config.in_dim_hsi, config.bi_dim, dropout=dropout, alpha=alpha, concat=True, device=device)
             for _ in range(config.n_head_hsi)])
        self.out_att_hsi = SparseGATLayer(config.bi_dim * config.n_head_hsi, config.bi_dim, dropout=dropout,
                                          alpha=alpha, concat=False, device=device)
        self.mid_out_hsi = SparseGATLayer(config.bi_dim * config.n_head_hsi, config.num_class, dropout=dropout, alpha=alpha, concat=False, device=device)

        self.attentions_lidar = nn.ModuleList([SparseGATLayer(config.in_dim_lidar, config.bi_dim, dropout=dropout,
                                                              alpha=alpha, concat=True, device=device)
                                               for _ in range(config.n_head_lidar)])
        self.out_att_lidar = SparseGATLayer(config.bi_dim * config.n_head_lidar, config.bi_dim, dropout=dropout,
                                            alpha=alpha, concat=False, device=device)
        self.mid_out_lidar = SparseGATLayer(config.bi_dim * config.n_head_lidar, config.num_class, dropout=dropout, alpha=alpha, concat=False, device=device)

        self.dhgslm = DHGSLM(config.bi_dim, d_config, device)
        self.gcn = GraphConvolution(config.bi_dim, config.num_class)
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))

        self.hsi_cl_head = ContrastiveHead(config.bi_dim, config.o_hidden_dim, config.o_hidden_dim, config.o_layer_num)
        self.lidar_cl_head = ContrastiveHead(config.bi_dim, config.o_hidden_dim, config.o_hidden_dim, config.o_layer_num)

    def forward(self, in_hsi, in_lidar, adj_hsi, adj_lidar, subgraph_adj, labels, idx_train):
        x_hsi = torch.cat([att(in_hsi, adj_hsi) for att in self.attentions_hsi], dim=1)
        x_hsi = F.dropout(x_hsi, self.dropout)
        hsi_mid_out = F.elu(self.mid_out_hsi(x_hsi, adj_hsi))
        hsi_mid_out = F.log_softmax(hsi_mid_out, dim=1)
        x_hsi = F.elu(self.out_att_hsi(x_hsi, adj_hsi))

        x_lidar = torch.cat([att(in_lidar, adj_lidar) for att in self.attentions_lidar], dim=1)
        x_lidar = F.dropout(x_lidar, self.dropout)
        lidar_mid_out = F.elu(self.mid_out_lidar(x_lidar, adj_lidar))
        lidar_mid_out = F.log_softmax(lidar_mid_out, dim=1)
        x_lidar = F.elu(self.out_att_lidar(x_lidar, adj_lidar))

        # CLM
        N = in_hsi.size()[0]
        # NCL
        hsi_cl = self.hsi_cl_head(x_hsi)
        lidar_cl = self.hsi_cl_head(x_lidar)
        subgraph_edge = subgraph_adj.indices()
        subgraph_value = subgraph_adj.values()
        sub_hsi = spmm(subgraph_edge, subgraph_value, n=N, m=N, matrix=hsi_cl)
        sub_lidar = spmm(subgraph_edge, subgraph_value, n=N, m=N, matrix=lidar_cl)
        subgraph_loss = (cal_infonce_loss(sub_hsi, sub_lidar) +
                         cal_infonce_loss(sub_lidar, sub_hsi)) / 2

        # LCL
        intra_mask, inter_mask = get_intra_inter_mask(labels[idx_train])
        intra_mask = intra_mask.to(self.device)
        inter_mask = inter_mask.to(self.device)
        sup_hsi = torch.gather(hsi_cl, index=idx_train, dim=0)
        sup_lidar = torch.gather(lidar_cl, index=idx_train, dim=0)
        sup_contrastive_loss = (cal_label_contrastive_loss(sup_hsi, sup_lidar, intra_mask, inter_mask,
                                                              idx_train.shape[0]) +
                                cal_label_contrastive_loss(sup_lidar, sup_hsi, intra_mask, inter_mask,
                                                              idx_train.shape[0])) / 2
        # get init het graph
        num_nodes = adj_hsi.size()[0]
        adj_bi = torch.zeros((num_nodes * 2, num_nodes * 2)).to(self.device)
        adj_bi[:num_nodes, :num_nodes] = adj_hsi.to_dense()
        adj_bi[num_nodes:, num_nodes:] = adj_lidar.to_dense()
        adj_bi[:num_nodes, num_nodes:] = torch.eye(num_nodes).to(self.device)
        adj_bi[num_nodes:, :num_nodes] = torch.eye(num_nodes).to(self.device)

        num_nodes = adj_hsi.size()[0]
        x_bi = torch.cat([x_hsi, x_lidar], dim=0)
        adj_bi = self.dhgslm(x_bi, adj_bi)
        x_bi = self.gcn(x_bi, adj_bi)
        return F.log_softmax(self.w1 * x_bi[:num_nodes] + self.w2 * x_bi[num_nodes:],
                             dim=1), hsi_mid_out, lidar_mid_out, subgraph_loss, sup_contrastive_loss

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        N = input.size()[0]
        edge = adj.nonzero().t()
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DHGSLM(nn.Module):
    def __init__(self, input_dim, config: DHGSLMConfig, device=torch.device('cuda')):
        super(DHGSLM, self).__init__()
        self.device = device
        self.num_nodes = config.num_nodes
        n_n = self.num_nodes
        self.node_types = ['H', 'L'] # hsi lidar
        self.edge_types = ['H-H', 'L-L', 'H-L']
        self.edge_position = {'H-H':(0, n_n, 0, n_n), 'L-L':(n_n, n_n * 2, n_n, n_n * 2), 'H-L':(n_n, n_n * 2, 0, n_n)} # r r c c
        self.non_linear = nn.ReLU()
        self.node_position = {'H': range(config.num_nodes), 'L': range(config.num_nodes, config.num_nodes * 2)}
        self.fsim_graph_builder, self.nsim_graph_builder1, self.nsim_graph_builder2, self.aggregate_het_graph_builder = nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleDict({}), nn.ModuleDict({})
        self.encoder = nn.ModuleDict(dict(zip(self.node_types, [nn.Linear(input_dim, config.HGSL_feature_dim) for _ in self.node_types])))

        for r in self.edge_types:
            self.fsim_graph_builder[r] = FeatureSimilarityGraphBuilder(config.HGSL_feature_dim, config.HGSL_head, config.HGSL_threshold_fg, self.device)
            self.nsim_graph_builder1[r] = FeatureSimilarityGraphBuilder(input_dim, config.HGSL_head, config.HGSL_threshold_fag, self.device)
            self.nsim_graph_builder2[r] = FeatureSimilarityGraphBuilder(input_dim, config.HGSL_head, config.HGSL_threshold_fag, self.device)
            self.aggregate_het_graph_builder[r] = GraphAggregateLayer(4)

        self.norm_order = 1

    def forward(self, features, adj_ori):
        all_feat_mat = torch.cat([self.non_linear(self.encoder[t](features[self.node_position[t]])) for t in self.node_types])
        new_adj = torch.zeros_like(adj_ori).to(self.device)
        for r in self.edge_types:
            init_adj = adj_ori[self.edge_position[r][0]:self.edge_position[r][1], self.edge_position[r][2]:self.edge_position[r][3]]
            fsim_adj = self.fsim_graph_builder[r](all_feat_mat[self.node_position[r[0]], :], all_feat_mat[self.node_position[r[-1]], :])
            features1, features2 = features[self.node_position[r[0]]], features[self.node_position[r[-1]]]
            sim_1, sim_2 = self.nsim_graph_builder1[r](features1, features1), self.nsim_graph_builder2[r](features2, features2)
            nsim_1, nsim_2 = sim_1.mm(init_adj), sim_2.mm(init_adj.t()).t()
            op_adj = self.aggregate_het_graph_builder[r]([fsim_adj, nsim_1, nsim_2, init_adj])
            new_adj[self.edge_position[r][0]:self.edge_position[r][1], self.edge_position[r][2]:self.edge_position[r][3]] = op_adj
        new_adj += new_adj.t().clone()
        new_adj = F.normalize(new_adj, dim=0, p=self.norm_order)
        return new_adj

class FeatureSimilarityGraphBuilder(nn.Module):
    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(FeatureSimilarityGraphBuilder, self).__init__()
        self.threshold = threshold
        self.num_head = num_head
        self.device = dev
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(1, dim))
            for _ in range(num_head)
        ])
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, feature1, feature2):
        if torch.sum(feature1) == 0 or torch.sum(feature2) == 0:
            return torch.zeros((feature1.shape[0], feature2.shape[0])).to(self.device)
        s = torch.zeros((feature1.shape[0], feature2.shape[0])).to(self.device)
        zero_lines = torch.nonzero(torch.sum(feature1, 1) == 0)
        if len(zero_lines) > 0:
            feature1[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            weighted_left_h = feature1 * self.weights[i]
            weighted_right_h = feature2 * self.weights[i]
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        return torch.where(s < self.threshold, torch.zeros_like(s), s)

class GraphAggregateLayer(nn.Module):
    def __init__(self, num_channel):
        super(GraphAggregateLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)
    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        adj_list = F.normalize(adj_list, dim=1, p=1)
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)

class SparseGATLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_dim, out_dim, dropout, alpha, concat=True, device=torch.device('cuda')):
        super(SparseGATLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = out_dim
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(input_dim, out_dim)))  # FxF'
        self.attn = nn.Parameter(torch.zeros(size=(1, 2 * out_dim)))  # 2F'
        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.device = device

    def forward(self, x, adj):
        '''
        :param x:   dense tensor. size: nodes*feature_dim
        :param adj:    parse tensor. size: nodes*nodes
        :return:  hidden features
        '''
        N = x.size()[0]
        edge = adj._indices()
        if x.is_sparse:
            h = torch.sparse.mm(x, self.W)
        else:
            h = torch.mm(x, self.W)
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        values = self.attn.mm(edge_h).squeeze()
        edge_e = self.leakyrelu(values)
        edge_e = torch.exp(edge_e)
        e_rowsum = spmm(edge, edge_e, m=N, n=N,matrix=torch.ones(size=(N, 1)).to(self.device))
        h_prime = spmm(edge, edge_e, n=N, m=N, matrix=h)
        h_prime = h_prime.div(e_rowsum + torch.Tensor([9e-15]).to(self.device))
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt