import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch.autograd import Variable
from torchvision.ops import box_iou


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, skip=True):
        super(GraphConvolution, self).__init__()
        self.skip = skip
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # TODO make fc more efficient via "pack_padded_sequence"
        # import ipdb; ipdb.set_trace()
        support = torch.bmm(input, self.weight.unsqueeze(
            0).expand(input.shape[0], -1, -1))
        output = torch.bmm(adj, support)
        #output = SparseMM(adj)(support)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand(input.shape[0], -1, -1)
        if self.skip:
            output += support

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN_sim(nn.Module):
    """
    Adapt from https://github.com/SunDoge/L-GCN/
    """
    def __init__(self, dim_in, dim_hidden, dim_out, dropout, num_layers):
        super(GCN_sim, self).__init__()
        assert num_layers >= 1
        self.fc_k = nn.Linear(dim_in, dim_hidden)
        self.fc_q = nn.Linear(dim_in, dim_hidden)

        dim_hidden = dim_out if num_layers == 1 else dim_hidden
        self.gcs = nn.ModuleList([
            GraphConvolution(dim_in, dim_hidden)
        ])

        for i in range(num_layers - 1):
            dim_tmp = dim_out if i == num_layers-2 else dim_hidden
            self.gcs.append(GraphConvolution(dim_hidden, dim_tmp))

        self.dropout = dropout

    def construct_graph(self, x, length):
        # TODO make fc more efficient via "pack_padded_sequence"
        emb_k = self.fc_k(x)
        emb_q = self.fc_q(x)

        s = torch.bmm(emb_k, emb_q.transpose(1, 2))

        s_mask = s.data.new(*s.size()).fill_(1).bool()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(length, length)):
            s_mask[i][:l_1, :l_2] = 0
        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data, -float("inf"))

        a_weight = F.softmax(s, dim=2)  # [B, t1, t2]
        # remove nan from softmax on -inf
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)

        return a_weight


    def forward(self, x, length):
        adj_sim = self.construct_graph(x, length)

        for gc in self.gcs:
            x = F.relu(gc(x, adj_sim))
            x = F.dropout(x, self.dropout, training=self.training)

        return x, adj_sim


class GCN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout, skip, num_layers):
        super(GCN, self).__init__()
        
        self.skip = skip      
        self.GCN_sim = GCN_sim(dim_in, dim_hidden, dim_out, dropout, num_layers)

    def forward(self, x, length, bboxes=None):

        out, adj_sim = self.GCN_sim(x, length)
        if self.skip:
            out += x

        return out, adj_sim


if __name__ == '__main__':
    model = GCN(512, 128, 512, 0.5, skip=True, num_layers=2)
    bs, T, N = 1, 5, 5
    n_node = T*N

    input = torch.rand(bs, n_node, 512)
    length = torch.LongTensor([n_node] * bs)
    output = model(input, length)
