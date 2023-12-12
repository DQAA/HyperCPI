from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax


class HAN(MessagePassing):
    r"""The hypergraph attention network

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          hyperedge indices :math:`(|\mathcal{V}|, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels, out_channels, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Linear(in_channels, out_channels, bias=False,
                            weight_initializer='glorot')
        self.lin2 = Linear(out_channels, out_channels, bias=False,
                            weight_initializer='glorot')
        self.att = Parameter(torch.Tensor(1, 1, 2 * out_channels))
        self.att2 = Parameter(torch.Tensor(1, 1, 2 * out_channels))
        self.a = Linear(out_channels, 1, bias=False,
                        weight_initializer='glorot')
        self.a2 = Linear(out_channels, 1, bias=False,
                        weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.lin2.reset_parameters()
        self.a.reset_parameters()
        self.a2.reset_parameters()
        glorot(self.att)
        glorot(self.att2)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (torch.Tensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        x = self.lin(x)

        u = x.view(-1, self.out_channels)
        u = F.leaky_relu(u)
        
        au = self.a(u)
        attn = softmax(au[hyperedge_index[0]], hyperedge_index[1])
            
        out = self.propagate(hyperedge_index, x=x, attn=attn, mode='v2e',
                             size=(num_nodes, num_edges))
        
        out = self.lin2(out)
        out = out.view(-1, self.out_channels)
        v = F.leaky_relu(out)
        
        va = self.a2(v)
        attn2 = softmax(va[hyperedge_index[1]], hyperedge_index[0])
        out = self.propagate(hyperedge_index.flip([0]), x=out, attn=attn2, mode='e2v',
                             size=(num_edges, num_nodes))

        out = out.view(-1, self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, attn: Tensor, mode) -> Tensor:
        F = self.out_channels

        if mode == 'v2e':
            out = attn.view(-1, 1, 1) * x_j.view(-1, 1, F)
        elif mode == 'e2v':
            out = attn.view(-1, 1, 1) * x_j.view(-1, 1, F)

        return out
