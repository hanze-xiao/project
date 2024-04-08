import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

# 定义异质图卷积层
class HeteroGraphConv(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroGraphConv, self).__init__()
        # 为每种边类型创建一个图卷积层
        self.layers = nn.ModuleDict({
            etype: GraphConv(in_size, out_size) for etype in etypes
        })

    def forward(self, g, h_dict):
        funcs = {}
        # 遍历所有边类型
        for srctype, etype, dsttype in g.canonical_etypes:
            # 计算每种边类型的卷积结果
            Wh = self.layersetype
            funcs[(srctype, etype, dsttype)] = (Wh, 'sum')
        # 将所有边类型的卷积结果相加
        g.multi_update_all(funcs, 'sum')
        return {ntype: g.nodes[ntype].data['h'] for ntype in g.ntypes}

# 定义异质图神经网络模型
class HeteroGraphSAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, etypes):
        super(HeteroGraphSAGE, self).__init__()
        # 定义两个异质图卷积层
        self.conv1 = HeteroGraphConv(in_size, hidden_size, etypes)
        self.conv2 = HeteroGraphConv(hidden_size, out_size, etypes)

    def forward(self, g, h_dict):
        # 计算第一层卷积结果并使用 ReLU 激活函数
        h_dict = self.conv1(g, h_dict)
        h_dict = {k: F.relu(h) for k, h in h_dict.items()}
        # 计算第二层卷积结果
        h_dict = self.conv2(g, h_dict)
        return h_dict