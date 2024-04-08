import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(torch.zeros(size=(2 * out_features, out_features)))
        nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.mm(inp, self.W)
        N = h.size()[0]
        Wh1 = torch.mm(h, self.a[:self.out_features, :])
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        attention = F.softmax(attention, dim=1)
        negative_attention = -F.softmax(negative_attention,dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        negative_attention = F.dropout(negative_attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, inp)
        h_prime_negative = torch.matmul(negative_attention, inp)
        h_prime_double = torch.cat([h_prime,h_prime_negative],dim=1)
        new_h_prime = torch.mm(h_prime_double,self.wtrans)

        if self.concat:
            return F.elu(new_h_prime)
        else:
            return new_h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Signed_GAT(nn.Module):
    def __init__(self,node_embedding,cosmatrix,nfeat, uV, original_adj, hidden = 16, \
                                            nb_heads = 4, n_output = 300, dropout = 0, alpha = 0.3):
        super(Signed_GAT, self).__init__()
        self.dropout = dropout
        self.uV = uV
        embedding_dim = 300
        self.user_tweet_embedding = nn.Embedding(num_embeddings=self.uV, embedding_dim=embedding_dim, padding_idx=0)
        self.user_tweet_embedding.from_pretrained(torch.from_numpy(node_embedding))
        # self.original_adj = torch.from_numpy(original_adj.astype(np.float64)).cuda()
        self.original_adj = torch.from_numpy(original_adj.astype(np.float64))
        # self.potentinal_adj = torch.where(cosmatrix>0.5,torch.ones_like(cosmatrix),torch.zeros_like(cosmatrix)).cuda()
        self.potentinal_adj = torch.where(cosmatrix > 0.5, torch.ones_like(cosmatrix), torch.zeros_like(cosmatrix))
        self.adj = self.original_adj + self.potentinal_adj
        self.adj = torch.where(self.adj>0,torch.ones_like(self.adj),torch.zeros_like(self.adj))
        self.attentions = [GraphAttentionLayer(nfeat, n_output, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nb_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nfeat * nb_heads, n_output, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, X_tid):
        # X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda()).to(torch.float32)
        # X = self.user_tweet_embedding(torch.arange(0, self.uV).long()).to(torch.float32)
        X = self.user_tweet_embedding(torch.arange(0, self.uV).long().cuda()).to(torch.float32)
        x = F.dropout(X, self.dropout, training=self.training)
        adj = self.adj.cuda().to(torch.float32)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(self.out_att(x, adj))
        return x[X_tid]