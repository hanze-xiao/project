
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.nn.pytorch.glob import SumPooling
import numpy as np
from sklearn.model_selection import KFold
import dgl
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
from dgl.dataloading import GraphDataLoader
import random
from sklearn.cluster import KMeans



drug_smiles=np.load(f"davis_mutilable/davis_kinase.csv_token.npy", allow_pickle=True)
protein_seq=np.load(f"davis_mutilable/davis_kinase.csv_protein.npy", allow_pickle=True)
print(drug_smiles.shape)
print(protein_seq.shape)

m2 = nn.Linear(2000, 500)
protein=m2(torch.tensor(protein_seq,dtype=torch.float32))
print(protein.shape)
input=np.concatenate((drug_smiles,protein.detach().numpy()),axis=1)
print(input.shape)

kmeans = KMeans(n_clusters=6)
kmeans.fit(input)
labels=kmeans.labels_
print(labels.shape)
print(labels)

np.save(f"davis_mutilable/davis_kinase.csv_kmeanstype.npy",labels)
