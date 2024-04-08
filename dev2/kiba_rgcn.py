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

txt_file="kiba_multilable\kiba.csv"

class Datasets(nn.Module):
    def __init__(self):
        drug_data = [
            np.load(f"{txt_file}_x.npy", allow_pickle=True),

            np.load(f"{txt_file}_a.npy", allow_pickle=True)
        ]
        self.drug_smiles=np.load(f"{txt_file}_token.npy", allow_pickle=True)
        self.drug_atom=drug_data[0]
        self.drug_adj=drug_data[1]
        self.drug_graph=[]
        for i in range(len(self.drug_smiles)):
               graph=dgl.from_scipy(sp.coo_matrix(self.drug_adj[i]))
               graph.ndata['feat']=torch.tensor(self.drug_atom[i])
               self.drug_graph.append(graph)

        self.protein=np.load(f"{txt_file}_protein.npy", allow_pickle=True)
        self.labels=np.load(f"{txt_file}_y.npy", allow_pickle=True)
        self.index=np.load(f"{txt_file}_index.npy", allow_pickle=True)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):

        return self.drug_graph[item],self.drug_smiles[item],self.protein[item],self.labels[item],self.index[item]


class RGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_rel):
        super().__init__()
        self.RGCNlayers=nn.ModuleList()

        num_layers=4
        for layer in range(num_layers-1):
            if layer==0:
                self.RGCNlayers.append(
                     RelGraphConv(input_dim,hidden_dim,num_rel)
                )
            else:
                self.RGCNlayers.append(
                    RelGraphConv(hidden_dim,hidden_dim,num_rel)
                )



        self.mtp=nn.Sequential(
            nn.Linear( 2000 ,  2048),

            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(2048),

            nn.Linear(2048,1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,1)

        )

    def forward(self, g, feat, etype, index, device):
        #f_unprocess = feat.tolist()
        for i, layer in enumerate(self.RGCNlayers):
            feat = layer(g, feat, etype)
            # feat=self.batch_norms[i](feat)
        # print(feat.shape)  #443 X 300
        # print('--------------------------------')
        index=index.permute(1,0)


        drug=feat[index[0]]

        protein=feat[index[1]]

        drug_protein=torch.cat((drug,protein),-1)


        # f = feat.tolist()  # 443 X  300
        # # print(len(f))  #443
        # # print('--------------------------------')
        # index = index.tolist()
        # drug_protein = []
        # for i, data in enumerate(index):
        #     drug = f[data[0]] #+ f_unprocess[data[0]]
        #     protein = f[data[1]] #+ f_unprocess[data[1]]
        #     drug_protein.append(drug + protein)
        # drug_protein = torch.tensor(drug_protein, dtype=torch.float32)

        dti_predict = self.mtp(drug_protein.to(device))
        # dti_predict = self.mtp(feat)
        return dti_predict,feat

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)
# 5层 过多 一般2-3 ，可学习参数E启用应该可以累加层数
class GIN_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=True)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        #预测最终结果
        self.mtp = nn.Sequential(  # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行
            nn.Linear(4120, 512),  # 线性连接层，输入通道数为101952=1062*96，输出通道数为1024

            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),  # 线性连接层，输入通道数为512，输出通道数为2
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
        )
        self.drop = nn.Dropout(0.1)
        #-----------------------------------------

        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets
        self.cnn_protein = nn.Sequential(  # 构建Sequential，属于特殊的module，类似于forward前向传播函数，同样的方式调用执行
            nn.Conv1d(1, 250, 4, bias=False),  # 卷积层，输入通道数为128，输出通道数为32，卷积核为[128,4]，包含在Sequential的子module，层层按顺序自动执行
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(250),
            nn.Conv1d(250, 128, 8, bias=False),  # 卷积层，输入通道数为32，输出通道数为64，卷积核为[32,8]，包含在Sequential的子module，层层按顺序自动执行
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 12, bias=True),  # 卷积层，输入通道数为64，输出通道数为96，卷积核为[64,12]，包含在Sequential的子module，层层按顺序自动执行
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 1, 24, bias=True),
            # nn.ReLU(inplace=True),
            #nn.MaxPool1d(2, 1)
        )

    def forward(self, g, h, drug, protein,feature_list,train_index,device):
        # list of hidden representation at each layer (including the input layer)
        feature=feature_list.tolist()
        index=train_index.tolist()
        durg_rgcn=[]
        protein_rgcn=[]
        for i, data in enumerate(index):

            drug2=feature[data[0]]
            protein2=feature[data[1]]
            durg_rgcn.append(drug2)
            protein_rgcn.append(protein2)
        rgcn1=torch.tensor(durg_rgcn,dtype=torch.float32)
        rgcn2=torch.tensor(protein_rgcn,dtype=torch.float32)

        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        protein=protein.unsqueeze(1)
        #print(protein.shape)
        #protein = protein.permute(1,0)

        protein_cnn = self.cnn_protein(protein)
        #print(protein_cnn.shape)
        # perform graph sum pooling over all nodes in each layer
        #整个图的表示，因为全部节点的特征作sum,使用药物的话就是整个药物的特征提取
        drug_rep=0
        for i, h in enumerate(hidden_rep):
            if i==0: continue
            pooled_h=self.pool(g,h)
            drug_rep+=self.drop(pooled_h)
        drug_rep=torch.cat((drug_rep,drug),-1)
        protein_cnn = protein_cnn.squeeze(1)

        dti_pre=torch.cat((drug_rep,protein_cnn),-1)
        dti_pre2=torch.cat((rgcn1,rgcn2),-1).to(device)

        dti_pre=torch.cat((dti_pre,dti_pre2),-1)
        # print("------------------------------------")
        #print(drug_rep.shape)  164
        #print(protein_cnn.shape)  778
        # print(dti_pre.shape)
        # print("------------------------------------")
        dti_pre=self.mtp(dti_pre)


        return dti_pre#,drug_rep,protein_cnn


def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i,data in enumerate(Y):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ / pair
    else:
        return 0





def evaluate(dataloader, device, model,index,feat):
    model.eval()
    total = 0
    num=0
    c_index=0
    for train_drug_graph,train_drug_smiles,train_protein,train_label,train_index in dataloader:


        train_drug_graph =train_drug_graph.to(device)
        train_feature=feat.to(device)
        gin_index=train_index.to(device)
        train_drug_smiles = train_drug_smiles.to(device)
        train_protein = train_protein.to(device)
        train_label = train_label.to(device)
        train_drug_atom = train_drug_graph.ndata.pop("feat")
        logits= model(train_drug_graph, train_drug_atom, train_drug_smiles, train_protein,train_feature,gin_index,device)
        logits = logits.cpu().detach().numpy()
        train_label=train_label.cpu()
        total+=mean_squared_error(logits,train_label)
        c_index += float(get_cindex(train_label, logits))
        num+=1

    return total/num,c_index/num


def evaluate_RGCN(graph,feat,etype,index,label, device, model):
    model.eval()

    graph = graph.to(device)
    feature = torch.tensor(np.array(feat),dtype=torch.float32).to(device)
    etype = torch.as_tensor(etype).to(device)
    index = torch.as_tensor(index,dtype=torch.long).to(device)
    logits,_ = model(graph, feature, etype, index, device)  # 返回各个节点特征   443维
    logits = logits.cpu().detach().numpy()

    total_loss = mean_squared_error(logits, label)
    c_index=get_cindex(label,logits)

    return total_loss,float(c_index)

def get_rgcn_feature(graph,feat,etype,index,device,model):
    model.eval()
    graph = graph.to(device)
    feature = torch.tensor(feat, dtype=torch.float32).to(device)
    etype = torch.as_tensor(etype).to(device)
    index = torch.as_tensor(index,dtype=torch.long).to(device)
    _,rgcn_feature= model(graph, feature, etype, index, device)



    return rgcn_feature


f=open("log_kiba_rgcn.txt", 'w')


dgl0 = np.load(f"{txt_file}_dgl0.npy",allow_pickle=True)
dgl1 = np.load(f"{txt_file}_dgl1.npy", allow_pickle=True)
etype = np.load(f"{txt_file}_edg_type.npy", allow_pickle=True)
index=np.load(f"{txt_file}_index.npy", allow_pickle=True)
adj=np.load(f"{txt_file}_adj.npy", allow_pickle=True)
#kmeans_etpye=np.load(f"{txt_file}_kmeanstype.npy", allow_pickle=True)

labels=np.load(f"{txt_file}_y.npy", allow_pickle=True)
all_data=Datasets()
all_num=all_data.__len__()
data_induce=np.arange(0,all_num)
kf=KFold(n_splits=5,shuffle=True,random_state=100)
data_loaders=dict()
j=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for train_index,val_index in kf.split(data_induce):
    print("----第",j,"折训练----")

    train_subset=torch.utils.data.Subset(all_data,train_index)
    val_subset=torch.utils.data.Subset(all_data,val_index)

    data_loaders['train']=GraphDataLoader(train_subset,batch_size=256,pin_memory=torch.cuda.is_available(),shuffle=True)

    data_loaders['val'] = GraphDataLoader(val_subset, batch_size=256,pin_memory=torch.cuda.is_available(),shuffle=True)

    data_loaders_RGCN = GraphDataLoader(all_data, pin_memory=torch.cuda.is_available())

    print('-------------graph training-------------')

    etype_mask = etype[train_index]
    g_val=dgl.graph((dgl0,dgl1))
    g=dgl.graph((dgl0[train_index],dgl1[train_index]))

    # drug_pre=[]
    # protein_pre=[]
    # t1=np.zeros((1,2000))
    # t2=np.zeros((1,100))
    # # print(t1.shape)
    # # print(t2.shape)
    # for _,train_drug_smiles,train_protein,_,_ in data_loaders_RGCN:
    #     # print(train_drug_smiles.shape)
    #     # print(train_protein.shape)
    #     drug_pre.append(np.concatenate((train_drug_smiles,t1),-1).squeeze())
    #     protein_pre.append(np.concatenate((t2,train_protein),-1).squeeze())

    np.random.seed(100)
    num=len(adj)
    feat=[list() for i in range(num)]
    all_pre=np.random.randn(all_num,2100)



    tra_label = labels[train_index]
    tra_index = index[train_index]
    val_label = labels[val_index]
    val_indice = index[val_index]



    # for i,data in enumerate(index):
    #     if len(feat[data[0]])==0:
    #         feat[data[0]]=drug_pre[i]
    #     if len(feat[data[1]])==0:
    #         feat[data[1]]=protein_pre[i]
    for i,data in enumerate(index):
        if len(feat[data[0]])==0:
            feat[data[0]]=all_pre[i]
        if len(feat[data[1]])==0:
            feat[data[1]]=all_pre[i]
    epoch_rgcn = 6000
    # feat=torch.tensor(feat,dtype=torch.float32)
    model_2 = RGCN(2100, 1000, 6).to(device)

    loss_model_2 = nn.MSELoss()
    optim_2 = torch.optim.Adam(model_2.parameters(), weight_decay=1e-7, lr=0.001)
    num_rgcn=0
    min_loss=1000

    for i in range(epoch_rgcn):
        model_2.train()

        graph=g.to(device)
        feature=torch.tensor(np.array(feat),dtype=torch.float32).to(device)
        etype_g=torch.as_tensor(etype_mask).to(device)
        label=torch.tensor(np.array(tra_label)).to(device)
        index_g=torch.as_tensor(np.array(tra_index),dtype=torch.long).to(device)
        logits,_=model_2(graph,feature,etype_g,index_g,device) # 返回各个节点特征   443维
        loss = loss_model_2(logits, label)
        optim_2.zero_grad()
        loss.backward()
        optim_2.step()
        num_rgcn+=1
        # if num_rgcn%20==0:
        #     f.write("RGCN LOSS: {:.4f}\n".format(loss.data.item()))
        #     val_acc, c_index = evaluate_RGCN(g, feat, etype_mask, val_indice, val_label, device, model_2)
        #     val_acc2, c_index2 = evaluate_RGCN(g_val, feat, etype, val_indice, val_label, device, model_2)
        #     f.write("Validation MSE AND CI. {:.4f} {:.4f}\n".format(val_acc,c_index))
        #     f.write("Validation INC MSE AND CI. {:.4f} {:.4f}\n".format(val_acc2, c_index2))
        #     print(
        #         " Validation MSE. {:.4f} .  ".format(
        #             val_acc
        #         ))
        #     print(c_index)
        #     print("--------------------")
        #     print(
        #         " Validation INC MSE. {:.4f} .  ".format(
        #             val_acc2
        #         ))
        #     print(c_index2)
        if num_rgcn>2000 and num_rgcn % 50==0:


            f.write("RGCN LOSS: {:.4f}\n".format(loss.data.item()))
            val_acc, c_index = evaluate_RGCN(g, feat, etype_mask, val_indice, val_label, device, model_2)
            # val_acc2, c_index2 = evaluate_RGCN(g_val, feat, etype, val_indice, val_label, device, model_2)
            f.write("Validation MSE AND CI. {:.4f} {:.4f}\n".format(val_acc, c_index))
            # f.write("Validation INC MSE AND CI. {:.4f} {:.4f}\n".format(val_acc2, c_index2))
            print(
                " Validation MSE. {:.4f} .  ".format(
                    val_acc
                ))
            print(c_index)
            # print("--------------------")
            # print(
            #     " Validation INC MSE. {:.4f} .  ".format(
            #         val_acc2
            #     ))
            # print(c_index2)
            if min_loss>val_acc:
                min_loss=val_acc
                torch.save(model_2,"./weight_kiba_5/"+str(j)+"_bestmodel.pkl")
            if loss.data.item()>0.1:
                break

        print(loss.data.item())




    val_acc,c_index=evaluate_RGCN(g,feat,etype_mask,val_indice,val_label,device,model_2)
    f.write( "RGCN Validation Acc. {:.4f} . \n {:.4f}\n".format(
            val_acc,float(c_index)
        ))


    print(
        " Validation Acc. {:.4f} .  ".format(
            val_acc
        ))
    print(c_index)




    # print('-------------get graph feature-------------')
    # best_model = torch.load("./weight_davis_rgcn/" + str(j) + "_bestmodel.pkl")
    # rgcn_feature=get_rgcn_feature(g,feat,etype_mask,tra_index,device,best_model)
    #
    # print('----------------GIN training----------------')
    # model_1 = GIN_CNN(34, 64).to(device)
    #
    # # 保存每折最优模型参数，再次训练获取各个表示
    # min_each_loss = 10000
    # loss_model = nn.MSELoss()
    # optim = torch.optim.Adam(model_1.parameters(), weight_decay=1e-6, lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=80, gamma=0.5)
    # num_print = 0
    #
    # epoch_getfeature=1
    # for i in range(epoch_getfeature):
    #     model_1.train()
    #     total_loss=0
    #     for  batch, (train_drug_graph,train_drug_smiles,train_protein,train_label,train_index1) in enumerate(data_loaders['train']):
    #        # train_drug_atom,train_drug_graph,train_drug_smiles,train_protein,train_label=train
    #         train_drug_graph=train_drug_graph.to(device)
    #         train_feature=rgcn_feature.to(device)
    #         gin_index=torch.as_tensor(train_index1).to(device)
    #         train_drug_smiles=train_drug_smiles.to(device)
    #         train_protein=train_protein.to(device)
    #         train_label=train_label.to(device)
    #         #print(train_drug_graph)
    #         train_drug_atom = train_drug_graph.ndata.pop("feat")
    #         logits=model_1(train_drug_graph,train_drug_atom,train_drug_smiles,train_protein,train_feature,gin_index,device)
    #         loss=loss_model(logits,train_label)
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
    #         total_loss += loss.item()
    #         print(loss.data.item())
    #     scheduler.step()
    #
    #     train_acc, ci_trian = evaluate(data_loaders['train'], device, model_1,tra_index,rgcn_feature)
    #     valid_acc, ci_valid = evaluate(data_loaders['val'], device, model_1,val_indice,rgcn_feature)
    #     print(
    #         "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Train CI. {:.4f} | Validation Acc. {:.4f} |  Validation CI. {:.4f}".format(
    #             i, total_loss / batch + 1, train_acc, ci_trian, valid_acc, ci_valid
    #         ))
    #
    #
    #
    #
    #     if num_print > 200:
    #         f.write(
    #             "Epoch {:05d} | Loss {:.4f} | Train Mse. {:.4f} | Train CI. {:.4f} | Validation Mse {:.4f} |  Validation CI. {:.4f}\n".format(
    #                 i, total_loss / batch + 1, train_acc, ci_trian, valid_acc, ci_valid))
    #         print(
    #             "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Train CI. {:.4f} | Validation Acc. {:.4f} |  Validation CI. {:.4f}".format(
    #                 i, total_loss / batch + 1, train_acc, ci_trian, valid_acc, ci_valid
    #             ))
    #     num_print+=1
    #
    #
    #
    #
    #
    #


    j = j + 1
f.close()















