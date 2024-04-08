import torch
import torch.nn as nn
import abc
import torch.nn.utils as utils
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.init as init
from config import config
from models.attention import Attention
from models.gat import Signed_GAT
from models.resnet import Resnet50
from models.clip2 import CLIP
from models.load_clip import load
from models.cross_model_fusion import CrossModule4Batch
#from src.transformers import logging, BertConfig, BertModel
from models.layer import AmbiguityLearning, MultiMessageFusion

from transformers import BertTokenizer,BertModel,BertConfig

import tsne

import numpy as np

class PGD(object):

    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.best_acc = 0
        self.init_clip_max_norm = None
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:0')

    @abc.abstractmethod
    def forward(self):
        pass

    def mfan(self, x_tid, x_text, y, loss, i, total, params, pgd_word):
        self.optimizer.zero_grad()
        logit_original, dist_og, attention_score, skl_score, e_t, e_v,_ = self.forward(x_tid, x_text)
        loss_classification = loss(logit_original, y)
        loss_mse = nn.MSELoss()
        loss_dis = loss_mse(dist_og[0], dist_og[1])
        loss_func_skl = torch.nn.KLDivLoss(reduction='batchmean')
        loss_kl = loss_func_skl(attention_score, skl_score)
        loss_defense = 1.8 * loss_classification + 2.4 * loss_dis + 1.0 * loss_kl
        loss_defense.backward()

        K = 3
        v=[]
        pgd_word.backup_grad()
        for t in range(K):
            pgd_word.attack(is_first_attack=(t == 0))
            if t != K - 1:
                self.zero_grad()
            else:
                pgd_word.restore_grad()
            loss_adv, dist, attention_score, skl_score, e_t, e_v ,vec= self.forward(x_tid, x_text)
            if t==0:
                v.append(vec.data.cpu().numpy().tolist())
            loss_adv = loss(loss_adv, y)
            loss_adv.backward()
        pgd_word.restore()

        self.optimizer.step()
        corrects = (torch.max(logit_original, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print(
            'Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                           loss_defense.item(),
                                                                           accuracy,
                                                                           corrects,
                                                                           y.size(0)))
        return v

    def fit(self, X_train_tid, X_train, y_train,
            X_dev_tid, X_dev, y_dev):

        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, weight_decay=0)

        X_train_tid = torch.LongTensor(X_train_tid)
        X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)

        dataset = TensorDataset(X_train_tid, X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True)
        loss = nn.CrossEntropyLoss()
        params = [(name, param) for name, param in self.named_parameters()]
        pgd_word = PGD(self, emb_name='word_embedding', epsilon=6, alpha=1.8)



        for epoch in range(self.config['epochs']):
            v = []
            y = []
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])
            self.train()
            avg_loss = 0
            avg_acc = 0
            for i, data in enumerate(dataloader):
                total = len(dataloader)
                # batch_x_tid, batch_x_text, batch_y = (item.cuda(device=self.device) for item in data)
                batch_x_tid, batch_x_text, batch_y = (item.to(self.device) for item in data)
                vec=self.mfan(batch_x_tid, batch_x_text, batch_y, loss, i, total, params, pgd_word)
                v.append(vec)
                y+=batch_y.cpu().numpy().tolist()


                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
            v=np.array(v)
            v=v.reshape(1408,300)
            tsne.tsne(v, y, epoch,"train")
            self.evaluate(X_dev_tid, X_dev, y_dev,epoch)


    def evaluate(self, X_dev_tid, X_dev, y_dev,epoch):
        y_pred,v = self.predict(X_dev_tid, X_dev)

        tsne.tsne(v,y_pred,epoch,"dev")
        acc = accuracy_score(y_dev[:len(y_pred)], y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev[:len(y_pred)], y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("saved model at ", self.config['save_path'])

    def predict(self, X_test_tid, X_test):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid)
        X_test = torch.LongTensor(X_test)
        # X_test_tid = torch.LongTensor(X_test_tid)
        # X_test = torch.LongTensor(X_test)

        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=64,drop_last=True)
        v=[]
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.to(self.device) for item in data)
                logits, grad, dist,_, _,_,vec= self.forward(batch_x_tid, batch_x_text)
                vec=vec.reshape(192,100).data.cpu().numpy().tolist()
                v.append(vec)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred,v[0]

    def predict_2(self, X_test_tid, X_test):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid)
        X_test = torch.LongTensor(X_test)
        # X_test_tid = torch.LongTensor(X_test_tid)
        # X_test = torch.LongTensor(X_test)

        dataset = TensorDataset(X_test_tid, X_test)
        dataloader = DataLoader(dataset, batch_size=128, drop_last=True)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid, batch_x_text = (item.to(self.device) for item in data)
                logits, grad, dist, _, _, _ = self.forward(batch_x_tid, batch_x_text)

                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()

        return y_pred


class MyModel1(NeuralNetwork):

    def __init__(self, config, adj, original_adj):
        super(MyModel1, self).__init__()
        self.config = config
        self.uV = adj.shape[0]
        embedding_weights = config['embedding_weights']
        # V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']
        self.img_proj = nn.Linear(64, 224)
        self.text_proj = nn.Linear(50, 77)
        self.mh_attention = Attention(input_size=300, n_heads=8, attn_dropout=0)

        model_name = "models/bert"
        bert_config = BertConfig.from_pretrained(model_name, num_labels=3)
        bert_config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(model_name, config=bert_config)
        for param in self.bert.parameters():
            param.requires_grad = False

        # self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
        #                                    _weight=torch.from_numpy(embedding_weights))
        self.word_to_feature = nn.Linear(768, 300)
        self.clip, preprocess = load("ViT-B/32", "cpu")
        self.cross_model = CrossModule4Batch(text_in_dim=512, image_in_dim=512, corre_out_dim=512)
        self.e_f_to_e_g = nn.Linear(128, 300)
        self.cosmatrix = self.calculate_cos_matrix()
        self.gat_relation = Signed_GAT(node_embedding=config['node_embedding'], cosmatrix=self.cosmatrix, nfeat=300, \
                                       uV=self.uV, nb_heads=1,
                                       original_adj=original_adj, dropout=0)
        self.classifier_corre = nn.Sequential(
            nn.Linear(900, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(600, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(300, 2)
        )
        self.senet = MultiMessageFusion(300, 128, 24, 3)
        self.image_embedding = Resnet50(config)
        self.ambiguity_module = AmbiguityLearning()
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(1800, 900)
        self.fc4 = nn.Linear(900, 600)
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=config['num_classes'])
        self.alignfc_g = nn.Linear(in_features=300, out_features=300)
        self.alignfc_t = nn.Linear(in_features=300, out_features=300)
        self.init_weight()

    def calculate_cos_matrix(self):
        a, b = torch.from_numpy(self.config['node_embedding']), torch.from_numpy(self.config['node_embedding'].T)
        c = torch.mm(a, b)
        aa = torch.mul(a, a)
        bb = torch.mul(b, b)
        asum = torch.sqrt(torch.sum(aa, dim=1, keepdim=True))
        bsum = torch.sqrt(torch.sum(bb, dim=0, keepdim=True))
        norm = torch.mm(asum, bsum)
        res = torch.div(c, norm)
        return res

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)

    def forward(self, X_tid, X_text):
        # fix dim
        input_x = X_tid.view(-1, X_tid.size(0))
        input_x = self.img_proj(input_x.float())
        input_x = input_x.expand(1, 3, 224, 224) # [1, 512]
        input_text = torch.cat((X_text, X_text[:, :27]), dim=1)
        input_text = input_text.expand(64, 77) # [64, 512]
        # input_x = torch.cat((input_x[:, :, :, :32], input_x[:, :, :, :32], input_x[:, :, :, :32]), dim=1)
        # # input_x = torch.cat((input_x[:, :, :, 32], input_x[:, :, :, 32], input_x[:, :, :, 32]), dim=2)
        img_out = self.clip.encode_image(input_x)
        text_out = self.clip.encode_text(input_text)
        # img_out, text_out = self.clip(input_x, X_text);
        # img_out = img_out.expand(64, 512)
        core = self.cross_model(text_out, img_out) # core (64, 512)
        e_v = torch.matmul(img_out, core.t())
        e_t = torch.matmul(text_out, core.t())
        e_v = e_v.expand(64, 64)
        e_f = torch.cat((e_v, e_t), dim=1)
        e_f = self.e_f_to_e_g(e_f)
        text_embedding = self.bert(X_text)
        text_embedding = text_embedding["last_hidden_state"] # text_embedding (64, 50, 768)
        X_text = self.word_to_feature(text_embedding)
        # X_text = self.word_embedding(X_text) # X_text out (64, 50, 300) # X_text input (64, 50)
        X_text = (X_text + text_out[:50, :300]) / 2
        if self.config['user_self_attention'] == True:
            X_text = self.mh_attention(X_text, X_text, X_text)
        X_text = X_text.permute(0, 2, 1)
        rembedding = self.gat_relation.forward(X_tid)
        iembedding = self.image_embedding.forward(X_tid)
        iembedding = (iembedding + img_out[:, 300] + e_f) / 3
        conv_block = [rembedding]
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = torch.squeeze(pool)
            conv_block.append(pool)

        conv_feature = torch.cat(conv_block, dim=1)
        graph_feature, text_feature = conv_feature[:, :300], conv_feature[:, 300:]
        # graph_feature [64, 300], text_feature [64, 300]
        text_se, image_se, corre_se = text_feature.unsqueeze(-1), graph_feature.unsqueeze(-1), e_f.unsqueeze(-1)

        attention_score = self.senet(torch.cat([text_se, image_se, corre_se], -1))

        text_final = text_feature * attention_score[:, 0].unsqueeze(1)
        img_final = graph_feature * attention_score[:, 1].unsqueeze(1)
        corre_final = e_f * attention_score[:, 2].unsqueeze(1)
        final_corre = torch.cat([text_final, img_final, corre_final], 1)
        pre_label = self.classifier_corre(final_corre)

        skl = self.ambiguity_module(e_t, e_v)
        weight_uni = (1-skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1)
        skl_score = torch.cat([weight_uni, weight_uni, weight_corre], 1)
        dist = [text_final, img_final.view(64, -1, 300)]
        return pre_label, dist, attention_score, skl_score, e_t, e_v,text_feature
        # return pre_label, dist
