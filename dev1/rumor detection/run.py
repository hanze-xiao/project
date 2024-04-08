import csv
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

import pickle
import json, os, time
import argparse
from config import config
import random
import sys
from models.mymodel import MyModel1

sys.path.append('/../image_part')
parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task",type=str, default="pheme")
parser.add_argument("-g", "--gpu_id",type=str, default="1")
parser.add_argument("-c", "--config_name",type=str, default="single3.json")
parser.add_argument("-T", "--thread_name",type=str, default="Thread-1")
parser.add_argument("-d", "--description",type=str, default="exp_description")
args = parser.parse_args()


def process_config(config):
    for k,v in config.items():
        config[k] = v[0]
    return config

def load_dataset():
    pre = 'dataset/pheme/pheme_files'
    X_train_tid, X_train, y_train, word_embeddings, adj = pickle.load(open(pre + "/train.pkl", 'rb'))
    X_dev_tid, X_dev, y_dev = pickle.load(open(pre + "/dev.pkl", 'rb'))
    X_test_tid, X_test, y_test = pickle.load(open(pre + "/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    config['node_embedding'] = pickle.load(open(pre + "/node_embedding.pkl", 'rb'))[0]
    print("#nodes: ", adj.shape[0])

    with open(pre + '/new_id_dic.json', 'r') as f:
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))
    content_path = 'dataset/pheme'
    with open(content_path + '/content.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        result = list(reader)[1:]
        mid2num = {}
        for line in result:
            mid2num[line[1]] = line[0]
    newid2num = {}
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    config['newid2imgnum'] = newid2num

    return X_train_tid, X_train, y_train, \
           X_dev_tid, X_dev, y_dev, \
           X_test_tid, X_test, y_test, adj


def load_original_adj(adj):
    pre = 'dataset/pheme/pheme_files/'
    path = os.path.join(pre, 'original_adj')
    with open(path, 'r') as f:
        original_adj_dict = json.load(f)
    original_adj = np.zeros(shape=adj.shape)
    for i, v in original_adj_dict.items():
        v = [int(e) for e in v]
        original_adj[int(i), v] = 1
    return original_adj


def train_and_test(model):
    model_suffix = model.__name__.lower().strip("text")
    res_dir = 'exp_result'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.task)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.description)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = config['save_path'] = os.path.join(res_dir, 'best_model_in_each_config')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix)
    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test, adj= load_dataset()
    original_adj = load_original_adj(adj)
    nn = model(config, adj, original_adj)

    nn.fit(X_train_tid, X_train, y_train,
           X_dev_tid, X_dev, y_dev)

    y_pred = nn.predict(X_test_tid, X_test)
    res = classification_report(y_test, y_pred, target_names=config['target_names'], digits=3, output_dict=True)
    for k, v in res.items():
        print(k, v)
    print("result:{:.4f}".format(res['accuracy']))
    res2={}
    res_final = {}
    res_final.update(res)
    res_final.update(res2)
    print(res)
    return res


config = process_config(config)
seed = config['seed']
torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
model = MyModel1
train_and_test(model)