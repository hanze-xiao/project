import torch

# processed_data_file_train = 'processed/' + "davis" + '_train.pt'
# processed_data_file_test = 'processed/' + "davis"+ '_test.pt'
# ar1,ar2=torch.load(processed_data_file_train)
# print(ar1)
# print(ar1.c_size)
# print(sum(ar1.c_size))
# print(ar2)
# print(ar2.keys())
# print(len(ar2['edge_index'].tolist()))

import pandas as pd

df1 = pd.read_csv('processed/kiba_test.csv')

df2 = pd.read_csv('processed/kiba_train.csv')

merged = pd.concat([df1, df2])

merged.to_csv('kiba/kiba.csv', index=False)