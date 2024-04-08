# pip install pandas
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import numpy as np


def plot_xy(x_values, label, title,  num,string):
    """绘图"""
    fig_path="models/tsne/"+string+"_"+str(num)
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    df['label'] = label
    fig =sns.scatterplot(x="x", y="y", hue="label", data=df)
    plt.title(title)
    plt.show()
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(fig_path, dpi=400)
    plt.close()


def tsne(x_value, y_value,num,str):
    x=np.array(x_value)
    y=np.array(y_value)
    print(x.shape)
    print(y.shape)
    print(y)

    # t-sne 降维
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit_transform(x)
    tsne = TSNE(n_components=2,perplexity=50,learning_rate=50)
    x_tsne = tsne.fit_transform(x,)
    plot_xy(kmeans, y, "t-sne",num,str)

