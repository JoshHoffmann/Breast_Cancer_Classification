import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


def PlotScatters(target:pd.Series, features):
    figure, ax = plt.subplots(nrows=6,ncols=5)
    ax = ax.flatten()

    for i, c in enumerate(features.columns):
        sn.scatterplot(x = features[c],y=target, alpha=0.7, edgecolor=None, ax = ax[i])
        ax[i].set_xlabel(c)
        ax[i].set_ylabel('Diagnosis')

    plt.subplots_adjust(hspace=1)

def PlotHists(features:pd.DataFrame):
    figure, ax = plt.subplots(nrows=6,ncols=5)
    ax = ax.flatten()

    for i, c in enumerate(features.columns):
        sn.histplot(x = features[c], ax = ax[i], bins=20)
        ax[i].set_xlabel(c)

    plt.subplots_adjust(hspace=1)

def PlotCorrelationMatrix(data:pd.DataFrame,title:str):
    plt.figure()
    corr = data.corr()
    sn.heatmap(corr)
    plt.title(title)