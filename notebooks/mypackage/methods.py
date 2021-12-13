import scipy.stats
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def barPlot_func_onedata(values, varLabels, plot_name):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])

    ax.set_ylabel('Scores')
    ax.set_title(plot_name)
    r = np.arange(len(varLabels))
    width = 0.75


    ax.bar(r, values, color = 'b',
        width = width,
        )

    plt.xticks(r + width/2, varLabels)
    plt.xticks(rotation=90)

    plt.xlabel('Variables')
    plt.ylabel('Dispersion Ratio')
    plt.show()



def pearson(X, Y, labels):
    pearson =[]
    for (columnName, columnData) in X.iteritems():
        pearson.append(scipy.stats.pearsonr(columnData, Y)[0])

    pearson = NormalizeData(pearson)
    barPlot_func_onedata(pearson, labels, "Pearson Correlation")
