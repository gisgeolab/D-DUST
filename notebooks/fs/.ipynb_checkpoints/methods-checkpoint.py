from datetime import datetime
from plotly.subplots import make_subplots
import scipy.stats
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import warnings
import borda.count

'''''
This file contain each method imported by the notebook in this repository. Each one of them is explained through docstring documentation.
If a method is marked with [GENERAL] it means that can be used generally for other cases of study
   '''''


def borda_voting(dataframe):
    """ [GENERAL]
    :param dataframe: Results where each column represent score obtained by each method
    :return: return a Series() containing the Borda Count score
    """
    labels = dataframe['Features']
    dataframe_new = dataframe.loc[:, dataframe.columns != 'Features']
    selection = borda.count.Election()
    selection.set_candidates(labels.tolist())
    for col in dataframe_new.columns:
        # Borda count
        method_ranking = getRanks(dataframe_new[col].tolist(), labels)
        voter = borda.count.Voter(selection, col)
        voter.votes(method_ranking)

    zipped = list(zip(list(selection.votes), list(selection.votes.values())))
    results = pd.DataFrame(data=zipped, columns=['Features', 'Scores'])
    results = results.set_index('Features')
    results = results.reindex(index=dataframe['Features'])
    results = results.reset_index()

    return results['Scores']


def getRanks(values, labels):
    """   [GENERAL]

    :param values:list of the scores of each label
    :param labels:list of the labels
    :return: the list of labels ordered by its score
    """
    zipped = list(zip(labels, values))

    data = pd.DataFrame(data=zipped, columns=['Features', 'Scores'])
    data.sort_values(by='Scores', axis=0, ascending=False, inplace=True, kind='quicksort')
    return data['Features'].tolist()


def process_data(data, k, sensor):
    """  [GENERAL]

    :param data: Dataset
    :param k: number of neighbours for knn
    :param sensor: name of the variable to increase its number of observation
    :return: Dataset with the observation of 'sensor' increased (and without not used and categorical variables)
    """
    if (k != 1):
        data = increase_data(data, sensor, k)
    # catagorical and not used variables
    data.pop('dusaf')
    data.pop('siarl')
    data.pop('top')
    data.pop('soil')
    data.pop('soil_text')
    data.pop('bottom')
    data.pop('right')
    data.pop('left')
    data.pop('area')
    data.pop('aq_zone')
    data.pop('wind_dir_st')
    data.pop(sensor[: -2] + 'int')

    return data


def increase_data(data, sensor, k):
    """  [GENERAL]

   :param data: Dataset
    :param k: number of neighbours for knn
    :param sensor: name of the variable to increase its number of observation
    :return: Dataset with the observation of 'sensor' increased
    """
    points_st = data[~data[sensor].isnull()]

    return add_buffer(points_st, data, data, k, sensor)


def add_buffer(points, data, uncleaned_data, k, sensor):
    """    [GENERAL]

    :param points: dataset where sensor variable is not null
    :param data: dataset
    :param uncleaned_data: dataset
    :param k: number of neighbours for knn
    :param sensor: name of the variable to increase its number of observation
    :return: Dataset with the observation of 'sensor' increased
    """
    warnings.filterwarnings("ignore")

    nA = np.array(list(points.geometry.centroid.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(data.geometry.centroid.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)

    for cell in nA:
        dist, idx = btree.query(cell, k)

        for i in range(0, k):
            uncleaned_data.at[idx[i], sensor] = uncleaned_data.loc[idx[i]][sensor[:-2] + 'int']
    return uncleaned_data


def NormalizeData1D(data):
    """   [GENERAL]

    :param data: Series() of a scores
    :return: Array of normalized score
    """
    data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    result = scaler.fit_transform(data).reshape((-1,))
    return result
def show_bars(labels_list, matrix, method, geopackages, order):
    """

    :param labels_list: list of lists the labels for each set of results
    :param matrix: List of lists of the results obtained in each period. A list represent the score obtained in in one period
    :param method: name of the method
    :param geopackages: list of periods
    :param order: It indicates how the scores are ordered
    """
    titles = []
    for g in geopackages:
        titles.append(g)
    fig = make_subplots(rows=int(len(geopackages) / 2) + 1, cols=2, subplot_titles=titles)
    for index, values in enumerate(matrix):
        labels = labels_list[index]
        # Decrescent order of the scores
        if (order == 'Scores'):
            zipped = list(zip(list(labels), list(values)))
            temp = pd.DataFrame(data=zipped, columns=['Features', 'Scores'])
            temp.sort_values(by='Scores', axis=0, ascending=False, inplace=True, kind='quicksort')
            values = temp['Scores'].to_numpy()
            labels = temp['Features']

        fig.add_trace(go.Bar(x=labels, y=values), row=int(index / 2) + 1, col=index % 2 + 1)
        fig.update_yaxes(row=int(index / 2) + 1, col=index % 2 + 1)
        fig.update_xaxes(type="category", row=int(index / 2) + 1, col=index % 2 + 1)

    fig.update_layout(height=1000, title_text=method)
    fig.update_layout(showlegend=False, autosize=True)
    fig.show()


def show_bars_log(labels_list, matrix, method, geopackages, order):
    """

     :param labels_list: list of lists the labels for each set of results
     :param matrix: List of lists of the results obtained in each period. A list represent the score obtained in in one period
     :param method: name of the method
     :param geopackages: list of periods
     :param order: It indicates how the scores are ordered
    """
    titles = []

    for g in geopackages:
        titles.append(g)
    fig = make_subplots(rows=int(len(geopackages) / 2) + 1, cols=2, subplot_titles=titles)
    for index, values in enumerate(matrix):
        labels = labels_list[index]
        # Decrescent order of the scores
        if (order == 'Scores'):
            zipped = list(zip(list(labels), list(values)))
            temp = pd.DataFrame(data=zipped, columns=['Features', 'Scores'])
            temp.sort_values(by='Scores', axis=0, ascending=False, inplace=True, kind='quicksort')
            values = temp['Scores'].to_numpy()
            labels = temp['Features']

        fig.add_trace(go.Bar(x=labels, y=values), row=int(index / 2) + 1, col=index % 2 + 1)
        # Scale of the y-axis is in log scale
        fig.update_yaxes(type="log", row=int(index / 2) + 1, col=index % 2 + 1)
        fig.update_xaxes(type="category", row=int(index / 2) + 1, col=index % 2 + 1)

    fig.update_layout(height=1000, title_text=method)
    fig.update_layout(showlegend=False, autosize=True)
    fig.show()


def fs_results_computation(X, Y):
    """ [GENERAL]

    :param X: Dataframe of independent variables
    :param Y: array of the target variable
    :return:  A dataframe where each column (Series) represent the score obtained by each method
    """
    labels = list(X.columns)
    results = pd.DataFrame()
    results['Features'] = labels

    # Pearson index computation
    pearson = []
    for (columnName, columnData) in X.iteritems():
        pearson.append((scipy.stats.pearsonr(columnData, Y)[0]))
    results['Pearson'] = pearson

    # Spearmnar index computation
    spearmanr = []
    for (columnName, columnData) in X.iteritems():
        spearmanr.append((scipy.stats.spearmanr(columnData, Y)[0]))
    results['Spearmanr'] = spearmanr

    # Kendall tau computation
    kendall = []
    for (columnName, columnData) in X.iteritems():
        kendall.append((scipy.stats.kendalltau(columnData, Y)[0]))
    results['Kendall'] = kendall

    # Fisher's score computation
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)

    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    fs.transform(X_train)
    # transform test input data
    fs.transform(X_test)
    results['Fisher'] = fs.scores_

    # Random Forest importance computation
    model = RandomForestRegressor()
    model.fit(X, Y)
    results['RF Importance'] = model.feature_importances_

    # Recursive Feature Selection
    results['RFS'] = recursive_feature_selection(X, Y.astype(int), 20)
    return results


def variance_threshold(data, th):
    """ [GENERAL]

    :param data:  Dataset
    :param th: threshold value used for VarianceThreshold
    :return: Dataset filtered without features with variance < th
    """
    selector = VarianceThreshold(threshold=th)
    selector.fit_transform(data)
    results = pd.DataFrame()
    scores = []
    for i in selector.variances_:
        if i >= th:
            scores.append(1)
        else:
            scores.append(0)

    results['Features'] = data.columns
    results['Scores'] = scores

    return results


def recursive_feature_selection(X, y, select):
    """[GENERAL]

    :param X: Dataframe of independent variables
    :param Y: array of the target variable
    :param select: number of variables selected by the Recursive Feature Selection method
    :return: a Series containing the results of RFS. if the feature is selected 1 is assigned, else 0
    """
    labels = list(X.columns)
    warnings.filterwarnings("ignore")
    # define RFE
    rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=None)
    # fit RFE
    rfe.fit(X, y)
    # summarize all features
    results = pd.DataFrame()
    results['Features'] = labels
    support = rfe.support_
    res = []
    for s in support:
        if (s == True):
            res.append(1)
        else:
            res.append(0)
    results['Ranking'] = res
    return results['Ranking']


def set_labels_df(dictionary, keys):
    """

    :param dictionary: list of dataframe of the results.
    :param keys: list of periods. Each period is a key of dictionary variable
    :return: return the list with more labels
    """
    labels = dictionary[keys[0]]['Features'].tolist()
    for k in keys:
        if (len(dictionary[k]['Features'].tolist()) > len(labels)):
            labels = dictionary[k]['Features'].tolist()
    return labels


def quasi_zero_variance(X, freqCut, uniqueCut):
    """

    :param X: Input dataframe
    :param freqCut:  threshold value
    :param uniqueCut:  threshold
    :return: return a list of 1 or 0, corresponding if a variable is discard (0) or not (1). A variable is discarded if,considering its sample,
    (percentage of unique values > freqCut) && (the ratio of the most prevalent over the second most prevalent value > uniqueCut)
    """
    scores = []
    for index, col in enumerate(X.columns):
        items_counts = X[col].value_counts()
        if(items_counts.iloc[0] == len(X[col])):
            scores.append(0)
        else:
            if items_counts.iloc[0] / len(X[col]) > uniqueCut and items_counts.iloc[0] / items_counts.iloc[1] > freqCut:
                scores.append(0)
            else:
                scores.append(1)
    results = pd.DataFrame()
    results['Features'] = X.columns
    results['Scores'] = scores
    return results

def get_tuple(s):

    d = s[2:4]
    m = s[0:2]
    y = s[10:14]
    s = d + '/' +m + '/' + y
    element =  datetime.strptime(s, "%d/%m/%Y")

    return datetime.timestamp(element)


