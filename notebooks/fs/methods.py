import time
from pathlib import Path
from timeit import timeit
from plotly.subplots import make_subplots
from ipywidgets import interact, interactive, fixed, interact_manual, VBox
import multiprocessing as mp
import ipywidgets as widgets
import scipy.stats
import pandas as pd
from ipywidgets import widgets, interact
from matplotlib.container import Container
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gpd
from numpy import arange, std, log, savetxt, loadtxt
from scipy.spatial import cKDTree
from scipy.stats import gmean, stats
import numpy as np
from IPython.display import display, clear_output
from ipywidgets import widgets
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, RFE, RFECV
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
import statistics
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import plotly.express as px
import warnings
from scipy.linalg import LinAlgWarning
import borda.count
import pyrankvote
from pyrankvote import Candidate, Ballot

def borda_voting(dataframe):
    labels = dataframe['Features']
    dataframe = dataframe.loc[:, dataframe.columns != 'Features']


    selection = borda.count.Election()
    selection.set_candidates(labels.tolist())
    for col in dataframe.columns:
        method_ranking = getRanks(dataframe[col].tolist(), labels)
        voter = borda.count.Voter(selection, col)
        voter.votes(method_ranking)
    return selection

def pyrankVote(dataframe):
    labels = dataframe['Features']
    candidates = []
    dataframe = dataframe.loc[:, dataframe.columns != 'Features']

    for candidate in labels:
        candidates.append(Candidate(candidate))

    ballots = []
    for col in dataframe.columns:
        method_ranking = getRanks(dataframe[col].tolist(), candidates)
        ballots.append(Ballot(ranked_candidates=method_ranking))

    election_result = pyrankvote.instant_runoff_voting(candidates, ballots)
    print(election_result)
    return election_result


def getRanks(values, labels):
    zipped = list(zip(labels, values))

    data = pd.DataFrame(data = zipped, columns=['Features', 'Scores'])
    data.sort_values(by='Scores', axis=0, ascending=False, inplace=True, kind='quicksort')
    return data['Features'].tolist()

def process_data(data, k):
    st = [col for col in data.columns if col.endswith('_st')]
    interpolated = [col for col in data.columns if col.endswith('_int')]
    data = increase_data(data, 'pm25_st', k)
    data.pop('dusaf')
    data.pop('siarl')
    data.pop('top')
    data.pop('bottom')
    data.pop('right')
    data.pop('left')
    data.pop('pm25_int')
    return data

"""
    for col in st:
        data = increase_data(data, col, k)

    for col in interpolated:
        if(col in ['pm25_int', 'nox_int', 'no2_int']):
            data.pop(col)
"""







def increase_data(data, sensor, k):
    points_st = data[~data[sensor].isnull()]
    return add_buffer(points_st, data, data, k, sensor[:-3])


def add_buffer(points, data, uncleaned_data, k, sensor):
    warnings.filterwarnings("ignore")

    nA = np.array(list(points.geometry.centroid.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(data.geometry.centroid.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)

    for cell in nA:
        dist, idx = btree.query(cell, k)

        for i in range(0, k):
            uncleaned_data.at[idx[i], sensor+'_st'] = uncleaned_data.loc[idx[i]][sensor+'_int']
    return uncleaned_data

# Not used
def NormalizeData(data):
    result = (data - np.min(data)) / (np.max(data) - np.min(data))
    return result


# It normalized 1D array with MinMaxscaler
def NormalizeData1D(data):
    data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    result = scaler.fit_transform(data).reshape((-1,))
    return result


# Not used
def NormalizeData2D(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def columnNotNull(array):
    for i in array:
        if (i == True):
            return True
    return False


def show_bar(labels, scores, name):
    df = pd.DataFrame(list(zip(labels, scores)), columns=['Features', 'Scores'])
    fig = px.bar(df, x="Features", y="Scores", color="Scores")
    fig.update_layout(title_text=name)
    fig.show()


def show_bars(labels_list, matrix, method, geopackages):
    titles = []
    for g in geopackages:
        titles.append(g)
    fig = make_subplots(rows=int(len(geopackages) / 2) + 1, cols=2, subplot_titles=titles)
    for index, values in enumerate(matrix):
        labels=labels_list[index]
        fig.add_trace(go.Bar(x=labels, y=values), row=int(index / 2) + 1, col=index % 2 + 1)
        fig.update_yaxes(row=int(index / 2) + 1, col=index % 2 + 1)
        fig.update_xaxes(type="category", row=int(index / 2) + 1, col=index % 2 + 1)

    fig.update_layout(height=1000, title_text=method)
    fig.update_layout(showlegend=False, autosize=True)
    fig.show()


def show_bars_log(labels_list, matrix, method, geopackages):
    titles = []
    for g in geopackages:
        titles.append(g)
    fig = make_subplots(rows=int(len(geopackages) / 2) + 1, cols=2, subplot_titles=titles)
    for index, values in enumerate(matrix):
        labels=labels_list[index]
        fig.add_trace(go.Bar(x=labels, y=values), row=int(index / 2) + 1, col=index % 2 + 1)
        fig.update_yaxes(type="log", row=int(index / 2) + 1, col=index % 2 + 1)
        fig.update_xaxes(type="category", row=int(index / 2) + 1, col=index % 2 + 1)

    fig.update_layout(height=1000, title_text=method)
    fig.update_layout(showlegend=False, autosize=True)
    fig.show()


def show_bar_log(labels, scores, name):
    df = pd.DataFrame(list(zip(labels, scores)), columns=['Features', 'Scores'])
    fig = px.bar(df, x="Features", y="Scores", color="Scores", log_y=True)
    fig.update_layout(title_text=name)
    fig.show()

def getTitle_gpkg(string):
   return string[11:13] + '/' + string[9:11] + ' - ' + string[16:18] + '/' + string[14:16] + ' (' + string[19:23] + ')'


def barPlot_func_onedata(values, plot_name):
    scale = widgets.RadioButtons(
        options=['Regular', 'Logaritmic'],
        description='Scale:',
        disabled=False,
    )

    order = widgets.RadioButtons(
        options=['Labels', 'Scores'],
        description='Order by:',
        disabled=False
    )

    norm = widgets.Checkbox(
        value=True,
        description='Results normalized',
        disabled=False,
        indent=True
    )

    def barPlot_manager(change_scale, change_order, normalized):
        df = pd.DataFrame(data=values)
        if (change_scale == 'Logaritmic'):
            if (change_order == 'Scores'):
                df = df.sort_values(by='Scores', ascending=False)
                if normalized:
                    show_bar_log(df['Features'], NormalizeData1D(df['Scores']), plot_name)
                else:
                    show_bar_log(df['Features'], df['Scores'], plot_name)
                return
            else:
                if normalized:
                    show_bar_log(df['Features'], NormalizeData1D(df['Scores']), plot_name)
                else:
                    show_bar_log(df['Features'], df['Scores'], plot_name)
                    return
        else:
            if (change_order == 'Scores'):
                df = df.sort_values(by='Scores', ascending=False)
                if normalized:
                    show_bar(df['Features'], NormalizeData1D(df['Scores']), plot_name)
                else:
                    show_bar(df['Features'], df['Scores'], plot_name)
                    return
            else:
                if normalized:
                    show_bar(df['Features'], NormalizeData1D(df['Scores']), plot_name)
                else:
                    show_bar(df['Features'], df['Scores'], plot_name)
                return

    ui = widgets.HBox([norm, scale, order])
    out = widgets.interactive_output(barPlot_manager,
                                     {'change_scale': scale, 'change_order': order, 'normalized': norm})
    display(ui, out)


# method which returns labels of features which have no nan values
def check_NotNull(df):
    bool = df.isna()
    labels = []
    for (columnName, columnData) in bool.iteritems():
        if not columnNotNull(columnData):
            labels.append(columnName)

    return labels


def pearson(X, Y):
    labels = list(X.columns)
    pearson = []
    for (columnName, columnData) in X.iteritems():
        pearson.append(scipy.stats.pearsonr(columnData, Y)[0])

    results = pd.DataFrame()
    results['Scores'] = pearson
    results['Features'] = labels

    barPlot_func_onedata(results, "Pearson Index")

    return pearson


def spearmanr(X, Y):
    labels = list(X.columns)

    spearmanr = []
    for (columnName, columnData) in X.iteritems():
        spearmanr.append(scipy.stats.spearmanr(columnData, Y)[0])

    results = pd.DataFrame()
    results['Scores'] = spearmanr
    results['Features'] = labels

    barPlot_func_onedata(results, "Spearmanr Rho")
    return spearmanr


def kendall(X, Y):
    labels = list(X.columns)

    kendall = []
    for (columnName, columnData) in X.iteritems():
        kendall.append(scipy.stats.kendalltau(columnData, Y)[0])

    results = pd.DataFrame()
    results['Scores'] = kendall
    results['Features'] = labels

    barPlot_func_onedata(results, "Kendall Tau")
    return kendall


def f_test(X, y):
    labels = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    fs.transform(X_train)
    # transform test input data
    fs.transform(X_test)

    results = pd.DataFrame()
    scores = fs.scores_
    results['Scores'] = scores
    results['Features'] = labels

    barPlot_func_onedata(results, "Fisher’s Score")
    return scores


def chi2_test(X, y):
    labels = list(X.columns)

    X = X.astype(int)
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    fs.transform(X_train)
    # transform test input data
    fs.transform(X_test)

    results = pd.DataFrame()

    scores = fs.scores_
    results['Scores'] = scores
    results['Features'] = labels

    barPlot_func_onedata(results, "Chi-Square Score")
    return scores


def fs_results_computation(X, Y):
    labels = list(X.columns)
    results = pd.DataFrame()
    results['Features'] = labels

    # Pearson index computation
    pearson = []
    for (columnName, columnData) in X.iteritems():
        pearson.append(scipy.stats.pearsonr(columnData, Y)[0])
    results['Pearson'] = pearson

    # Spearmnar index computation
    spearmanr = []
    for (columnName, columnData) in X.iteritems():
        spearmanr.append(scipy.stats.spearmanr(columnData, Y)[0])
    results['Spearmanr'] = spearmanr

    # Kendall tau computation
    kendall = []
    for (columnName, columnData) in X.iteritems():
        kendall.append(scipy.stats.kendalltau(columnData, Y)[0])
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
    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, Y)
    # get importance
    results['RF Importance'] = model.feature_importances_

    # results['Betas Median (MGWR)'] = mgwr_results(X, Y, 10, coords)

    return results


def compute_dispersion_ratio(X):
    labels = list(X.columns)

    dispersion_ratio = []

    for i in range(len(X[0])):
        dispersion_ratio.append(statistics.mean(X[:, i]) / gmean(X[:, i]))

    results = pd.DataFrame()
    results['Scores'] = dispersion_ratio
    results['Features'] = labels

    barPlot_func_onedata(dispersion_ratio, 'Dispersion Ratio for each variable')

    for i in range(len(labels)):
        print(labels[i], ': ', dispersion_ratio[i])

    return dispersion_ratio


def variance_threshold(data, th):
    # define thresholds to check
    #thresholds = arange(0.0, 0.55, 0.05)
    # apply transform with each threshold
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


def exhaustive_feature_selection(X, y):
    labels = list(X.columns)

    warnings.filterwarnings("ignore")

    X = X.to_numpy()
    y = y.astype(int)
    lr = LinearRegression()

    efs1 = EFS(lr,
               min_features=10,
               max_features=20,
               scoring='mae',
               n_jobs=-1,
               cv=5)

    efs1.fit(X, y)
    print('Best accuracy score: %.2f' % efs1.best_score_)
    print('Best subset (indices):', efs1.best_idx_)
    print('Best subset (corresponding names):')
    for i in efs1.best_idx_:
        print(labels[i])

    df = pd.DataFrame.from_dict(efs1.get_metric_dict()).T
    df.sort_values('avg_score', inplace=True, ascending=False)
    for i in efs1.best_idx_:
        print(labels[i])

    return df


def RF_importance(X, y):
    labels = list(X.columns)

    # define the model
    model = RandomForestRegressor(n_estimators=130)
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    results = pd.DataFrame()

    results['Scores'] = importance
    results['Features'] = labels

    # plot feature importance
    barPlot_func_onedata(results, "RF Importance")
    return importance


def detect_n_feature_RFE(X, y):
    X = X.to_numpy()
    # create pipeline
    rfe = RFECV(estimator=DecisionTreeClassifier())
    model = DecisionTreeClassifier()
    pipeline = Pipeline(steps=[('s', rfe), ('m', model)])  # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print('Accuracy: %.3f (%.3f)' % (statistics.mean(n_scores), std(n_scores)))


def recursive_feature_selection(X, y, select):
    labels = list(X.columns)

    warnings.filterwarnings("ignore")
    # define RFE
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=select)
    # fit RFE
    rfe.fit(X, y)
    # summarize all features

    results = pd.DataFrame()
    results['Features'] = labels
    results['isSelected'] = rfe.support_
    results['Ranking'] = rfe.ranking_
    return results


def get_models():
    models = dict()
    for i in range(2, 10):
        rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=i)
        model = DecisionTreeClassifier()
        models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])
    return models


def evaluate_model(model, X1, y1):
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
                               random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=3, n_jobs=-1)
    return scores


def mgwr_beta(data, target, iterations, geopackage):

    warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='mgwr')
    warnings.filterwarnings("ignore")

    labels = check_NotNull(data)  # temp.drop(temp.tail(200).index,
    #           inplace=True)
    X = pd.DataFrame(data=data, columns=labels)
    Y = X[target]
    X.pop(target)
    Y = Y.values.ravel()

    coords = list(zip(X['lat_cen'], X['lng_cen']))
    X.pop('lat_cen')
    X.pop('lng_cen')

    X = X.to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = Y.reshape((-1, 1))
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)


    sel = Sel_BW(coords, Y, X, multi=True, kernel='gaussian', spherical=True, fixed=True)
    n_proc = 2
    pool = mp.Pool(n_proc)
    bw = sel.search(pool=pool, criterion='CV', max_iter_multi=iterations)
    mgwr = MGWR(coords, Y, X, selector=sel, spherical=True, kernel='gaussian', fixed=True)
    mgwr_results = mgwr.fit(pool=pool)
    pool.close()  # Close the pool when you finish
    pool.join()

    bandwidths = np.delete(bw, 0)
    med = np.median(mgwr_results.params, axis=0)
    med = np.delete(med, 0)

    res = pd.DataFrame()
    res['Bandwidthds'] = bandwidths
    res['Betas Median'] = med

    return res


def mgwr_results(X, target, iterations, coords):
    warnings.filterwarnings("ignore")
    X = X.apply(stats.zscore)
    X = X.dropna(axis=1)
    X = X.to_numpy()
    Y = target.values.ravel()
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    sel = Sel_BW(coords, Y, X, multi=True, kernel='gaussian', spherical=True, fixed=True)

    n_proc = 8
    pool = mp.Pool(n_proc)
    # bw = sel.search(pool=pool)
    # print('bw:', bw)

    bw = sel.search(pool=pool, criterion='CV', max_iter_multi=iterations)

    mgwr = MGWR(coords, Y, X, selector=sel, spherical=True, kernel='gaussian', fixed=True)
    mgwr_results = mgwr.fit(pool=pool)

    # Close the pool when you finish
    pool.close()
    pool.join()
    # bandwidths = np.delete(bw, 0)
    mgwr_results.summary()
    return np.median(mgwr_results.params, axis=0)


# Not used
def compute_mgwr_betas(df_betas, labels):
    mean = df_betas.mean(axis=0)
    median = df_betas.median(axis=0)

    results = pd.DataFrame()
    results['Labels'] = labels
    results['Mean'] = mean.tolist()
    results['Median'] = median.tolist()

    return results


# Not used
def compute_mgwr_bw(df_bw, labels):
    labels = labels.insert(0, 'intercept')
    results = pd.DataFrame()
    results['Labels'] = labels
    results['Bandwidth'] = df_bw


# exclude all features with '_st' and '_lcs' in the name
def noSensor_features(strings):
    result = []
    for i in strings:
        if ("_st" in i) == False:
            if (("_lcs" in i) == False):
                result.append(i)

    return result


# Not used
def clean_dataset_nosensor(df, labels):
    X = pd.DataFrame(df, columns=labels).dropna()
    labels = check_NotNull(X)
    X = pd.DataFrame(X, columns=labels)
    X.pop('geometry')
    return X


# Not used
def score_rfe(dataframe):
    scores = []
    for i in dataframe['Ranking']:
        if (i == 1):
            scores.append(1)
        if (i == 2):
            scores.append(0.5)
        else:
            scores.append(0)

    return scores
