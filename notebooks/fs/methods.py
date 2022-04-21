from __future__ import print_function

import time
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
from numpy import arange, std, log
from scipy.stats import gmean
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

#Not used
def NormalizeData(data):
    result = (data - np.min(data)) / (np.max(data) - np.min(data))
    return result

#It normalized 1D array with MinMaxscaler
def NormalizeData1D(data):
    data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    result = scaler.fit_transform(data).reshape((-1,))
    return result

#Not used
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


def show_bars(labels, matrix, method, geopackages):
    fig = make_subplots(rows=len(geopackages), cols=1,  subplot_titles=geopackages)
    for index, values in enumerate(matrix):
        fig.add_trace(go.Bar(x=labels, y=values), row=index+1, col=1)
        fig.update_yaxes(row=index + 1, col=1)
        fig.update_xaxes(type="category", row=index + 1, col=1)


    fig.update_layout(height=1500, width=600, title_text=method)
    fig.update_layout(showlegend=False)
    fig.show()

def show_bars_log(labels, matrix, method, geopackages):
    fig = make_subplots(rows=len(geopackages), cols=1, subplot_titles=geopackages)
    for index, values in enumerate(matrix):
        fig.add_trace(go.Bar(x=labels, y=values), row=index+1, col=1)
        fig.update_yaxes(type="log", row=index + 1, col=1)
        fig.update_xaxes(type="category", row=index + 1, col=1)

    fig.update_layout(height=1200, width=600, title_text=method)
    fig.update_layout(showlegend=False)
    fig.show()





def show_bar_log(labels, scores, name):
    df = pd.DataFrame(list(zip(labels, scores)), columns=['Features', 'Scores'])
    fig = px.bar(df, x="Features", y="Scores", color="Scores", log_y=True)
    fig.update_layout(title_text=name)
    fig.show()


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


#method which returns labels of features which have no nan values
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

    #Pearson index computation
    pearson = []
    for (columnName, columnData) in X.iteritems():
        pearson.append(scipy.stats.pearsonr(columnData, Y)[0])
    results['Pearson'] = pearson

    #Spearmnar index computation
    spearmanr = []
    for (columnName, columnData) in X.iteritems():
        spearmanr.append(scipy.stats.spearmanr(columnData, Y)[0])
    results['Spearmanr'] = spearmanr

    #Kendall tau computation
    kendall = []
    for (columnName, columnData) in X.iteritems():
        kendall.append(scipy.stats.kendalltau(columnData, Y)[0])
    results['Kendall'] = kendall

    #Fisher's score computation
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

    #Random Forest importance computation
    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, Y)
    # get importance
    results['Random Forest Importance'] = model.feature_importances_

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


def variance_threshold(X_train, th):
    # define thresholds to check
    thresholds = arange(0.0, 0.55, 0.05)
    # apply transform with each threshold
    selector = VarianceThreshold(threshold=th)
    selector.fit_transform(X_train)

    results = pd.DataFrame()

    scores = []
    for i in selector.get_support():
        if i == False:
            scores.append(1)
        else:
            scores.append(0)

    results['Features'] = selector.feature_names_in_
    results['Scores'] = scores

    barPlot_func_onedata(results, "Variance Threshold")
    return scores


def exhaustive_feature_selection(X, y):
    labels = list(X.columns)

    warnings.filterwarnings("ignore")

    X = X.to_numpy()
    y = y.astype(int)
    lr = LinearRegression()

    efs1 = EFS(lr,
               min_features=10,
               max_features=20,
               scoring='r2',
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
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    results = pd.DataFrame()

    results['Scores'] = importance
    results['Features'] = labels

    # plot feature importance
    barPlot_func_onedata(results, "Random Forest Importance")
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

#Method used to process data before the use of FastGWR. It returns a Dataframe of the params as mentioned in https://github.com/Ziqi-Li/FastGWR
def mgwr_param(data, target):
    warnings.filterwarnings("ignore")
    temp = pd.DataFrame(data).dropna(axis=1)
    #Used to decrease the sample size
    #temp.drop(temp.tail(200).index, inplace=True)
    temp.drop(temp.tail(100).index, inplace=True)
    y = temp[target]

    df = pd.DataFrame(temp)

    y = y.values.ravel()

    df.pop('bottom')
    df.pop('top')
    df.pop('geometry')
    df.pop('left')
    df.pop('right')
    df = df.iloc[:, :-70]


    X = df.to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = y.reshape((-1, 1))
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    csv = pd.DataFrame()
    csv['X'] = temp['lat_cen']
    csv['Y'] = temp['lng_cen']
    csv['target'] = temp[target]
    csv = pd.concat([csv, pd.DataFrame(X)], axis=1)
    csv.to_csv(r'results/params.csv', index=False)




def mgwr(data, target, iterations):
    warnings.filterwarnings("ignore")
    temp = pd.DataFrame(data).dropna(axis=1)
#    temp.drop(temp.tail(200).index,
#           inplace=True)

    y = temp[target]

    df = pd.DataFrame(temp)

    y = y.values.ravel()


    df.pop('bottom')
    df.pop('top')
    df.pop('geometry')
    df.pop('left')
    df.pop('right')
    df.pop('lat_cen')
    df.pop('lng_cen')

    # X = df.drop(['prim_road', 'sec_road', 'highway', 'farms'], axis=1)
    #labels = list(X.columns)
    labels = list(df.columns)
    coords = list(zip(temp['lat_cen'], temp['lng_cen']))
    
    X = df.to_numpy()

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    Y = y.reshape((-1, 1))
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    sel = Sel_BW(coords, Y, X, multi=True, kernel='gaussian', spherical=True, fixed=True)

    n_proc = 8
    pool = mp.Pool(n_proc)
    #bw = sel.search(pool=pool)
    #print('bw:', bw)
    
    bw = sel.search(pool=pool, criterion='CV', max_iter_multi=iterations)

    print('bw(intercept):', bw[0])

    mgwr = MGWR(coords, Y, X, selector=sel, spherical=True, kernel='gaussian', fixed=True)
    mgwr_results = mgwr.fit(pool=pool)
    pool.close()  # Close the pool when you finish
    pool.join()
    bandwidths = np.delete(bw, 0)
    df_bw = pd.DataFrame(bandwidths)
    df_betas = pd.DataFrame(mgwr_results.params)
    df_bw.to_csv(r'results/mgwr_bw'+str(iterations)+'.csv', index=False)
    df_betas.to_csv(r'results/mgwr_betas'+str(iterations)+'.csv', index=False)


    mgwr_results.summary()

    fig = go.Figure(data=[
       # go.Bar(name='Mean', x=labels, y=df_betas.mean(axis=0)),
        go.Bar(name='Median', x=labels, y=bandwidths),
      #  go.Bar(name='Bandwidth', x=labels, y=df_bw)
    ])
    # Change the bar mode
    fig.show()

    fig2 = go.Figure(data=[
        go.Bar(name='Bandwidth', x=labels, y=pd.Series(df_bw))
    ])
    fig2.show()



#Not used
def compute_mgwr_betas(df_betas, labels):
    mean = df_betas.mean(axis=0)
    median = df_betas.median(axis=0)

    results = pd.DataFrame()
    results['Labels'] = labels
    results['Mean'] = mean.tolist()
    results['Median'] = median.tolist()

    return results

#Not used
def compute_mgwr_bw(df_bw, labels):
    labels = labels.insert(0, 'intercept')
    results = pd.DataFrame()
    results['Labels'] = labels
    results['Bandwidth'] = df_bw

#exclude all features with '_st' and '_lcs' in the name
def noSensor_features(strings):
    result = []
    for i in strings:
        if ("_st" in i) == False:
            if (("_lcs" in i) == False):
                result.append(i)

    return result


#Not used
def clean_dataset_nosensor(df, labels):
    X = pd.DataFrame(df, columns=labels).dropna()
    labels = check_NotNull(X)
    X = pd.DataFrame(X, columns=labels)
    X.pop('geometry')
    return X

#Not used
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


