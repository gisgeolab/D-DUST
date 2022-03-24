from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
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


def NormalizeData(data):
    result = (data - np.min(data)) / (np.max(data) - np.min(data))
    return result


def NormalizeData1D(data):
    data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    result = scaler.fit_transform(data).reshape((-1,))
    return result


def NormalizeData2D(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


def columnNotNull(array):
    for i in array:
        if (i == True):
            return True
    return False


def barPlot_func_onedata(values, plot_name):

    def barPlot_manager(change_scale, change_order):
        fig = go.Figure()
        if (change_scale == 'Logaritmic'):
            if(change_order=='Scores'):
                df = values.sort_values(by='Scores', ascending=False)
                trace = go.Bar(x=df['Variables'], y=np.log(df['Scores']))
            else:
                trace = go.Bar(x=values['Variables'], y=np.log(values['Scores']))
        else:
            if(change_order == 'Scores'):
                df = values.sort_values(by = 'Scores',  ascending=False)
                trace = go.Bar(x=df['Variables'], y=df['Scores'])
            else:
                trace = go.Bar(x=values['Variables'], y=values['Scores'])

        fig.add_trace(trace)
        fig.show()
        container = widgets.Box([scale, order])
        display(container)

    scale = widgets.RadioButtons(
        options=['Regular', 'Logaritmic'],
        description='Score scales:',
        disabled=False,
    )
    order = widgets.RadioButtons(
        options=['Labels', 'Scores'],
        description='Order by:',
        disabled=False
    )

    interact(barPlot_manager, change_scale = scale, change_order = order)

def check_NotNull(df):
    bool = df.isna()
    labels = []
    for (columnName, columnData) in bool.iteritems():
        if not columnNotNull(columnData):
            labels.append(columnName)

    return labels


def pearson(X, Y, normalized):
    labels = list(X.columns)
    pearson = []
    for (columnName, columnData) in X.iteritems():
        pearson.append(scipy.stats.pearsonr(columnData, Y)[0])
    if (normalized):
        pearson = NormalizeData1D(pearson)

    results = pd.DataFrame()
    results['Scores'] = pearson
    results['Variables'] = labels

    barPlot_func_onedata(results, "Pearson Index")

    return pearson


def spearmanr(X, Y, normalized):
    labels = list(X.columns)

    spearmanr = []
    for (columnName, columnData) in X.iteritems():
        spearmanr.append(scipy.stats.spearmanr(columnData, Y)[0])

    if (normalized):
        spearmanr = NormalizeData1D(spearmanr)

    results = pd.DataFrame()
    results['Scores'] = spearmanr
    results['Variables'] = labels

    barPlot_func_onedata(results, "Spearmanr Rho")
    return spearmanr


def kendall(X, Y, normalized):
    labels = list(X.columns)

    kendall = []
    for (columnName, columnData) in X.iteritems():
        kendall.append(scipy.stats.kendalltau(columnData, Y)[0])

    if (normalized):
        kendall = NormalizeData1D(kendall)

    results = pd.DataFrame()
    results['Scores'] = kendall
    results['Variables'] = labels

    barPlot_func_onedata(results, "Kendall Tau")
    return kendall


def f_test(X, y, normalized):
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
    if (normalized):
        scores = NormalizeData1D(scores)
    results['Scores'] = scores
    results['Variables'] = labels

    barPlot_func_onedata(results, "Fisher’s Score")
    return scores


def chi2_test(X, y, normalized):
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
    if (normalized):
        scores = NormalizeData1D(scores)

    results['Scores'] = scores
    results['Variables'] = labels

    barPlot_func_onedata(results, "Chi-Square Score")
    return scores


def compute_dispersion_ratio(X):
    labels = list(X.columns)

    dispersion_ratio = []

    for i in range(len(X[0])):
        dispersion_ratio.append(statistics.mean(X[:, i]) / gmean(X[:, i]))

    results = pd.DataFrame()
    results['Scores'] = dispersion_ratio
    results['Variables'] = labels

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

    results['Variables'] = selector.feature_names_in_
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
               min_features=2,
               max_features=4,
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


def RF_importance(X, y, normalized):
    labels = list(X.columns)

    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    results = pd.DataFrame()

    if (normalized):
        importance = NormalizeData1D(importance)
    results['Scores'] = importance
    results['Variables'] = labels

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
    results['Variables'] = labels
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


def mgwr(data, labels, coords, y):
    df = pd.DataFrame(data, columns=labels).dropna()

    X = df.drop(['prim_road', 'sec_road', 'highway', 'farms'], axis=1)
    labels = list(X.columns)
    X = X.to_numpy()
    #  lat = pd.DataFrame(data, columns=['lat'])
    # lat = lat['lat'].tolist()

    #  lon = pd.DataFrame(data, columns=['lon'])
    #  lon = lon['lon'].tolist()
    #  print(matrix_rank(X))

    # coords = list(zip(lat, lon))

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    Y = y.reshape((-1, 1))

    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    sel = Sel_BW(coords, Y, X)

    bw = sel.search()
    print('bw:', bw)

    selector = Sel_BW(coords, Y, X, multi=True, constant=True)
    bw = selector.search(multi_bw_min=[2])

    print("bw( intercept ):", bw[0])

    df_bw = pd.DataFrame(bw)

    df_bw.to_csv(r'results/mgwr_bw.csv', index=False)

    mgwr = MGWR(coords, Y, X, selector, constant=True)
    mgwr_results = mgwr.fit()

    print('aicc:', mgwr_results.aicc)
    print('sigma2:', mgwr_results.sigma2)
    print('ENP(model):', mgwr_results.ENP)
    print('adj_alpha(model):', mgwr_results.adj_alpha[1])
    print('critical_t(model):', mgwr_results.critical_tval(alpha=mgwr_results.adj_alpha[1]))
    alphas = mgwr_results.adj_alpha_j[:, 1]
    critical_ts = mgwr_results.critical_tval()
    print('\n')
    print('ENP(intercept):', mgwr_results.ENP_j[0])
    print('adj_alpha(intercept):', alphas[0])
    print('critical_t(intercept):', critical_ts[0])
    print('\n')
    for i in range(len(labels)):
        print("ENP(", labels[i], '): ', mgwr_results.ENP_j[i + 1])
        print("adj_alpha(", labels[i], '): ', alphas[i + 1])
        print("critical_t(", labels[i], '): ', critical_ts[i + 1])

    df_betas = pd.DataFrame(mgwr_results.params)

    df_betas.to_csv(r'results/mgwr_betas.csv', index=False)
    mgwr_results.summary()


def compute_mgwr_betas(df_betas, labels):
    mean = df_betas.mean(axis=0)
    median = df_betas.median(axis=0)

    results = pd.DataFrame()
    results['Labels'] = labels
    results['Mean'] = mean.tolist()
    results['Median'] = median.tolist()

    return results


def compute_mgwr_bw(df_bw, labels):
    labels = labels.insert(0, 'intercept')
    results = pd.DataFrame()
    results['Labels'] = labels
    results['Bandwidth'] = df_bw


def noSensor_features(strings):
    result = []
    for i in strings:
        if ("_st" in i) == False:
            if (("_lcs" in i) == False):
                result.append(i)

    return result


def clean_dataset_nosensor(df, labels):
    X = pd.DataFrame(df, columns=labels).dropna()
    labels = check_NotNull(X)
    X = pd.DataFrame(X, columns=labels)
    X.pop('geometry')
    return X


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


def scale_change():
    clear_output(wait=True)
    print(scale.value)


def order_change():
    print(order.value)
