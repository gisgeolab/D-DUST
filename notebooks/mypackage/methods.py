import warnings
import scipy.stats
import pandas as pd
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from numpy import arange, std
from numpy.linalg import matrix_rank
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, VarianceThreshold, RFE, RFECV
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import chi2
import statistics
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

import plotly.express as px


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def barPlot_func_onedata(values, plot_name):
    fig = px.bar(values, y='Scores', x='Variables', text_auto='0.2f', title=plot_name)
    fig.show()
    

def pearson(X, Y, labels):
    pearson = []
    for (columnName, columnData) in X.iteritems():
        pearson.append(scipy.stats.pearsonr(columnData, Y)[0])

    pearson = NormalizeData(pearson)

    results = pd.DataFrame()
    results['Scores'] = pearson
    results['Variables'] = labels

    barPlot_func_onedata(results, "Pearson Index")

    return results


def spearmanr(X, Y, labels):
    spearmanr = []
    for (columnName, columnData) in X.iteritems():
        spearmanr.append(scipy.stats.spearmanr(columnData, Y)[0])

    spearmanr = NormalizeData(spearmanr)

    results = pd.DataFrame()
    results['Scores'] = spearmanr
    results['Variables'] = labels

    barPlot_func_onedata(results , "Spearmanr Rho")
    return results


def kendall(X, Y, labels):
    kendall = []
    for (columnName, columnData) in X.iteritems():
        kendall.append(scipy.stats.kendalltau(columnData, Y)[0])

    kendall = NormalizeData(kendall)

    results = pd.DataFrame()
    results['Scores'] = kendall
    results['Variables'] = labels

    barPlot_func_onedata(results, "Kendall Tau")
    return results


def f_test(X, y, labels):
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
    results['Scores'] = NormalizeData(fs.scores_)
    results['Variables'] = labels

    barPlot_func_onedata(results, "Fisher’s Score")
    return results


def chi2_test(X, y, labels):
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
    results['Scores'] = NormalizeData(fs.scores_)
    results['Variables'] = labels

    barPlot_func_onedata(results,  "Chi-Square Score")
    return results


def compute_dispersion_ratio(X, labels):
    dispersion_ratio = []

    for i in range(len(X[0])):
        dispersion_ratio.append(statistics.mean(X[:, i]) / gmean(X[:, i]))

    results = pd.DataFrame()
    results['Scores'] = dispersion_ratio
    results['Variables'] = labels

    barPlot_func_onedata(dispersion_ratio, 'Dispersion Ratio for each variable')

    for i in range(len(labels)):
        print(labels[i], ': ', dispersion_ratio[i])



    return results


def variance_threshold(X_train, labels):
    # define thresholds to check
    thresholds = arange(0.0, 0.55, 0.05)
    # apply transform with each threshold
    selector = VarianceThreshold(threshold=0)
    selector.fit_transform(X_train)

    results = pd.DataFrame()
    results['Scores'] = NormalizeData(selector.variances_)
    results['Variables'] = labels

    barPlot_func_onedata(results, "Variance Threshold")
    return results


def exhaustive_feature_selection(X, y, labels):
    import warnings
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


def RF_importance(X, y, labels):
    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = NormalizeData(model.feature_importances_)
    results = pd.DataFrame()
    results['Scores'] = importance
    results['Variables'] = labels

    # plot feature importance
    barPlot_func_onedata(results, "Random Forest Importance")
    return results


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


def recursive_feature_selection(X, y, labels, select):
    warnings.filterwarnings("ignore")
    # define RFE
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=select)
    # fit RFE
    rfe.fit(X, y)
    # summarize all features

    for i in range(X.shape[1]):
        print('Label: %s, Selected=%s, Rank: %s' % (labels[i], rfe.support_[i], rfe.ranking_[i]))


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


def mgwr(data, labels):
    warnings.filterwarnings("ignore")
    df = pd.DataFrame(data, columns=labels)

    X = df.to_numpy()
    lat = pd.DataFrame(data, columns=['lat'])
    lat = lat['lat'].tolist()

    lon = pd.DataFrame(data, columns=['lon'])
    lon = lon['lon'].tolist()
    print(matrix_rank(X))

    coords = list(zip(lat, lon))

    Y = pd.DataFrame(data, columns=['FBpop_tot']).to_numpy()

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    Y = Y.reshape((-1, 1))

    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    sel = Sel_BW(coords, Y, X)

    bw = sel.search()
    print('bw:', bw)

    selector = Sel_BW(coords, Y, X, multi=True, constant=True)
    bw = selector.search(multi_bw_min=[2], multi_bw_max=[159])

    print("bw( intercept ):", bw[0])

    print('\n')
    for i in range(len(labels)):
        print("bw(", labels[i], '): ', bw[i + 1])
    print('\n')

    tuple = ["MGWR Bandwidths", bw]
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

    print("--betas coefficient--")
    for i in range(len(labels)):
        print(labels[i], ": ", np.mean(mgwr_results.params[:, i]), np.median(mgwr_results.params[:, i]))

    tuple.append(["Beta Coefficient", [[mgwr_results.mean(axis=0)], [mgwr_results.median(axis=0)]]])
    return tuple
