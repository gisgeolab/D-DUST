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
from sklearn.tree import DecisionTreeClassifier

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def barPlot_func_onedata(values, varLabels, plot_name):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_ylabel('Scores')
    ax.set_title(plot_name)
    r = np.arange(len(varLabels))
    width = 0.75

    ax.bar(r, values, color='b',
           width=width,
           )

    plt.xticks(r + width / 2, varLabels)
    plt.xticks(rotation=90)

    plt.xlabel('Variables')
    plt.ylabel('Dispersion Ratio')
    plt.show()


def pearson(X, Y, labels):
    pearson = []
    for (columnName, columnData) in X.iteritems():
        pearson.append(scipy.stats.pearsonr(columnData, Y)[0])

    pearson = NormalizeData(pearson)
    barPlot_func_onedata(pearson, labels, "Pearson Index")


def spearmanr(X, Y, labels):
    spearmanr = []
    for (columnName, columnData) in X.iteritems():
        spearmanr.append(scipy.stats.spearmanr(columnData, Y)[0])

    spearmanr = NormalizeData(spearmanr)
    barPlot_func_onedata(spearmanr, labels, "Spearmanr Rho")


def kendall(X, Y, labels):
    kendall = []
    for (columnName, columnData) in X.iteritems():
        kendall.append(scipy.stats.kendalltau(columnData, Y)[0])

    kendall = NormalizeData(kendall)
    barPlot_func_onedata(kendall, labels, "Kendall Tau")


def f_test(X_train, y_train, X_test, labels):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    fs.transform(X_train)
    # transform test input data
    fs.transform(X_test)
    barPlot_func_onedata(NormalizeData(fs.scores_), labels, "Fisher’s Score")


def chi2_test(X_train, y_train, X_test, labels):
    X_train = X_train.astype(int)
    y_train = y_train.astype(int)
    X_test = X_test.astype(int)

    # configure to select all features
    fs = SelectKBest(score_func=chi2, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    fs.transform(X_train)
    # transform test input data
    fs.transform(X_test)
    barPlot_func_onedata(NormalizeData(fs.scores_), labels, "Chi-Square Score")


def compute_dispersion_ratio(X, labels):
    dispersion_ratio = []

    for (columnName, columnData) in X.iteritems():
        x_new = [i for i in columnData if i != 0]
        dispersion_ratio.append(statistics.mean(x_new) / gmean(x_new))

    barPlot_func_onedata(dispersion_ratio, labels, 'Dispersion Ratio for each variable')

    for i in range(len(labels)):
        print(labels[i], ': ', dispersion_ratio[i])


def variance_threshold(X_train, labels):
    # define thresholds to check
    thresholds = arange(0.0, 0.55, 0.05)
    # apply transform with each threshold
    selector = VarianceThreshold(threshold=0)
    selector.fit_transform(X_train)
    barPlot_func_onedata(NormalizeData(selector.variances_), labels, "Variance Threshold")


def exhaustive_feature_selection(X_train, y_train, labels):
    import warnings
    warnings.filterwarnings("ignore")

    X = X_train.to_numpy()
    y = y_train.astype(int)
    lr = LinearRegression()

    efs1 = EFS(lr,
               min_features=4,
               max_features=6,
               scoring='r2',
               n_jobs=-1,
               cv=5)

    efs1.fit(X, y)
    print('Best accuracy score: %.2f' % efs1.best_score_)
    print('Best subset (indices):', efs1.best_idx_)
    print('Best subset (corresponding names):')
    for i in efs1.best_idx_:
        print(labels[i])


def RF_importance(X_train, y_train, labels):
    # define the model
    model = RandomForestRegressor()

    # fit the model
    model.fit(X_train, y_train)

    # get importance
    importance = NormalizeData(model.feature_importances_)
    # summarize feature importance
    for i, v in enumerate(importance):
        print(labels[i], ': ', '%.5f' % (v))
    # plot feature importance
    barPlot_func_onedata(importance, labels, "Random Forest Importance")


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


def recursive_feature_selection(X_train, y_train, labels, select):
    warnings.filterwarnings("ignore")
    # define RFE
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=select)
    # fit RFE
    rfe.fit(X_train, y_train)
    # summarize all features

    for i in range(X_train.shape[1]):
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

    mgwr_results.summary()



