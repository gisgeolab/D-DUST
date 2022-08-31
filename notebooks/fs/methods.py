
from plotly.subplots import make_subplots
import scipy.stats
import pandas as pd
from numpy import std
from scipy.spatial import cKDTree
import numpy as np
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
import borda.count


def borda_voting(dataframe):
    labels = dataframe['Features']
    dataframe_new = dataframe.loc[:, dataframe.columns != 'Features']

    selection = borda.count.Election()
    selection.set_candidates(labels.tolist())
    for col in dataframe_new.columns:
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
    zipped = list(zip(labels, values))

    data = pd.DataFrame(data=zipped, columns=['Features', 'Scores'])
    data.sort_values(by='Scores', axis=0, ascending=False, inplace=True, kind='quicksort')
    return data['Features'].tolist()


def process_data(data, k, sensor):

    data = increase_data(data, sensor, k)
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
    data.pop(sensor[: -2]+'int')

    return data


def increase_data(data, sensor, k):
    points_st = data[~data[sensor].isnull()]

    return add_buffer(points_st, data, data, k, sensor)


def add_buffer(points, data, uncleaned_data, k, sensor):
    warnings.filterwarnings("ignore")

    nA = np.array(list(points.geometry.centroid.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(data.geometry.centroid.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)

    for cell in nA:
        dist, idx = btree.query(cell, k)

        for i in range(0, k):
            uncleaned_data.at[idx[i], sensor] = uncleaned_data.loc[idx[i]][sensor[:-2] + 'int']
    return uncleaned_data
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


def show_bars(labels_list, matrix, method, geopackages, order):

    titles = []
    for g in geopackages:
        titles.append(g)
    fig = make_subplots(rows=int(len(geopackages) / 2) + 1, cols=2, subplot_titles=titles)
    for index, values in enumerate(matrix):
        labels = labels_list[index]
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
    titles = []

    for g in geopackages:
        titles.append(g)
    fig = make_subplots(rows=int(len(geopackages) / 2) + 1, cols=2, subplot_titles=titles)
    for index, values in enumerate(matrix):
        labels = labels_list[index]

        if (order == 'Scores'):
            zipped = list(zip(list(labels), list(values)))
            temp = pd.DataFrame(data=zipped, columns=['Features', 'Scores'])
            temp.sort_values(by='Scores', axis=0, ascending=False, inplace=True, kind='quicksort')
            values = temp['Scores'].to_numpy()
            labels = temp['Features']

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

# method which returns labels of features which have no nan values
def check_NotNull(df):
    bool = df.isna()
    labels = []
    for (columnName, columnData) in bool.iteritems():
        if not columnNotNull(columnData):
            labels.append(columnName)

    return labels

def fs_results_computation(X, Y):
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
    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, Y)
    # get importance
    results['RF Importance'] = model.feature_importances_
    
    #Recursive Feature Selection
    results['RFS']= recursive_feature_selection(X, Y.astype(int), 20)
    return results


def variance_threshold(data, th):
    # define thresholds to check
    # thresholds = arange(0.0, 0.55, 0.05)
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
    
    support = rfe.support_
    
    res = []
    for s in support:
            if(s == True):
                res.append(1)
            else:
                res.append(0)
    
    results['Ranking'] = res
    return results['Ranking']

def set_labels_df(dictionary, keys):
    labels = dictionary[keys[0]]['Features'].tolist()
    for k in keys:
        if (len(dictionary[k]['Features'].tolist()) > len(labels)):
            labels = dictionary[k]['Features'].tolist()

    return labels
