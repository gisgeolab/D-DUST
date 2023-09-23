import os
import numpy as np
np.bool = np.bool_
import pandas as pd
import sklearn as skl
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import simps

def sample_df(indbin,k):
    indb = indbin.copy(deep=True)
    indices = np.random.choice(indb.shape[0],k,replace=False)
    sampled = indb.iloc[indices]
    sampled = sampled.sample(frac=1).reset_index(drop=True)
    return(sampled,indices)

def concat_shuffle(dflist):
    concatenated = pd.concat(dflist)
    shuffled = concatenated.sample(frac=1).reset_index(drop=True)
    return(shuffled)

def rf_implement(train_X,train_Y,test_X,test_Y,attnames,printstr):
    classifier_name = "RF"
    rf = RF(random_state=42)
    param_grid = {'n_estimators': [50, 100, 150],
                  'max_depth': [10, 20, 30]}
    clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1_macro')
    modelstring = "RandomForest"
    print(printstr,' - TRAINING')
    clf.fit(train_X,train_Y.ravel())
    print(printstr,' - TESTING')
    pred_prob=clf.predict_proba(test_X)
    roc_testy = pd.DataFrame({'REAL':test_Y})
    roc_testpredprob = pd.DataFrame({'PROB':pred_prob[:,1]})
    rocvals,auc = compute_roc(roc_testy,roc_testpredprob,0.01)
    optpars = clf.best_params_
    ne = optpars.get('n_estimators')
    md = optpars.get('max_depth')
    shclf = RF(n_estimators=ne,max_depth=md)
    shclf.fit(train_X,train_Y.ravel())
    shap_values = shap.TreeExplainer(shclf).shap_values(train_X)

    return(rocvals,auc,shap_values[1])

def compute_roc(yreal,yprob,step):

    rocvals = pd.DataFrame()
    for a in np.arange(0,1+step,step):
        yproc = yprob.copy(deep=True)
        yproc.loc[yproc['PROB'] >= a,'PRED'] = 1
        yproc.loc[yproc['PROB'] < a,'PRED'] = 0
        ycomp = pd.DataFrame()
        ycomp['REAL'] = yreal['REAL']
        ycomp['PRED'] = yproc['PRED']
        tp = ycomp.loc[(ycomp['REAL']==1)&(ycomp['PRED']==1)].shape[0]
        fp = ycomp.loc[(ycomp['REAL']==0)&(ycomp['PRED']==1)].shape[0]
        tn = ycomp.loc[(ycomp['REAL']==0)&(ycomp['PRED']==0)].shape[0]
        fn = ycomp.loc[(ycomp['REAL']==1)&(ycomp['PRED']==0)].shape[0]
        allrecs = tp + tn + tp + fn
        allactpos = tp + fn
        allactneg = tn + fp
        #allposcalls = fp + tp
        #llnegcalls = fn + tn
        tpperc = tp / allrecs
        tnperc = tn / allrecs
        fpperc = fp / allrecs
        fnperc = fn / allrecs
        sens = tp / allactpos
        spec = tn / allactneg
        revspec = 1 - spec
        #acc = (tp + tn) / allrecs
        rocvals.loc[a,'ALPHA'] = a
        rocvals.loc[a,'1-SPEC'] = revspec
        rocvals.loc[a,'SENS'] = sens
    rocvals['REVALPHA'] = 1- rocvals['ALPHA']
    rocvals.sort_values(by=['1-SPEC','SENS','REVALPHA'],inplace=True)
    rocvals.reset_index(inplace=True,drop=True)
    rocvals.drop(columns={'REVALPHA'},inplace=True)
    auc = 0
    for ind in list(rocvals.index.values):
        if ind>0:
            xi = rocvals.loc[ind-1,'1-SPEC']
            xf = rocvals.loc[ind,'1-SPEC']
            yi = rocvals.loc[ind-1,'SENS']
            yf = rocvals.loc[ind,'SENS']
            newarea = ((yf+yi) * (xf-xi))/2
            auc = auc + newarea
    return(rocvals,auc)

def compute_rf(inputset,suffix,spatial_protocol,tempp,target_pollutant,target_measure):

    ncycles = 10
    testfraction = 0.1
    genuid = 'fid'
    rfuid = 'rfuid'
    baseprint = suffix + ' ' + spatial_protocol + '/' + tempp + ' - ' + target_measure + ' (' + target_pollutant + '):'

    zerosdb = inputset.loc[inputset['Target'] == 0].copy(deep=True)
    nzeros = zerosdb.shape[0]
    onesdb = inputset.loc[inputset['Target'] == 1].copy(deep=True)
    nones = onesdb.shape[0]
    cutdim = min([nzeros,nones])
    shaplist = list()
    xlist = list()
    maproc = pd.DataFrame()
    aucvals = list()
    for cyc in range(ncycles):
        nc = cyc + 1
        cycprint = 'Computing RANDOM FOREST cycle ' + str(nc) + ' out of ' + str(ncycles)
        print(baseprint,cycprint)
        cyc_onesdb,oidx = sample_df(onesdb,cutdim)
        cyc_zerosdb,zidx = sample_df(zerosdb,cutdim)
        train_ones,toidx = sample_df(cyc_onesdb,round(cutdim*(1-testfraction)))
        train_zeros,tzidx = sample_df(cyc_zerosdb,round(cutdim*(1-testfraction)))
        test_ones = cyc_onesdb[~cyc_onesdb.index.isin(toidx)].copy(deep=True)
        test_zeros = cyc_zerosdb[~cyc_zerosdb.index.isin(tzidx)].copy(deep=True)
        trainset = concat_shuffle([train_ones,train_zeros])
        testset = concat_shuffle([test_ones,test_zeros])

        train_uids = pd.DataFrame()
        test_uids = pd.DataFrame()
        train_uids[[rfuid,genuid]] = trainset[[rfuid,genuid]].copy(deep=True)
        test_uids[[rfuid,genuid]] = testset[[rfuid,genuid]].copy(deep=True)
        train_Y = np.asarray(trainset['Target'])
        test_Y = np.asarray(testset['Target'])
        attnames = list(trainset.columns.difference([rfuid,genuid,'Target']))
        train_X = np.asarray(trainset[attnames])
        test_X = np.asarray(testset[attnames])

        rocvals,auc,shapv = rf_implement(train_X,train_Y,test_X,test_Y,attnames,baseprint+cycprint)
        rocvals.sort_index(inplace=True)
        if maproc.empty:
            maproc = rocvals.copy(deep=True)
            maproc.rename({'ALPHA':str(nc)+'_ALPHA'},inplace=True)
            maproc.rename({'1-SPEC':str(nc)+'_1-SPEC'},inplace=True)
            maproc.rename({'SENS':str(nc)+'_SENS'},inplace=True)
        else:
            maproc[str(nc)+'_ALPHA'] = rocvals['ALPHA']
            maproc[str(nc)+'_1-SPEC'] = rocvals['1-SPEC']
            maproc[str(nc)+'_SENS'] = rocvals['SENS']
        aucvals.append(auc)
        shaplist.append(shapv)
        xlist.append(train_X)

    aucmed = np.median(aucvals)
    aucli = np.quantile(aucvals,0.025)
    aucui = np.quantile(aucvals,0.975)
    auctext = str((round(aucmed*1000))/1000) + ' [' + str((round(aucli*1000))/1000) + '-' + str((round(aucui*1000))/1000) +']'
    plottext = baseprint.replace(suffix,'') + '--  AUC = ' + auctext
    all_shap_values = np.vstack(shaplist)
    all_train_X = np.vstack(xlist)
    matplotlib.use('Agg')
    shap.summary_plot(all_shap_values,all_train_X,feature_names=attnames)
    axes = plt.gca()
    axes.set_title(plottext,fontsize=12)
    outpath = 'Outputs\\RF\\' + suffix + '\\' + target_pollutant + '\\SHAP\\'
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    outname = spatial_protocol + '_' + tempp + '_' + target_measure + '_SHAP_VALUES.png'
    plt.savefig(outpath+outname)
    plt.close()

    br = 1