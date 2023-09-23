import pandas as pd
from RSU_Z import *
import pathlib
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mlp

mlp.use('TkAgg')

def compute_ranksum(files,uid,target):

    fi = 1
    cumdb = pd.DataFrame()
    cumdb[uid] = files[0][uid]
    maxv = files[0].shape[0]
    for f in files:
        subset = f[[uid,target]].copy(deep=True)
        subset.sort_values(target,inplace=True)
        subset.reset_index(inplace=True,drop=True)
        subset['RS'] = subset.index.values
        uidsuff = '_'+str(fi)
        cumdb = cumdb.join(subset.set_index(uid),on=[uid],rsuffix=uidsuff,how='inner')
        cumdb.drop(columns=[target],inplace=True)
        cumdb.rename(columns={'RS':'RS_'+str(fi)},inplace=True)
        fi = fi +1
    if fi > 2:
        cumdb['RS_CUM'] = cumdb.iloc[:,1:fi-1].sum(axis=1)
    else:
        cumdb['RS_CUM'] = cumdb['RS_1']
    rscumdb = cumdb[[uid,'RS_CUM']].copy(deep=True)
    rscumdb.sort_values('RS_CUM',inplace=True)
    rscumdb.reset_index(inplace=True,drop=True)
    rscumdb['RS_'+target] = rscumdb.index.values
    rscumdb.drop(columns=['RS_CUM'],inplace=True)
    rscumdb.sort_values(uid,inplace=True)
    rscumdb.reset_index(inplace=True,drop=True)
    scalevec = pd.DataFrame(rscumdb['RS_'+target])
    normalized_df=(scalevec-scalevec.min())/(scalevec.max()-scalevec.min())
    scaledrsdb = pd.DataFrame()
    scaledrsdb[uid] = rscumdb[uid]
    scaledrsdb['RS_'+target] = normalized_df['RS_'+target]

    return(scaledrsdb)

def compute_correlation(db,xname,yname,params,protocolname,savefig="foo"):
    if savefig =="foo":
        savefig = 0
    dspath = params.dspath
    version = params.version
    model = params.model
    uid = params.uid
    respath = params.respath
    xvec = np.asarray(db['X'])
    yvec = np.asarray(db['Y'])
    lincorr = scipy.stats.linregress(xvec, yvec)
    r, p = scipy.stats.pearsonr(xvec, yvec)
    if savefig!=0:
        slope=lincorr.slope
        intercept=lincorr.intercept
        line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.3f}'
        fig, ax = plt.subplots()
        ax.plot(xvec, yvec, linewidth=0, marker='s')
        ax.plot(xvec, intercept + slope * xvec, label=line)
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.legend(facecolor='white')
        #plt.show()
        figname = version+'_'+model+'_'+xname+'VS'+yname+'_'+protocolname
        plt.savefig(respath+figname+'.png')
        plt.close()
    return(r)

def compute_correlations(params):
    print('COMPUTING CORRELATIONS\n\n')
    dspath = params.dspath
    version = params.version
    model = params.model
    uid = params.uid
    target = params.target
    report = pd.DataFrame()
    inputset = pd.read_csv(dspath+'DS_'+version+'_'+model+'_preprocessed_set.csv',encoding='ISO-8859-1')
    inputcells = list(inputset[uid].unique())
    inputfields = list(inputset.columns.values)
    singlefields = list()
    ranksumset = pd.DataFrame()
    ranksumset[uid] = inputset[uid]
    attributeslist = list()
    for inf in inputfields:
        if inf != uid:
            field_rs = compute_ranksum([inputset],uid,inf)
            ranksumset = ranksumset.join(field_rs.set_index(uid),on=[uid],rsuffix=inf,how='inner')
            attribute = field_rs.columns.values[1]
            attributeslist.append(attribute)
            inf = inf.replace('_SURR','')
            inf = inf.replace('_BLOCK','')
            if inf not in singlefields:
                singlefields.append(inf)
    timeprotocolspath = 'Univariate//timeframes_protocols'
    outfolder = 'Univariate//results//'
    tprot_ospath = pathlib.Path(timeprotocolspath)
    tprot_list = list(tprot_ospath.iterdir())
    n_protocols = len(tprot_list)
    pi = 0
    for p in tprot_list:
        pi = pi + 1
        prot_strparts = str(p).split('\\')
        prot_name = prot_strparts[len(prot_strparts)-1]
        protstring = 'Computing correlations for protocol ' + prot_name + ' n ' + str(pi) + '/' + str(n_protocols)
        print(protstring)
        prot_groups = list(p.iterdir())
        n_groups = len(prot_groups)
        pgi = 0
        for pg in prot_groups:
            pgi = pgi + 1
            group_strparts = str(pg).split('\\')
            group_name = group_strparts[len(group_strparts)-1]
            groupstring = '\tGroup ' + group_name + ' n ' + str(pgi) + '/' + str(n_groups)
            print(protstring+groupstring)
            protocolname = prot_name + '_' + group_name
            gridslist = list()
            fileslist = list(pg.iterdir())
            for f in fileslist:
                fstr = str(f)
                loadgrid = pd.read_csv(fstr)
                gogrid = loadgrid.loc[loadgrid[uid].isin(inputcells)].copy(deep=True)
                gridslist.append(gogrid)
            target_ranksum = compute_ranksum(gridslist,uid,target)
            for att in attributeslist:
                corrdb = pd.DataFrame()
                corrdb[uid] = target_ranksum[uid]
                corrdb['X'] = ranksumset[att]
                corrdb['Y'] = target_ranksum['RS_'+target]
                attname = att.replace('RS_','')
                rvalue = compute_correlation(corrdb,attname,target,params,protocolname,1)
                report.loc[protocolname,attname] = rvalue

    report.to_csv(params.respath+'PearsonsR_'+target+'.csv',encoding='ISO-8859-1')