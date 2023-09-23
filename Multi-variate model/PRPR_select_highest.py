import pandas as pd
import pathlib
import numpy as np
import os

target = 'pm25_int'
uid = 'fid'
fullrepo_path = pathlib.Path('Datasources\\timeframes_protocols\\All')
protocol_path = ('Datasources\\timeframes_protocols\\')
fullrepo_files = list(fullrepo_path.iterdir())
dbslist = list()
statsdb = pd.DataFrame()
outrow = 0
for frf in fullrepo_files:
    newdbraw = pd.read_csv(frf)
    splitpath = str(frf).split('\\')
    dbname = splitpath[len(splitpath)-1].replace('.csv','')
    newdb = newdbraw[[uid,target]]
    targarray = np.asarray(newdb[target])
    med = np.median(targarray)
    q75 = np.quantile(targarray,0.75)
    q90 = np.quantile(targarray,0.9)
    maxval = max(targarray)
    statsdb.loc[outrow,'WEEK'] = dbname
    statsdb.loc[statsdb['WEEK']==dbname,'MEDIAN'] = med
    statsdb.loc[statsdb['WEEK']==dbname,'3rdQUART'] = q75
    statsdb.loc[statsdb['WEEK']==dbname,'90thPERC'] = q90
    statsdb.loc[statsdb['WEEK']==dbname,'MAXVAL'] = maxval
    outrow = outrow + 1

params = list(statsdb.columns.values)
params.pop(params.index('WEEK'))
for p in params:
    q50thresh = np.quantile(np.asarray(statsdb.loc[:,p]),0.5)
    q75thresh = np.quantile(np.asarray(statsdb.loc[:,p]),0.75)
    q90thresh = np.quantile(np.asarray(statsdb.loc[:,p]),0.9)
    q50subs = statsdb.loc[statsdb[p]>=q50thresh].copy(deep=True)
    q75subs = statsdb.loc[statsdb[p]>=q75thresh].copy(deep=True)
    q90subs = statsdb.loc[statsdb[p]>=q90thresh].copy(deep=True)
    q50weeks = list(q50subs['WEEK'])
    q75weeks = list(q75subs['WEEK'])
    q90weeks = list(q90subs['WEEK'])
    q50folder = protocol_path+'HighestConc\\MEDIAN_'+p
    q75folder = protocol_path+'HighestConc\\3rdQUART_'+p
    q90folder = protocol_path+'HighestConc\\90thPERC_'+p
    if not os.path.isdir(q50folder):
        os.makedirs(q50folder)
    if not os.path.isdir(q75folder):
        os.makedirs(q75folder)
    if not os.path.isdir(q90folder):
        os.makedirs(q90folder)
    for q50 in q50weeks:
        newdb = pd.read_csv(str(fullrepo_path)+'\\'+q50+'.csv')
        newdb.to_csv(q50folder+'\\'+q50+'.csv',index=False)
    for q75 in q75weeks:
        newdb = pd.read_csv(str(fullrepo_path)+'\\'+q75+'.csv')
        newdb.to_csv(q75folder+'\\'+q75+'.csv',index=False)
    for q90 in q90weeks:
        newdb = pd.read_csv(str(fullrepo_path)+'\\'+q90+'.csv')
        newdb.to_csv(q90folder+'\\'+q90+'.csv',index=False)

br = 1