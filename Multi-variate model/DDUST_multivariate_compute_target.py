import pandas as pd
import os
import pathlib
import numpy as np
import scipy.stats as stats

def get_threshold(columns):
    for c in columns:
        if 'TH' in c:
            splitted = c.split('=')
            th = float(splitted[1])
            thcol = c
    return(thcol,th)

def compute_multipliers(indbraw,att,uid,protocol,zindex):
    allvalues = np.asarray(indbraw[att])
    wholemed = np.median(allvalues)
    iqr = np.quantile(allvalues,0.75)-np.quantile(allvalues,0.25)
    outliers_db = indbraw.loc[(indbraw[att]>((zindex*iqr)+wholemed))&(indbraw[att]<(wholemed-(zindex*iqr)))]
    if outliers_db.shape[0]>0:
        outliers = list(outliers_db[uid].unique())
        indb = indbraw.loc[not(indbraw[uid].isin(outliers))].copy(deep=True)
    else:
        indb = indbraw.copy(deep=True)
    cells = list(indb[uid].unique())
    totiters = len(cells)
    iti = 0
    uncert_db = indb.loc[indb[att+'_class']==2].copy(deep=True)
    alluncvec = np.asarray(uncert_db[att])
    medval = np.median(np.asarray(uncert_db[att]))
    allover_db = indb.loc[indb[att]>=medval].copy(deep=True)
    allover = list(allover_db[uid].unique())
    allovervec = np.asarray(allover_db[att])
    allunder_db = indb.loc[indb[att]<medval].copy(deep=True)
    allunder = list(allunder_db[uid].unique())
    allundervec = np.asarray(allunder_db[att])
    overunc_db = allover_db.loc[indb[att+'_class']==2].copy(deep=True)
    overunc = list(overunc_db[uid].unique())
    underunc_db = allunder_db.loc[indb[att+'_class']==2].copy(deep=True)
    underunc = list(underunc_db[uid].unique())
    over_db = allover_db.loc[indb[att+'_class']!=2].copy(deep=True)
    over = list(over_db[uid].unique())
    under_db = allunder_db.loc[indb[att+'_class']!=2].copy(deep=True)
    under = list(under_db[uid].unique())
    iqrup = np.quantile(allovervec,0.75)-np.quantile(allovervec,0.25)
    iqrlo = np.quantile(allundervec,0.75)-np.quantile(allundervec,0.25)
    if iqrlo == 0:
        iqrlo = np.std(allundervec)
    for c in cells:
        iti = iti + 1
        print('Protocol',protocol,': computing multipliers for attribute ',att,'processing = ',str((iti/totiters)*100),'%')
        v = indb.loc[indb[uid]==c,att].values[0]
        uflag = 0
        if c in allover:
            nrep = round((abs(v-medval) / iqrup) * 10) + 1
            if c in over:
                indb.loc[indb[uid]==c,att+'_Nones'] = nrep
                indb.loc[indb[uid]==c,att+'_Nzers'] = 0
            else:
                uflag = 1
        if c in allunder:
            nrep = round((abs(v-medval) / iqrlo) * 10) + 1
            if c in under:
                indb.loc[indb[uid]==c,att+'_Nones'] = 0
                indb.loc[indb[uid]==c,att+'_Nzers'] = nrep
            else:
                uflag = 1
        if uflag == 1:
            q = stats.percentileofscore(alluncvec,v)/100
            nones = round(nrep*q)
            indb.loc[indb[uid]==c,att+'_Nones'] = nones
            indb.loc[indb[uid]==c,att+'_Nzers'] = nrep-nones

    return(indb)


def compute_targets(suffix,spatial_protocol,target_pollutant,basestring):

    uid = 'fid'
    tp = target_pollutant
    print(basestring,': computing target values')
    thresholds_map = pd.read_excel('Datasources\\workflow\\DS_target_thresholds.xlsx',sheet_name=target_pollutant)
    classcol,thresh = get_threshold(list(thresholds_map.columns.values))
    #spatial_map = pd.read_excel('Datasources\\workflow\\DS_territorial_protocols.xlsx')
    #cellslist = list(spatial_map[spatial_protocol])
    timeprotocolspath = 'Datasources\\timeframes_protocols\\'
    fth = thresholds_map[[classcol,'f']].copy(deep=True)
    Ith = thresholds_map[[classcol,'I']].copy(deep=True)
    Exth = thresholds_map[[classcol,'Ex']].copy(deep=True)
    tprot_ospath = pathlib.Path(timeprotocolspath)
    tprot_list = list(tprot_ospath.iterdir())
    n_protocols = len(tprot_list)
    pi = 0
    for p in tprot_list:
        pi = pi + 1
        prot_strparts = str(p).split('\\')
        prot_name = prot_strparts[len(prot_strparts)-1]
        prot_cont= list(p.iterdir())
        prot_groups = list()
        for pc in prot_cont:
            if os.path.isdir(pc):
                prot_groups.append(pc)
        if len(prot_groups)==0:
            prot_groups.append(p)
        n_groups = len(prot_groups)
        pgi = 0
        for pg in prot_groups:
            pgi = pgi + 1
            group_strparts = str(pg).split('\\')
            group_name = group_strparts[len(group_strparts)-1]
            groupstring = '\tGroup ' + group_name + ' n ' + str(pgi) + '/' + str(n_groups)
            if prot_name != group_name:
                protocolname = prot_name + '_' + group_name
            else:
                protocolname = prot_name
            gridslist = list()
            fileslist = list(pg.iterdir())
            for f in fileslist:
                fstr = str(f)
                loadgrid = pd.read_csv(fstr)
                gogrid = loadgrid[[uid,target_pollutant]].copy(deep=True)
                gridslist.append(gogrid)
            finalgrid = pd.concat(gridslist)

            pgoutdb = pd.DataFrame()
            cells = finalgrid[uid].unique()
            totiters = len(cells)
            iti = 0
            for c in cells:
                iti = iti+1
                print('Protocol ',protocolname,': computing target on cell',c,' processing = ',str((iti/totiters)*100),'%')
                subset = finalgrid.loc[finalgrid[uid]==c].copy(deep=True)
                ovth = subset.loc[subset[tp]>=thresh].copy(deep=True)
                if ovth.shape[0]>0:
                    f = ovth.shape[0]/subset.shape[0]
                    ovth['DIFF'] = ovth[tp]-thresh
                    ovth['RELDIFF'] = ovth['DIFF']/thresh
                    reldiff = np.asarray(ovth['RELDIFF'])
                    #iabs = np.mean(np.asarray(ovth['DIFF']))
                    #I = iabs/thresh
                    I = np.quantile(reldiff,0.75)
                    Ex = f*I
                    fthsub = fth.loc[fth['f']<=f]
                    Ithsub = Ith.loc[Ith['I']<=I]
                    Exthsub = Exth.loc[Exth['Ex']<=Ex]
                    fclass = fth[classcol].values[fthsub.shape[0]]
                    Iclass = Ith[classcol].values[Ithsub.shape[0]]
                    Exclass = Exth[classcol].values[Exthsub.shape[0]]
                else:
                    f = 0
                    I = 0
                    Ex = 0
                    fclass = 0
                    Iclass = 0
                    Exclass = 0
                pgoutdb.loc[c,'f'] = f
                pgoutdb.loc[c,'f_class'] = fclass
                pgoutdb.loc[c,'I'] = I
                pgoutdb.loc[c,'I_class'] = Iclass
                pgoutdb.loc[c,'Ex'] = Ex
                pgoutdb.loc[c,'Ex_class'] = Exclass

            pgoutdb[uid] = pgoutdb.index.values
            pgoutdb.sort_values(by=uid)
            allatts = ['f','I','Ex']
            for att in allatts:
                mdb = compute_multipliers(pgoutdb,att,uid,spatial_protocol,5)
                mdb.sort_values(by=uid)
                pgoutdb[att+'_Nones'] = mdb[att+'_Nones']
                pgoutdb[att+'_Nzeros'] = mdb[att+'_Nzers']
            pgoutdb[uid] = pgoutdb.index
            pgoutdb.drop(columns=[uid],inplace=True)
            pgoutdb.index.rename(uid,inplace=True)
            pgoutdb.to_csv('Datasources\\workflow\\DS_'+suffix+'_'+spatial_protocol+'_'+protocolname+'_'+target_pollutant+'_target_values.csv')

    return(pgoutdb)

    br = 1