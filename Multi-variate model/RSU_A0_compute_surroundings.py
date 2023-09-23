import pandas as pd
import math
from RSU_Z import *

def compute_surroundings(params):
    dspath = params.dspath
    version = params.version
    model = 'WT'
    uid = params.uid
    areaf = params.areaf

    basegrid = pd.read_csv(dspath+'workflow\\'+'DS_grid_time_invariant_UA.csv')
    wholecellslist = list(basegrid[uid].unique())
    cellslist = wholecellslist
    distlim = 10000
    totcells = len(cellslist)
    ici = 0
    surrdb = pd.DataFrame()
    surrdb['SURR_ID'] = []
    for c in cellslist:
        ici = ici + 1
        print('Computing surroundings cells ',str(ici),'/',str(totcells),': processing = ',str((ici/totcells)*100),'%')
        xc = basegrid.loc[basegrid[uid]==c,'X_UTM'].values[0]
        yc = basegrid.loc[basegrid[uid]==c,'Y_UTM'].values[0]
        carea = basegrid.loc[basegrid[uid]==c,areaf].values[0]
        tempdb = basegrid.loc[basegrid[uid]!=c].copy(deep=True)
        for cc in wholecellslist:
            if cc != c:
                xcc = tempdb.loc[tempdb[uid]==cc,'X_UTM'].values[0]
                ycc = tempdb.loc[tempdb[uid]==cc,'Y_UTM'].values[0]
                ccarea = tempdb.loc[tempdb[uid]==cc,areaf].values[0]
                dist = math.sqrt(((xcc-xc)**2)+((ycc-yc)**2))
                tempdb.loc[tempdb[uid]==cc,'DIST'] = dist
                if dist<distlim:
                    try:
                        checkout = surrdb.loc[c,'SURR_ID']
                        surrdb.loc[c,'SURR_ID'] = checkout + ';' + str(cc)
                    except:
                        surrdb.loc[c,'SURR_ID'] = str(cc)
        surrlist = surrdb.loc[c,'SURR_ID']
        sclist = list()
        for sr in list(surrlist.split(';')):
            sc = int(sr)
            sclist.append(sc)
        totsurrarea = basegrid.loc[basegrid[uid].isin(sclist),areaf].sum()
        totblockarea = totsurrarea + carea
        surrdb.loc[c,'SURR_PB'] = str(round((carea/totblockarea)*1000)/1000)
        surrdb.loc[c,'SURR_PS'] = ''
        for sc in sclist:
            scarea = basegrid.loc[basegrid[uid]==sc,areaf].values[0]
            scpercs = scarea/totsurrarea
            scpercb = scarea/totblockarea
            pbcell = surrdb.loc[c,'SURR_PB']
            pscell = surrdb.loc[c,'SURR_PS']
            surrdb.loc[c,'SURR_PB'] = pbcell + ';' + str(round(scpercb*1000)/1000)
            surrdb.loc[c,'SURR_PS'] = pscell + ';' + str(round(scpercs*1000)/1000)

    surrdb.index.rename(uid,inplace=True)
    surrdb.to_csv(dspath+'workflow\\'+'DS_'+version+'_'+model+'_surroundings_map.csv')

    return(surrdb)