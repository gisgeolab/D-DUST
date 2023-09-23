import pandas as pd
from RSU_A0_compute_surroundings import *
from RSU_Z import *

def preprocess_grid(params):
    dspath = params.dspath
    version = params.version
    model = params.model
    uid = params.uid

    basegrid = pd.read_csv(dspath+'DS_grid_time_invariant_UA.csv')

    try:
        preprocdb = pd.read_csv(dspath+'DS_WT_preprocessed_set.csv')
    except:
        try:
            surrmap = pd.read_csv(dspath+'DS_WT_surroundings_map.csv')
        except:
            surrmap = compute_surroundings(params)
        mapfields = list(surrmap.columns.values)
        if uid not in mapfields:
            surrmap[uid] = surrmap.index.values
        preprocmap = pd.read_excel(dspath+'DS_preprocessing_map.xlsx')
        preprocdb = pd.DataFrame()
        cellslist = list(basegrid[uid].unique())
        totcells = len(cellslist)
        preprocdb[uid] = basegrid[uid]
        npreprocfields = preprocmap.shape[0]
        ppi = 0

        for r in preprocmap.iterrows():
            ppi = ppi + 1
            outfield = r[1]['field_new_name']
            ppstr = 'Pre-processing field ' + outfield + ' field ' + str(ppi) + '/' + str(npreprocfields)
            print(ppstr)
            procfields = r[1]['field_old_name'].split(',')
            nfields = len(procfields)
            if nfields == 1:
                getfield = procfields[0]
                preprocdb[outfield] = basegrid[getfield]
                searchvalsdb = basegrid.copy(deep=True)
            else:
                op = r[1]['operation']
                if op == 'sum':
                    tempdb = preprocdb.copy(deep=True)
                    tempdb['SUM'] = [0]*tempdb.shape[0]
                    for sf in procfields:
                        gsf = sf.replace(' ','')
                        tempdb['SUM'] = tempdb['SUM'] + basegrid[gsf]
                preprocdb[outfield] = tempdb['SUM']
                getfield = outfield
                searchvalsdb = preprocdb.copy(deep=True)
            ici = 0
            for c in cellslist:
                print(ppstr,'on cell',str(ici),'/',str(totcells),': processing = ',str((ici/totcells)*100),'%')
                surrcells = list(surrmap.loc[surrmap[uid]==c,'SURR_ID'].values[0].split(';'))
                surrval = 0
                surrweights = list(surrmap.loc[surrmap[uid]==c,'SURR_PS'].values[0].split(';'))
                blockweights = list(surrmap.loc[surrmap[uid]==c,'SURR_PB'].values[0].split(';'))
                blockval = (searchvalsdb.loc[searchvalsdb[uid]==c,getfield].values[0]) * float(blockweights[0])
                sci = 1
                for isc in surrcells:
                    sc = int(isc)
                    surrnewval = (searchvalsdb.loc[searchvalsdb[uid]==sc,getfield].values[0])* float(surrweights[sci])
                    surrval = surrval + surrnewval
                    blocknewval = (searchvalsdb.loc[searchvalsdb[uid]==sc,getfield].values[0])* float(blockweights[sci])
                    blockval = blockval + blocknewval
                    sci = sci + 1
                preprocdb.loc[preprocdb[uid]==c,outfield+'_SURR'] = surrval
                preprocdb.loc[preprocdb[uid]==c,outfield+'_BLOCK'] = blockval
                ici = ici + 1
        preprocdb.to_csv(dspath+'DS_WT_preprocessed_set.csv')

    if model == 'WT':
        outcellslist = list(basegrid[uid].unique())
    elif model == 'UB':
        outcellslist = [77,96,113,132,152,175,201,230,253,254,299,298,319,339,359,401,421,441,485,484,509,535,561]
    elif model == 'UI':
        outcellslist = [74,149,272,434,637,705]
    elif model == 'UBUI':
        outcellslist = [74,149,272,434,637,705,77,96,113,132,152,175,201,230,253,254,299,298,319,339,359,401,421,441,485,484,509,535,561]
    elif model == 'WU':
        outcellslist = list(basegrid.loc[basegrid['urban']==1,uid].unique())

    outdb = preprocdb.loc[preprocdb[uid].isin(outcellslist)].copy(deep=True)

    outdb.to_csv(dspath+'DS_'+version+'_'+model+'_preprocessed_set.csv',encoding='ISO-8859-1',index=False)

    return()