import pandas as pd
import numpy as np

def rescale_db(indb,scalecols):

    scaled = indb.copy()

    for sc in scalecols:
        scmin = scaled[sc].min()
        scmax = scaled[sc].max()
        scaled[sc] = (indb[sc] - scmin) / (scmax - scmin)

    return(scaled)

def generate_ds(onescol,zeroscol,dataset,printstr):

    uid = 'fid'
    noise = 0.1
    totiters = dataset.shape[0]
    iti = 0
    rowlist = list()
    for ri,row in dataset.iterrows():
        iti = iti + 1
        print(printstr,'pre-processing dataset for RF (',str((iti/totiters)*100),'%)')
        nones = int(onescol[ri])
        nzeros = int(zeroscol[ri])
        for o in range(nones):
            newrow = row.to_frame().transpose()
            newrow.loc[:,'Target'] = 1
            rowlist.append(newrow)
        for z in range(nzeros):
            newrow = row.to_frame().transpose()
            newrow.loc[:,'Target'] = 0
            rowlist.append(newrow)

    outdb = pd.concat(rowlist)
    outdb.reset_index(inplace=True,drop=True)
    attcols = outdb.columns.difference([uid,'Target'])
    outdb[attcols] *= np.random.uniform((1-(1*noise)),(1+(1*noise)),size=(outdb.shape[0],len(attcols)))
    outdb['rfuid'] = range(outdb.shape[0])
    attlist = ['rfuid',uid,'Target'] + list(attcols)
    ordoutdb = pd.DataFrame(columns=attlist)
    ordoutdb[attlist] = outdb[attlist]
    scaledout = rescale_db(ordoutdb,attcols)
    return(scaledout)

def rf_preprocess(inputdata,targetdata,suffix,spatial_protocol,tempp,target_pollutant,target_measure):

    printstr = suffix + ' ' + spatial_protocol + '/' + tempp + ' - ' + target_measure + ' (' + target_pollutant + '):'
    onescol = targetdata[target_measure+'_Nones']
    zeroscol = targetdata[target_measure+'_Nzeros']
    preprocessed_ds = generate_ds(onescol,zeroscol,inputdata,printstr)
    preprocessed_ds.to_csv('Datasources\\workflow\\RF_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_'+target_measure+'_preprocessed_dataset.csv',index=False)
    return (preprocessed_ds)