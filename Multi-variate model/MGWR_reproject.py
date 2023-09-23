import os
import pandas as pd
import numpy as np

def reproject_with_mgwr(inputset,contributions,bandwidths,wm,suffix,target_pollutant,spatial_protocol,tm,tempp):

    basestr = suffix + ' working on ' + spatial_protocol + '/' + tempp + '-' + tm + ' for ' + target_pollutant
    uid = 'fid'
    outdb = pd.DataFrame()
    outdb[uid] = inputset[uid]
    mgwr_attributes = list(bandwidths['ATTRIBUTE'].unique())
    mgwr_attributes.pop(mgwr_attributes.index('Intercept'))
    atti = 0
    cells = list(inputset.index.values)
    iti = 0
    totiters = len(cells)*len(mgwr_attributes)
    for att in mgwr_attributes:
        atti = atti + 1
        attwm = wm[atti]
        for c in cells:
            iti = iti + 1
            print(basestr+' - reprojecting values of ',att,'for cell',c+1,': processing = ',str((iti/totiters)*100),'%')
            cellwarr = attwm[c]
            non_zeros = {idx: val for idx,val in enumerate(cellwarr) if val != 0}
            cumval = 0
            cumw = 0
            for k in non_zeros.keys():
                w = non_zeros.get(k)
                val = inputset.loc[k,att]
                newval = w*val
                cumval = cumval + newval
                cumw = cumw + w
            rpval = cumval/cumw
            outdb.loc[c,att] = rpval

    outdb.to_csv('Datasources\\workflow\\MGWR_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_'+tm+'_reprojected_dataset.csv',index=False)
    return(outdb)