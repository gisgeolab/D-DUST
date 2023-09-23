import pandas as pd
import numpy as np

def MGWR_preprocess_set(rsu_results,suffix,target_pollutant,target_measure,spatial_protocol):

    rsu_R_thresh = 0.5
    uid = 'fid'

    inputset = pd.read_csv('Datasources\\workflow\\'+'DS_'+suffix+'_WT_preprocessed_set.csv',encoding='ISO-8859-1')
    baseset = pd.read_csv('Datasources\\workflow\\'+'DS_grid_time_invariant_UA.csv',encoding='ISO-8859-1')
    inputset.sort_values(by=uid,inplace=True)
    baseset.sort_values(by=uid,inplace=True)
    inputset['X'] = baseset['lng_cen']
    inputset['Y'] = baseset['lat_cen']

    allattributes = list(rsu_results.columns.values)
    baseatts = list()
    for att in allattributes:
        modflag = 0
        if 'SURR' in att:
            modatt = att.replace('_SURR','')
            modflag = 1
        if 'BLOCK' in att:
            modatt = att.replace('_BLOCK','')
            modflag = 1
        if modflag == 0:
            modatt = att
        if modatt not in baseatts:
            baseatts.append(modatt)
    baseatts.pop(baseatts.index('Protocol'))
    timeprotocols = (rsu_results['Protocol'].unique())
    for tp in timeprotocols:
        targetset = pd.read_csv('Datasources\\workflow\\DS_'+suffix+'_'+spatial_protocol+'_'+tp+'_'+target_pollutant+'_target_values.csv')
        inputset[target_measure] = targetset[target_measure]
        rsu_selfields = list()
        ressub = rsu_results.loc[rsu_results['Protocol']==tp]
        for ba in baseatts:
            selfields = list(filter(lambda x: ba in x, allattributes))
            subset = ressub[selfields]
            subind = subset.index.values[0]
            colmax = subset.idxmax(axis=1)[subind]
            maxval = subset[colmax].values[0]
            colmin = subset.idxmin(axis=1)[subind]
            minval = subset[colmin].values[0]
            if maxval >= rsu_R_thresh:
                rsu_selfields.append(colmax)
            else:
                if minval < 0 and abs(minval)>= rsu_R_thresh:
                    rsu_selfields.append(colmin)

        selected_inputset = inputset[[uid,'X','Y',target_measure]+rsu_selfields]
        scaled_inputset=pd.DataFrame()
        scaled_inputset[[uid,'X','Y',target_measure]] = selected_inputset[[uid,'X','Y',target_measure]]
        for sf in rsu_selfields:
            sfvec = np.asarray(selected_inputset[sf])
            maxval = max(sfvec)
            minval = min(sfvec)
            scaledvec = (sfvec-minval)/(maxval-minval)
            scaled_inputset[sf] = scaledvec

        scaled_inputset.to_csv('Datasources\\workflow\\MGWR_'+suffix+'_'+spatial_protocol+'_'+tp+'_'+target_pollutant+'_'+target_measure+'_preprocessed_set.csv',index=False)


        br = 1