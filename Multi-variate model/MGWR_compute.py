import os
import pandas as pd
import numpy as np
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

def identify_collinearities(input_df):
    correlation_matrix = input_df.corr()
    threshold = 0.8
    collinear_columns = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                collinear_columns.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

    return collinear_columns

def get_most_frequent_column(collinear_columns, ordered_columns):
    column_counts = {}

    for column_pair in collinear_columns:
        for column in column_pair:
            if column in column_counts:
                column_counts[column] += 1
            else:
                column_counts[column] = 1

    most_frequent_column = max(column_counts, key=lambda x: (column_counts[x], ordered_columns.index(x)))
    return (most_frequent_column)

def run_mgwr(suffix,target_pollutant,spatial_protocol,inputset,target_measure,tempp):

    uid = 'fid'
    corrcoeffs = pd.read_csv('Outputs\\RSU_FS\\results\\'+suffix+'_'+spatial_protocol+'\\'+target_pollutant+'\\PearsonsR'+'_'+target_pollutant+'.csv',low_memory=False)
    tpcoeff = corrcoeffs.loc[corrcoeffs['Protocol']==tempp].copy(deep=True)
    tpcoeff.set_index('Protocol',drop=True,inplace=True)
    tpcoeff.drop(columns=['Unnamed: 0'],inplace=True)
    tpcoeff = tpcoeff.abs()
    sorted_colind_all = np.argsort(tpcoeff.values[0])[::-1]

    target = np.asarray(inputset[target_measure]).reshape((-1,1))
    coords = list(zip(inputset['X'],inputset['Y']))
    target = (target - target.mean(axis=0)) / target.std(axis=0)

    attributesdf = inputset.drop([uid,'X','Y',target_measure],axis=1).copy(deep=True)
    performed = 0
    while performed == 0:
        try:
            print('CHECKING COLLINEARITY')
            collinear_columns = identify_collinearities(attributesdf)
            if len(collinear_columns)>0:
                allcols = list(attributesdf.columns.values)
                sorted_colind = [i for i in sorted_colind_all if tpcoeff.columns[i] in allcols]
                sorted_cols = list(tpcoeff.columns[sorted_colind])
                print('IDENTIFY COLLINEARITY')
                dropcol = get_most_frequent_column(collinear_columns,sorted_cols)
                print('COLLINEARITY IDENTIFIED: DROPPING',dropcol)
                attributesdf.drop(columns={dropcol},inplace=True)
            allatts = list(attributesdf.columns.values)
            datavecs = list()
            for aa in allatts:
                datavec = np.asarray(attributesdf[aa]).reshape((-1,1))
                datavecs.append(datavec)
            attributes = np.hstack(datavecs)
            attributes = (attributes - attributes.mean(axis=0)) / attributes.std(axis=0)
            selector = Sel_BW(coords,target,attributes,multi=True)
            bw = selector.search(multi_bw_min=[1],verbose=True)
            mgwr_model = MGWR(coords,target,attributes,selector)
            mgwr_results = mgwr_model.fit()
            att_contributions = mgwr_results.params
            att_bandwidths = bw
            wm = mgwr_results.W
            performed = 1
        except:
            performed = 0
    outdb_contr = pd.DataFrame()
    outdb_contr[[uid,'X','Y',target_measure]] = inputset[[uid,'X','Y',target_measure]]
    outdb_bw = pd.DataFrame()
    atti = 0
    outdb_contr['Intercept'] = att_contributions[:,0]
    outdb_bw.loc['Intercept','Bandwidth'] = bw[0]
    for att in allatts:
        atti = atti + 1
        outdb_contr[att] = att_contributions[:,atti]
        outdb_bw.loc[att,'Bandwidth'] = bw[atti]
    outdb_bw.index.rename('ATTRIBUTE',inplace=True)

    outpath = 'Outputs\\MGWR\\' + suffix +'\\' + target_pollutant + '\\'
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    outdb_contr.to_csv(outpath+spatial_protocol+'_'+tempp+'_'+target_measure+'_MGWR_contributions.csv',index=False)
    outdb_bw.to_csv(outpath+spatial_protocol+'_'+tempp+'_'+target_measure+'_MGWR_bandwidths.csv')
    np.save(outpath+spatial_protocol+'_'+tempp+'_'+target_measure+'_weights.npy',wm,allow_pickle=True)

    br = 1