import pandas as pd
from MGWR_prepare_dataset import *
from MGWR_compute import *
from MGWR_reproject import *

def MGWR_run(rsu_results,suffix,spatial_protocol,target_pollutant,temporal_protocol,targets):

    tempp = temporal_protocol

    for t in targets:
        try:
            inputset = pd.read_csv('Datasources\\workflow\\MGWR_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_'+t+'_preprocessed_set.csv')
        except:
            MGWR_preprocess_set(rsu_results,suffix,target_pollutant,t,spatial_protocol)
            inputset = pd.read_csv('Datasources\\workflow\\MGWR_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_'+t+'_preprocessed_set.csv')
        try:
            contributions = pd.read_csv('Outputs\\MGWR\\' + suffix +'\\' + target_pollutant + '\\'+spatial_protocol+'_'+tempp+'_'+t+'_MGWR_contributions.csv')
            bandwidths = pd.read_csv('Outputs\\MGWR\\' + suffix +'\\' + target_pollutant + '\\'+spatial_protocol+'_'+tempp+'_'+t+'_MGWR_bandwidths.csv')
            wm = np.load('Outputs\\MGWR\\' + suffix +'\\' + target_pollutant + '\\'+spatial_protocol+'_'+tempp+'_'+t+'_weights.npy',allow_pickle=True)
        except:
            run_mgwr(suffix,target_pollutant,spatial_protocol,inputset,t,tempp)
            contributions = pd.read_csv('Outputs\\MGWR\\' + suffix +'\\' + target_pollutant + '\\'+spatial_protocol+'_'+tempp+'_'+t+'_MGWR_contributions.csv')
            bandwidths = pd.read_csv('Outputs\\MGWR\\' + suffix +'\\' + target_pollutant + '\\'+spatial_protocol+'_'+tempp+'_'+t+'_MGWR_bandwidths.csv')
            wm = np.load('Outputs\\MGWR\\' + suffix +'\\' + target_pollutant + '\\'+spatial_protocol+'_'+tempp+'_'+t+'_weights.npy',allow_pickle=True)

        try:
            reprojected_dataset = pd.read_csv('Datasources\\workflow\\MGWR_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_'+t+'_reprojected_dataset.csv')
        except:
            reprojected_dataset = reproject_with_mgwr(inputset,contributions,bandwidths,wm,suffix,target_pollutant,spatial_protocol,t,tempp)

    return(reprojected_dataset)