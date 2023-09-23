import pandas as pd
from DDUST_multivariate_compute_target import *
from RSU_main import *
from MGWR_main import *
from RF_main import *

suffix = '_FV'
target_pollutant = 'pm25_cams' # pm25_cams, nh3_cams
spatial_protocols = ['WT','WU','UBUI'] # Chose among: WT = whole territory, WU = whole urban, UB = urban belt, UI = urban islands, UBUI = urban belt + urban islands
temporal_protocol = ['All','Spills','Nospills'] # Chose among 'All' = all available, 'Spills' = spills periods, March April October November, 'Nospills' = no spills periods, all others
target_modes = ['f','I','Ex'] #List all desired modes: 'f' for frequency, 'I' for intensity, 'Ex' for exposure

for spatial_protocol in spatial_protocols:
########################################## TARGET VALUE ##############################################
    for tempp in temporal_protocol:
        basestring = 'Version ' + suffix + ' protocol ' + spatial_protocol + '/' + tempp + ' for ' + target_pollutant
        try:
            target_values = pd.read_csv('Datasources\\workflow\\DS_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_target_values.csv')
        except:
            print(basestring+': target values not computed - computation starting')
            target_values = compute_targets(suffix,spatial_protocol,target_pollutant,basestring)

    ########################################## RANK-SUM FS ##############################################
        try:
            rsu_results = pd.read_csv('Outputs\\RSU_FS\\results\\'+suffix+'_'+spatial_protocol+'\\'+target_pollutant+'\\PearsonsR_'+target_pollutant+'.csv',low_memory=False)
        except:
            print(basestring,': rank-sum test for features selection not compute - computation starting')
            RSU_main_cycle(suffix,spatial_protocol,target_pollutant)
            rsu_results = pd.read_csv('Outputs\\RSU_FS\\results\\'+suffix+'_'+spatial_protocol+'\\'+target_pollutant+'\\PearsonsR_'+target_pollutant+'.csv',low_memory=False)


    ############################################## MGWR ###################################################
        try:
            reprojected_datasets = dict()
            for tm in target_modes:
                rpds = pd.read_csv('Datasources\\workflow\\MGWR_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_'+tm+'_reprojected_dataset.csv')
                reprojected_datasets[tm] = rpds
        except:
            print(basestring,': MGWR not computed - computation starting')
            MGWR_run(rsu_results,suffix,spatial_protocol,target_pollutant,tempp,target_modes)
            reprojected_datasets = dict()
            for tm in target_modes:
                rpds = pd.read_csv('Datasources\\workflow\\MGWR_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_'+tm+'_reprojected_dataset.csv')
                reprojected_datasets[tm] = rpds

    ############################################## RANDOM FOREST ###################################################
        for tm in target_modes:
            run_RF(suffix,spatial_protocol,tempp,target_pollutant,tm)

br = 1

