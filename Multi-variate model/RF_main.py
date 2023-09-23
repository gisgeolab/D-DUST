import pandas as pd

from RF_prepare_inputset import *
from RF_modelling import *

def run_RF(suffix,spatial_protocol,temporal_protocol,target_pollutant,target_measure):

    tm = target_measure
    tempp = temporal_protocol

    try:
        preprocessed_ds = pd.read_csv('Datasources\\workflow\\RF_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_'+target_measure+'_preprocessed_dataset.csv')
    except:
        inputdata = pd.read_csv('Datasources\\workflow\\MGWR_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_'+tm+'_reprojected_dataset.csv')
        targetdata = pd.read_csv('Datasources\\workflow\\DS_'+suffix+'_'+spatial_protocol+'_'+tempp+'_'+target_pollutant+'_target_values.csv')
        preprocessed_ds = rf_preprocess(inputdata,targetdata,suffix,spatial_protocol,tempp,target_pollutant,tm)

    compute_rf(preprocessed_ds,suffix,spatial_protocol,tempp,target_pollutant,target_measure)

    br = 1