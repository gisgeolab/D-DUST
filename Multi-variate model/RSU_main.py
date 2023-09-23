import os
import pandas as pd
from RSU_Z import *
from RSU_A1_preprocess_grid import *
from RSU_B_compute_correlations import *

def RSU_main_cycle(version,model,target):
    params = input_parameters()
    params.version = version
    params.model = model
    params.target = target
    params.dspath = 'Datasources\\'
    params.uid = 'fid'
    params.areaf = 'area'
    params.respath = 'Outputs\\RSU_FS\\results\\' + params.version + '_' + params.model + '\\' + params.target + '\\'
    if not(os.path.isdir(params.respath)):
        os.makedirs(params.respath)

    dspath = params.dspath
    version = params.version
    model = params.model

    try:
        inputset = pd.read_csv(dspath+'workflow\\'+'DS_'+version+'_'+model+'_preprocessed_set.csv',encoding='ISO-8859-1')
    except:
        inputset = preprocess_grid(params)

    compute_correlations(params)

br = 1