import os
import pandas as pd
from RSU_Z import *
from RSU_A1_preprocess_grid import *
from RSU_B_compute_correlations import *

params = input_parameters()
params.version = 'FV_PM25'
params.model = 'WT' # Chose among: WT = whole territory, WU = whole urban, UB = urban belt, UI = urban islands, UBUI = urban belt + urban islands
params.target = 'pm25_cams' # pm25_int, nh3_cams
params.dspath = 'datasources\\'
params.uid = 'fid'
params.areaf = 'area'
params.respath = 'Univariate\\results\\' + params.version + '_' + params.model + '\\' + params.target + '\\'
if not(os.path.isdir(params.respath)):
    os.makedirs(params.respath)

dspath = params.dspath
version = params.version
model = params.model

try:
    inputset = pd.read_csv(dspath+'DS_'+version+'_'+model+'_preprocessed_set.csv',encoding='ISO-8859-1')
except:
    inputset = preprocess_grid(params)

compute_correlations(params)

br = 1