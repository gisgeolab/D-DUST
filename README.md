# D-DUST
In this branch you can find each tool used for the *feature selection* and *ML models*. 


## Feature Selection
In order to evaluate a weighted score for each variable [fs_results.ipynb](https://github.com/opengeolab/D-DUST/blob/thesis_MB/notebooks/fs_results.ipynb) is implemented. 

## ML Models
The results of the feature selection (which are in the folder [assets/fs_results](https://github.com/opengeolab/D-DUST/tree/thesis_MB/notebooks/assets/fs_results)) are used for building 2 models. Each of them are implemented in these notebooks:
- *[Keras prediction model](https://github.com/opengeolab/D-DUST/blob/thesis_MB/notebooks/Keras_prediction_model.ipynb)*: Prediction model based on neural networks regression;
- *[Random Forest prediction model](https://github.com/opengeolab/D-DUST/blob/thesis_MB/notebooks/RandomForest_prediction_model.ipynb)*: Prediction model based on Random Forest regression;


Results of ML models already computed can be view in this [notebook](https://github.com/opengeolab/D-DUST/blob/thesis_MB/notebooks/model.ipynb), by selecting the model, period and configuration.


An overview of how input and output of the different notebooks are used is displayed.
</br>
</br>
<img width="1200" src = notebooks/assets/images/overview.png>

### Structure of folders
root/ 
├── functions/                                    # Folder containing D-DUST functions
│   ├── DDUST_methods.py                          # DDUST functions
├── LICENSE                                       # FIX THIS AT THE END
├── README.md                                     # Readme file
├── environment.yml                               # Environment file
├── date.json                                     # JSON file used to define the processing time range
├── keys.json                                     # JSON file used to store tokens and key for CDSAPI and ARPA Socrata API
├── Date selection.ipynb                          # Jupyter Notebook for selecting best time ranges
├── Grid Processing.ipynb                         # Jupyter Notebook for grid processing
├── Model Variables Request.ipynb                 # Jupyter Notebook for requesting model data
├── Satellite Variables Request.ipynb             # Jupyter Notebook for requesting satellite data
└── Ground Sensor Variables Request.ipynb         # Jupyter Notebook for requesting ground sensor data
