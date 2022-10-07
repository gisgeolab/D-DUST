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
```bash
root/ 
├── README.md                                     # Readme file
├── notebook/                                     # Folder containing notebooks implemented in my work thesis
    ├──fs_results.ipynb                           # notebook used for computing Feature Selection
    ├──Keras_prediction_model.ipynb               # notebook used for building Keras Neural Network model for prediction
    ├──RandomForest_prediction_model.ipynb        # notebook used for building Random Forest Model for prediction
    ├──model.ipynb                                # notebook used for showing results achieved by ML models
    ├──buffer_knn.ipynb                           # notebook used for showing how KNN is performed in order to increase the size of ARPA observation
    ├──fs/                                        # It contains package in which I collect method implemented and used by the notebooks
    |   ├──methods.py
    |   ├──_init_.py
    ├──assets/    
        ├──fs_results/                            # It contains the fs results already run and saved in .csv files
        ├──test/                                  # It contains the test results using fs  saved in .csv and .xlsx files        
        ├──test_random_selection/                 # It contains the test results without using fs  saved in .csv and .xlsx files  
        ├──grids_0_1/                             # It contains the grids used in my case of study at 0.1° resolution
        ├──grids_0_01/                            # It contains the grids used in my case of study at 0.01° resolution
        ├──grids_0_066/                           # It contains the grids used in my case of study at 0.066 resolution
        ├──images/                                # It contains images used in this branch
