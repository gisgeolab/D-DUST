# D-DUST
In this branch you can find each tool used for the *feature selection* and *ML models*. 


## Feature Selection
In order to evaluate a weighted score for each variable [this interactive notebook](https://github.com/opengeolab/D-DUST/blob/thesis_MB/notebooks/fs_results.ipynb) is implemented.

## ML Models
The results of the feature selection (which are in the folder [assets/fs_results](https://github.com/opengeolab/D-DUST/tree/thesis_MB/notebooks/assets/fs_results)) are used for building 2 models:
- *[Keras prediction model](https://github.com/opengeolab/D-DUST/blob/thesis_MB/notebooks/Keras_prediction_model.ipynb)*: Prediction model based on neural networks regression;
- *[Random Forest prediction model](https://github.com/opengeolab/D-DUST/blob/thesis_MB/notebooks/RandomForest_prediction_model.ipynb)*: Prediction model based on Random Forest regression;

Results of ML models already computed can be view in this [notebook](https://github.com/opengeolab/D-DUST/blob/thesis_MB/notebooks/model.ipynb), by selecting the model, period and configuration.
