# D-DUST Project - WP2
<img style="margin-right:80px; margin-left:80px" src=img/DDUST__Nero.png width="150"> <img style="margin-right:80px" src=img/01_Polimi_centrato_BN_positivo.png width="150"> <img style="margin-right:80px" src=img/sigillo_testo_colori_300dpi.png width="150"> <img style="margin-right:80px" src=img/fondazione-politecnico-di-milano.png width="150"> <img src=img/ML_FCARIPLO_cmyk__base_100mm.png width="150">


---
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/opengeolab/D-DUST.git/WP2)

The WP2 of the D-DUST project focuses on the implementation of the data repository for the D-DUST project. The following notebooks have been developed for data preparation and preprocessing.

## Notebooks description

### Data request and preparation
These notebooks are used to require data (e.g. from API services), perform data preparation for following phases and export these data with an appropriate format.
- [**Model Variables Request Notebook**](https://github.com/opengeolab/D-DUST/blob/WP2/Model%20Variables%20Request.ipynb): access modelled air quality data from the [Copernicus Atmosphere Monitoring Service](https://atmosphere.copernicus.eu/data) API and allows to export them as NetCDF files. It also requests [Copernicus Climate Change Service](https://climate.copernicus.eu/) meteorological data using Google Earth Engine API (using also [geemap](https://geemap.org/) library), downloading them in .tif format. These are used as input in the grid processing notebook.
- [**Ground Sensor Variables Request**](https://github.com/gisgeolab/D-DUST/blob/WP2/Ground%20Sensor%20Variables%20Request%20.ipynb) : access [ARPA Lombardia](https://www.arpalombardia.it/Pages/ARPA_Home_Page.aspx) ground sensor and [ESA LPS Air Quality Stations](https://aqp.eo.esa.int/aqstation/) data for both meteorological and air quality stations. It allows to download meteorological and air quality data directly from the API if the current month or year are selected (additional notes about this topics are described in the notebook with more detail). If data of previous years are selected the corresponding .csv files from [Open Data Regione Lombardia](https://www.dati.lombardia.it/) are automatically considered. It allows to export a geopackage file for each sensor type and a .tif file with interpolated values over the bounding box region. These are used as input in the grid processing notebook.
- [**Satellite Variables Request**](https://github.com/opengeolab/D-DUST/blob/WP2/Satellite%20Variables%20Request.ipynb): access [Google Earth Engine API](https://developers.google.com/earth-engine/datasets) (using also [geemap](https://geemap.org/) library) to retrieve satellite data, mainly for air pollutants. It allows to export .tif files for each selected variable over the defined bounding box. These are used as input in the grid processing notebook.
- [**Date selection**](https://github.com/opengeolab/D-DUST/blob/WP2/Date%20selection.ipynb): used to select low precipitation and high temperature periods, favorable to manuring activities. It access to ARPA ground sensors data for Lombardy region (similarly to the ground sensor notebook) and calculates average precipitation and temperature for all sensors in a given month. At the end it provides a visualization of the time series. 

### Data processing
- [**Grid Processing**](https://github.com/gisgeolab/D-DUST/blob/WP2/Grid%20Processing.ipynb): this notebook allows to calculate the required statistic for each variable in each cell, aggregating all the data previously preprocessed and downloaded (for example, the mean value for pollutants in each cell over the time range or the density of roads in each cell).

## Setting up the environment

It is possible to set up an environment using Anaconda. Open Anaconda terminal to create an environment called `ddust`. To do this type: <br>
`$ conda create -n ddust`

Now activate the environment previously created: <br>
`$ conda activate ddust`

Install the necessary libraries required to run the notebooks for downloading the data and grid them:<br>
`$  python conda install -c conda-forge rasterio geopandas geemap xarray rioxarray rasterstats`

Finally, it is also required to install `cdsapi` to access Copernicus data and `sodapy` to access ARPA API: <br>
`$ pip install cdsapi`<br>
`$ pip install sodapy`

---

### Repository structure

```bash
root/ 
├── functions/                                    # Folder containing D-DUST functions
│   ├── DDUST_methods.py                          # DDUST functions
├── LICENSE                                       # FIX THIS AT THE END
├── README.md                                     # Readme file
├── environment.yml                               # Environment file
├── date.json                                     # JSON file used to define processing time range
├── kays.json                                     # JSON file used to store tokens and key for CDSAPI and ARPA Socrata API
├── Date selection.ipynb                          # Jupyter Notebook for selecting best time ranges
├── Grid Processing.ipynb                         # Jupyter Notebook for grid processing
├── Model Variables Request.ipynb                 # Jupyter Notebook for requesting model data
├── Satellite Variables Request.ipynb             # Jupyter Notebook for requesting satellite data
└── Ground Sensor Variables Request.ipynb         # Jupyter Notebook for requesting ground sensor data
