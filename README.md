# D-DUST Project - WP2
<img style="margin-right:80px; margin-left:80px" src=img/DDUST__Nero.png width="150"> 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/opengeolab/D-DUST.git/WP2)

The WP2 of the D-DUST project focuses on the implementation of the Analysis-ready Data Repository for the D-DUST project. The following notebooks have been developed for data preparation and preprocessing.

## Notebooks description

### Data request and preparation
These notebooks are used to require data (e.g. from API services), perform data preparation for following phases and export these data with an appropriate format.
- [**Model Variables Request Notebook**](https://github.com/opengeolab/D-DUST/blob/WP2/Model%20Variables%20Request.ipynb): this notebook is used to access modelled air quality data from the [Copernicus Atmosphere Monitoring Service](https://atmosphere.copernicus.eu/data) API and allows to export them as *NetCDF* files. It also requests [Copernicus Climate Change Service](https://climate.copernicus.eu/) meteorological data using Google Earth Engine API (using also [geemap](https://geemap.org/) library), downloading them in *.tif* format. These are used as input in the **Grid processing** notebook.
- [**Ground Sensor Variables Request**](https://github.com/gisgeolab/D-DUST/blob/WP2/Ground%20Sensor%20Variables%20Request%20.ipynb) : this notebook is used to access [ARPA Lombardia](https://www.arpalombardia.it/Pages/ARPA_Home_Page.aspx) ground sensor and [ESA LPS Air Quality Stations](https://aqp.eo.esa.int/aqstation/) data for both meteorological and air quality stations. It allows downloading meteorological and air quality data directly from the API if the current month or year is selected (additional notes about these topics are described in the notebook comments in detail). If data from previous years are selected, the corresponding *.csv* files from [Open Data Regione Lombardia](https://www.dati.lombardia.it/) are automatically considered. It allows exporting a *geopackage* file for each sensor type and a *.tif* file with interpolated values over the bounding box region. These are used as input in the **Grid processing** notebook.
- [**Satellite Variables Request**](https://github.com/opengeolab/D-DUST/blob/WP2/Satellite%20Variables%20Request.ipynb): this notebook is used to access [Google Earth Engine API](https://developers.google.com/earth-engine/datasets) (by using also [geemap](https://geemap.org/) library) to retrieve, mainly, satellite observations of athmospheric pollutants. It allows exporting a *.tif* file for each selected variable over a defined bounding box. These files are used as input in the **Grid processing** notebook.
- [**Date selection**](https://github.com/opengeolab/D-DUST/blob/WP2/Date%20selection.ipynb): this notebook is used to select low precipitation and high-temperature periods, favourable to manuring activities. It accesses ARPA ground sensors data for the Lombardy region (similarly to the **Ground Sensor Variables Request** notebook) and calculates average precipitation and temperature for all sensors in a given month. Results are visualised as a time series.
### Data processing
- [**Grid Processing**](https://github.com/gisgeolab/D-DUST/blob/WP2/Grid%20Processing.ipynb): this notebook allows computing summary statistics for each variable in each grid cell, aggregating all the data previously downloaded and preprocessed (e.g., the mean value for pollutants in each cell over a time range or the density of roads in each cell).


## Environment setup

It is possible to set up a virtual Python environment using [Anaconda](https://anaconda.org). Open the **Anaconda terminal** and create an environment called `ddust`. To do this type: <br>
```sh
$ conda create -n ddust
```

Activate the `ddust` envionment: <br>
```sh
$ conda activate ddust
```

Install the required libraries to run the notebooks for data downloading and aggregation on a user-defined grid:<br>
```sh
$  python conda install -c conda-forge rasterio geopandas geemap xarray rioxarray rasterstats
```

Additionally, it is necessary to install `cdsapi` to access Copernicus data and `sodapy` to access ARPA API: <br>
```sh
$ pip install cdsapi
$ pip install sodapy
```

## Repository structure

```bash
root/ 
├── functions/                                    # Folder containing D-DUST functions
│   ├── DDUST_methods.py                          # DDUST functions
├── LICENSE                                       # FIX THIS AT THE END
├── README.md                                     # Readme file
├── environment.yml                               # Environment file
├── date.json                                     # JSON file used to define the processing time range
├── kays.json                                     # JSON file used to store tokens and key for CDSAPI and ARPA Socrata API
├── Date selection.ipynb                          # Jupyter Notebook for selecting best time ranges
├── Grid Processing.ipynb                         # Jupyter Notebook for grid processing
├── Model Variables Request.ipynb                 # Jupyter Notebook for requesting model data
├── Satellite Variables Request.ipynb             # Jupyter Notebook for requesting satellite data
└── Ground Sensor Variables Request.ipynb         # Jupyter Notebook for requesting ground sensor data
