# D-DUST Project - WP2
---
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/opengeolab/D-DUST.git/WP2)

WP2 focuses on the implementation of the data repository for the D-DUST project. The following notebooks have been developed for data preparation and preprocessing.

## Notebooks description
---
### Data request and preparation
These notebooks are used to require data (e.g. from API services), perform data preparation for following phases and export these data with an appropriate format.
- [Model Variables Request Notebook](https://github.com/opengeolab/D-DUST/blob/WP2/Model%20Variables%20Request.ipynb): access modelled air quality data from the [Copernicus Atmosphere Monitoring Service](https://atmosphere.copernicus.eu/data) API and allows to export NetCDF files to be used as input in the data processing notebook. It also requests [Copernicus Climate Change Service](https://climate.copernicus.eu/) meteorological data using Google Earth Engine API (using also [geemap](https://geemap.org/) library), downloading them in .tif format.
- [Ground Sensor Variables Request](https://github.com/opengeolab/D-DUST/blob/WP2/Ground%20Sensor%20Variables%20Request%20-%20ARPA%20Lombardia.ipynb) : access [ARPA Lombardia](https://www.arpalombardia.it/Pages/ARPA_Home_Page.aspx) ground sensor and [ESA LPS Air Quality Stations](https://aqp.eo.esa.int/aqstation/) data for both meteorological and air quality stations. It is possible to download data directly from the API for the current year, while for previous years .csv files from [Open Data Regione Lombardia](https://www.dati.lombardia.it/) are automatically used. It allows to export a geopackage file for each sensor type and a .tif file with interpolated values over the bounding box region. These are used as input in the data processing notebook.
- [Satellite Variables Request](https://github.com/opengeolab/D-DUST/blob/WP2/Satellite%20Variables%20Request.ipynb) : access [Google Earth Engine API](https://developers.google.com/earth-engine/datasets) (using also [geemap](https://geemap.org/) library) to retrieve satellite data, mainly for air pollutants. It allows to export .tif files for each selected variable over the defined bounding box. These are used as input in the data processing notebook.
- [Date selection](https://github.com/opengeolab/D-DUST/blob/WP2/Date%20selection.ipynb) : used to select low precipitation and high temperature periods, favorable to manuring activities. Access to ARPA ground sensors data for Lombardy region and calculates average precipitation and temperature for all sensors in a given month. It provides a visualization of the time series. 
---
### Data processing
- [Data processing](https://github.com/opengeolab/D-DUST/blob/WP2/grid_processing.ipynb) : this notebook allows to calculate the required statistic for each variable in each cell, using the data previously prepared and downloaded (for example, the mean value for pollutants in each cell over the selected time range, the density of the population in each cell or the fraction for each soil/land use category in each cell etc.).
