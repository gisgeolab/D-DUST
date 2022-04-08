# D-DUST Project - WP2
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/opengeolab/D-DUST.git/WP2)

WP2 focuses on the implementation of the data repository for the D-DUST project. The following notebooks have been developed for data preparation and preprocessing.

## Notebooks description

### Data request and preparation
These notebooks are used to require data (e.g. from API services), perform data preparation for following phases and export these data with an appropriate format.
- [Model Variables Request Notebook](https://github.com/opengeolab/D-DUST/blob/WP2/Model%20Variables%20Request.ipynb): this notebook allows to access the [Copernicus Atmosphere Monitoring Service](https://atmosphere.copernicus.eu/data) API and allows to export NetCDF files to be used as input in the data processing notebook.
- [Ground Sensor Variables Request](https://github.com/opengeolab/D-DUST/blob/WP2/Ground%20Sensor%20Variables%20Request%20-%20ARPA%20Lombardia.ipynb) : this notebook access to [ARPA Lombardia](https://www.arpalombardia.it/Pages/ARPA_Home_Page.aspx) ground sensor and [ESA LPS Air Quality Stations](https://aqp.eo.esa.int/aqstation/) data for both meteorological and air quality stations. It is possible to download data directly from the API for the current year, while for previous years .csv files from [Open Data Regione Lombardia](https://www.dati.lombardia.it/) are automatically used. It allows to export a geopackage file for each sensor type or a .tif file with interpolated values. These are used as input in the data processing notebook.
- [Satellite Variables Request](https://github.com/opengeolab/D-DUST/blob/WP2/Satellite%20Variables%20Request.ipynb) : this notebook access to [Google Earth Engine API](https://developers.google.com/earth-engine/datasets) using also [geemap](https://geemap.org/) library to retrieve satellite data, mainly for air pollutants. It allows to export .tif files for each selected variable over the defined bounding box. These are used as input in the data processing notebook.

### Data processing
- [Data processing](https://github.com/opengeolab/D-DUST/blob/WP2/grid_processing.ipynb) : this notebook allows to calculate the required statistic for each variable in each cell. For example, it evaluates the mean concentration est

