# D-DUST
---
### Notebook description
- ADS_API_Copernicus https://hub.gke2.mybinder.org/user/opengeolab-d-dust-mtdhys9b/lab/tree/AQ_ESA_Stations.ipynb: Access to [Copernicus Atmosphere Monitoring Service](https://atmosphere.copernicus.eu/data) API and allows to export NetCDF files to be used as input in the grid_processing notebook.
- ARPA_API_ground_sensors: Access to ARPA Ground Sensors for both meteorological and air quality stations. It allows to export a geopackage file for each sensor type. It is possible to download data directly from the API for the current year. For previous years .csv files from [Open Data Regione Lombardia](https://www.dati.lombardia.it/) are used.
- GEE_API: Access to [Google Earth Engine API](https://developers.google.com/earth-engine/datasets) using [geemap](https://geemap.org/) library to retrieve satellite data. It allows to export .tif files for each selected variable.
- AQ_ESA_Stations: access to ESA Air Quality stations API and downloads data for each station in the time range.
- grid_processing: it allows to calculate the mean value in each cell for all the selected variables.

