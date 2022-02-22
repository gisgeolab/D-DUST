# D-DUST
---
### Notebook description
- ADS_API_Copernicus: Access to [Copernicus Atmosphere Monitoring Service](https://atmosphere.copernicus.eu/data) API and allows to export NetCDF files to be used as input in the grid_processing notebook.
- ARPA_API_ground_sensors: Access to ARPA Ground Sensors API for both meteorological and air quality stations. It allows to export a geopackage file for each sensor type.
- GEE_API: Access to Google Earth Engine API to retrieve satellite data. It allows to export .tif files for each selected variable.
- grid_processing: it allows to calculate the mean value in each cell for all the selected variables.

|Variable|Description|Source|
|---|---|---|
|temp_2m|Mean air temperature at 2 m above the land surface|https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY|