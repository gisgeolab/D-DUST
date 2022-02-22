# D-DUST
---
### Notebook description
- ADS_API_Copernicus: Access to [Copernicus Atmosphere Monitoring Service](https://atmosphere.copernicus.eu/data) API and allows to export NetCDF files to be used as input in the grid_processing notebook.
- ARPA_API_ground_sensors: Access to ARPA Ground Sensors API for both meteorological and air quality stations. It allows to export a geopackage file for each sensor type.
- GEE_API: Access to Google Earth Engine API to retrieve satellite data. It allows to export .tif files for each selected variable.
- grid_processing: it allows to calculate the mean value in each cell for all the selected variables.

|Climate Data|
|Variable|Description|Source|
|---|---|---|
|temp_2m|Mean air temperature at 2 m above the land surface|https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY|
|temp_st|Mean temperature - ARPA monitor station|https://www.dati.lombardia.it/Ambiente/Dati-sensori-meteo/647i-nhxk|
|e_wind|Mean eastward wind component 10 m above the land surface|https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY|
|n_wind|Mean northward wind component 10 m above the land surface|https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY|
|wind_dir_st|Mean wind direction ground - ARPA monitoring station|https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY|
|wind_speed_st|Mean wind speed ground  - ARPA monitoring station|https://www.dati.lombardia.it/Ambiente/Dati-sensori-meteo/647i-nhxk|
|prec|Mean accumulated liquid and frozen water, including rain and snow, that falls to the Earth's surface. It is the sum of large-scale precipitation|https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY|
|prec_st|Mean precipitation in each cell in the time range - ARPA monitor station|https://www.dati.lombardia.it/Ambiente/Dati-sensori-meteo/647i-nhxk|
|air_hum_st|Mean air moisture measurement in the time range - ARPA monitoring station|https://www.dati.lombardia.it/Ambiente/Dati-sensori-meteo/647i-nhxk|
|press|Mean weight of all the air in a column vertically above the area of the Earth's surface represented at a fixed point|https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY|
|rad_glob_st|Global radiation measurement - ARPA monitoring station|https://www.dati.lombardia.it/Ambiente/Dati-sensori-meteo/647i-nhxk|

