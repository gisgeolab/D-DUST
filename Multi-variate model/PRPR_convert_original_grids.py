import os
import pandas as pd
import geopandas as gpd

folder_path = 'D:\\Lorenzo Documents\\Lorenzo\\Research Documents\\2022 06 -- D-DUST\\23 05 01 - Multivariate Model\\GridsDownload'

for root,dirs,files in os.walk(folder_path):
    for file in files:
        if file.endswith('.gpkg'):
            gpkg_path = os.path.join(root,file)
            csv_path = os.path.splitext(gpkg_path)[0] + '.csv'
            gdf = pd.DataFrame(gpd.read_file(gpkg_path))
            colnames = list()
            colnames.append('fid')
            for c in list(gdf.columns.values):
                colnames.append(c)
            gdf['fid'] = gdf.index.values
            gdf['fid'] = gdf['fid'] + 1
            outgdf = gdf[colnames]
            outgdf.to_csv(csv_path,index=False)

            print(f"Converted '{gpkg_path}' to '{csv_path}' ")

br = 1