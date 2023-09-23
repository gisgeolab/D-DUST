import os
import pandas as pd

folder_path = 'Datasources\\Grids\\'
for root,dirs,files in os.walk(folder_path):
    for file in files:
        if file.endswith('.csv') and file.startswith('grid'):
            filename = str(file).replace('.csv','')
            nameparts = filename.split('_')
            yearstr = nameparts[5][2:4]
            startstr = nameparts[3]
            endstr = nameparts[4]
            outname = yearstr + startstr + '_' + yearstr + endstr + '.csv'
            outpath = root + '\\' + outname
            infile = pd.read_csv(root+'\\'+file,encoding='ISO-8859-1',low_memory=False)
            #infile.to_csv(outpath,index=False)
            infile.to_csv('Datasources\\timeframes_protocols\\All\\'+outname,index=False)

            print(f"Saved '{filename}' as '{outname}' ")

br = 1