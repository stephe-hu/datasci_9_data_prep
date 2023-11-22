import pandas as pd 

## get data 
# Landing Page: https://data.cityofnewyork.us/Public-Safety/NYPD-Shooting-Incident-Data-Historic-/833y-fsy8
# Data Download Link: 
datalink = 'https://data.cityofnewyork.us/api/views/833y-fsy8/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df.size
df.sample(5)


## save as csv to model_dev/data/raw
df.to_csv('model_dev/data/raw/nypdshooting_data.csv', index=False)

## save as pickle to model_dev/data/raw
df.to_pickle('model_dev/data/raw/nypdshooting_data.pkl')