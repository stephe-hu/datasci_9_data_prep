import pandas as pd 

## get data 
# Landing Page: https://data.cityofnewyork.us/Public-Safety/NYPD-Arrest-Data-Year-to-Date-/uip8-fykc
# Data Download Link: 
datalink = 'https://data.cityofnewyork.us/api/views/uip8-fykc/rows.csv?accessType=DOWNLOAD'
df = pd.read_csv(datalink)
df.size
df.sample(5)


## save as csv to model_dev2/data/raw
df.to_csv('model_dev2/data/raw/nypd_arrests_data.csv', index=False)

## save as pickle to model_dev2/data/raw
df.to_pickle('model_dev2/data/raw/nypd_arrests_data.pkl')