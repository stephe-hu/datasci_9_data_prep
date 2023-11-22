import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## get data raw
df = pd.read_pickle('model_dev/data/raw/nypdshooting_data.pkl')

## get column names
df.columns

## do some data cleaning of colun names, 
## make them all lower case, replmove white spaces and rpelace with _ 
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## get data types
df.dtypes # nice combination of numbers and strings/objects 
len(df)

## drop columns
to_drop = [
    'incident_key',
    'occur_time',
    'precinct',
    'loc_of_occur_desc',
    'jurisdiction_code',
    'loc_classfctn_desc',
    'statistical_murder_flag',
    'x_coord_cd',
    'y_coord_cd',
    'latitude',
    'longitude',
    'lon_lat',
]
df.drop(to_drop, axis=1, inplace=True, errors='ignore')
df.shape 
df.sample(5)

## clean occur_date column so it is just the date without the time
df['occur_date'] = pd.to_datetime(df['occur_date']).dt.date
## now encode occur_date so it is a day of the week
df['occur_date'] = pd.to_datetime(df['occur_date'])
df['occur_date'] = df['occur_date'].dt.day_name()
## perform ordinal encoding on occur_date
enc = OrdinalEncoder()
enc.fit(df[['occur_date']])
df['occur_date'] = enc.transform(df[['occur_date']])
## create dataframe with mapping
df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['occur_date'])
df_mapping_date['occur_date_ordinal'] = df_mapping_date.index
df_mapping_date
## save mapping to csv
df_mapping_date.to_csv('model_dev/data/processed/mapping_date.csv', index=False)



## boro --> will need to encode this
df.boro.value_counts()
## perform orindla encoding on boro
enc = OrdinalEncoder()
enc.fit(df[['boro']])
df['boro'] = enc.transform(df[['boro']])
## create dataframe with mapping
df_mapping_boro = pd.DataFrame(enc.categories_[0], columns=['boro'])
df_mapping_boro['boro_ordinal'] = df_mapping_boro.index
df_mapping_boro.head(5)
# save mapping to csv
df_mapping_boro.to_csv('model_dev/data/processed/mapping_boro.csv', index=False)





## location_desc --> will need to encode this
df.location_desc.value_counts()
## get count of missing for location_desc
df.location_desc.isna().sum()
## replace isna with 'Not Reported'
df.location_desc.fillna('Not Reported', inplace=True)
## drop row if location_desc is equal to (null) or NONE
df = df[df['location_desc'] != '(null)']
df = df[df['location_desc'] != 'NONE']
df.location_desc.value_counts()
## perform ordinal encoding on location_desc
enc = OrdinalEncoder()
enc.fit(df[['location_desc']])
df['location_desc'] = enc.transform(df[['location_desc']])
df.location_desc.value_counts()
## create dataframe with mapping
df_mapping_location_desc = pd.DataFrame(enc.categories_[0], columns=['location_desc'])
df_mapping_location_desc['location_desc_ordinal'] = df_mapping_location_desc.index
df_mapping_location_desc.head(5)
# save mapping to csv
df_mapping_location_desc.to_csv('model_dev/data/processed/mapping_location_desc.csv', index=False)




## perp_age_group --> will need to encode this
df.perp_age_group.value_counts()

## get count of missing for perp_age_group
df.perp_age_group.isna().sum()

## replace isna with 'Not Reported'
df.perp_age_group.fillna('Not Reported', inplace=True)

## drop row if age_group is equal to UNKNOWN, 940, 224, or 1020
df = df[df['perp_age_group'] != 'UNKNOWN']
df = df[df['perp_age_group'] != '940']
df = df[df['perp_age_group'] != '224']
df = df[df['perp_age_group'] != '1020']
df.perp_age_group.value_counts()

## perform ordinal encoding on perp_age_group
enc = OrdinalEncoder()
enc.fit(df[['perp_age_group']])
df['perp_age_group'] = enc.transform(df[['perp_age_group']])

## create dataframe with mapping
df_mapping_perp_age_group = pd.DataFrame(enc.categories_[0], columns=['perp_age_group'])
df_mapping_perp_age_group['perp_age_group_ordinal'] = df_mapping_perp_age_group.index
df_mapping_perp_age_group.head(5)
# save mapping to csv
df_mapping_perp_age_group.to_csv('model_dev/data/processed/mapping_perp_age_group.csv', index=False)



## perp_sex --> will need to encode this
df.perp_sex.value_counts()

## get count of missing for perp_sex
df.perp_sex.isna().sum()

## replace isna with 'Not Reported'
df.perp_sex.fillna('Not Reported', inplace=True)

## drop row if sex is equal to U or (null)
df = df[df['perp_sex'] != 'U']
df = df[df['perp_sex'] != '(null)']
df.perp_sex.value_counts()

## perform ordinal encoding on perp_sex
enc = OrdinalEncoder()
enc.fit(df[['perp_sex']])
df['perp_sex'] = enc.transform(df[['perp_sex']])

## create dataframe with mapping
df_mapping_perp_sex = pd.DataFrame(enc.categories_[0], columns=['perp_sex'])
df_mapping_perp_sex['perp_sex_ordinal'] = df_mapping_perp_sex.index
df_mapping_perp_sex.head(5)
# save mapping to csv
df_mapping_perp_sex.to_csv('model_dev/data/processed/mapping_perp_sex.csv', index=False)



## perp_race --> will need to encode this
df.perp_race.value_counts()

## get count of missing for perp
df.perp_race.isna().sum()

## replace isna with 'Not Reported'
df.perp_race.fillna('Not Reported', inplace=True)

## drop row if race is equal to UNKNOWN
df = df[df['perp_race'] != 'UNKNOWN']
df.perp_race.value_counts()

## perform ordinal encoding on perp_race
enc = OrdinalEncoder()
enc.fit(df[['perp_race']])
df['perp_race'] = enc.transform(df[['perp_race']])

## create dataframe with mapping
df_mapping_perp_race = pd.DataFrame(enc.categories_[0], columns=['perp_race'])
df_mapping_perp_race['perp_race_ordinal'] = df_mapping_perp_race.index
df_mapping_perp_race.head(5)
# save mapping to csv
df_mapping_perp_race.to_csv('model_dev/data/processed/mapping_perp_race.csv', index=False)



## vic_age_group --> will need to encode this
df.vic_age_group.value_counts()

## drop row if age_group is equal to UNKNOWN, or 1022
df = df[df['vic_age_group'] != 'UNKNOWN']
df = df[df['vic_age_group'] != '1022']
df.vic_age_group.value_counts()

## perform ordinal encoding on vic_age_group
enc = OrdinalEncoder()
enc.fit(df[['vic_age_group']])
df['vic_age_group'] = enc.transform(df[['vic_age_group']])

## create dataframe with mapping
df_mapping_vic_age_group = pd.DataFrame(enc.categories_[0], columns=['vic_age_group'])
df_mapping_vic_age_group['vic_age_group_ordinal'] = df_mapping_vic_age_group.index
df_mapping_vic_age_group.head(5)
# save mapping to csv
df_mapping_vic_age_group.to_csv('model_dev/data/processed/mapping_vic_age_group.csv', index=False)



## vic_sex --> will need to encode this
df.vic_sex.value_counts()

## drop row if sex is equal to U
df = df[df['vic_sex'] != 'U']
df.vic_sex.value_counts()

## perform ordinal encoding on vic_sex
enc = OrdinalEncoder()
enc.fit(df[['vic_sex']])
df['vic_sex'] = enc.transform(df[['vic_sex']])

## create dataframe with mapping
df_mapping_vic_sex = pd.DataFrame(enc.categories_[0], columns=['vic_sex'])
df_mapping_vic_sex['vic_sex_ordinal'] = df_mapping_vic_sex.index
df_mapping_vic_sex.head(5)
# save mapping to csv
df_mapping_vic_sex.to_csv('model_dev/data/processed/mapping_vic_sex.csv', index=False)



## vic_race --> will need to encode this
df.vic_race.value_counts()

## drop row if race is equal to UNKOWN
df = df[df['vic_race'] != 'UNKNOWN']
df.vic_race.value_counts()

## perform ordinal encoding on vic_race
enc = OrdinalEncoder()
enc.fit(df[['vic_race']])
df['vic_race'] = enc.transform(df[['vic_race']])

## create dataframe with mapping
df_mapping_vic_race = pd.DataFrame(enc.categories_[0], columns=['vic_race'])
df_mapping_vic_race['vic_race_ordinal'] = df_mapping_vic_race.index
df_mapping_vic_race.head(5)
# save mapping to csv
df_mapping_vic_race.to_csv('model_dev/data/processed/mapping_vic_race.csv', index=False)


len(df)

#### save a temporary csv file of 10000 rows to test the model
df.head(10000).to_csv('model_dev/data/processed/nypdshooting_data.csv', index=False)
df