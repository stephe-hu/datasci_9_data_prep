import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## get data raw
df = pd.read_pickle('model_dev2/data/raw/nypd_arrests_data.pkl')

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
    'arrest_key',
    'pd_cd',
    'ky_cd',
    'law_code',
    'arrest_precinct',
    'jurisdiction_code',
    'x_coord_cd',
    'y_coord_cd',
    'latitude',
    'longitude',
    'new_georeferenced_column',
]
df.drop(to_drop, axis=1, inplace=True, errors='ignore')
df.shape 
df.sample(5)


## arrest_date --> will need to encode this
df.arrest_date.value_counts()
## now encode arrest_date so it is a day of the week
df['arrest_date'] = pd.to_datetime(df['arrest_date'])
df['arrest_date'] = df['arrest_date'].dt.day_name()
## perform ordinal encoding on arrest_date
enc = OrdinalEncoder()
enc.fit(df[['arrest_date']])
df['arrest_date'] = enc.transform(df[['arrest_date']])
## create dataframe with mapping
df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['arrest_date'])
df_mapping_date['arrest_date_ordinal'] = df_mapping_date.index
df_mapping_date
## save mapping to csv
df_mapping_date.to_csv('model_dev2/data/processed/mapping_arrest_date.csv', index=False)



## pd_desc --> will need to encode this
df.pd_desc.value_counts()
## drop row if pd_desc is equal to (null)
df = df[df['pd_desc'] != '(null)']
## perform orindla encoding on pd_desc
enc = OrdinalEncoder()
enc.fit(df[['pd_desc']])
df['pd_desc'] = enc.transform(df[['pd_desc']])
## create dataframe with mapping
df_mapping_pd_desc = pd.DataFrame(enc.categories_[0], columns=['pd_desc'])
df_mapping_pd_desc['pd_desc_ordinal'] = df_mapping_pd_desc.index
df_mapping_pd_desc.head(5)
# save mapping to csv
df_mapping_pd_desc.to_csv('model_dev2/data/processed/mapping_pd_desc.csv', index=False)



## ofns_desc --> will need to encode this
df.ofns_desc.value_counts()
## get count of missing for ofns_desc
df.ofns_desc.isna().sum()
## perform ordinal encoding on ofns_desc
enc = OrdinalEncoder()
enc.fit(df[['ofns_desc']])
df['ofns_desc'] = enc.transform(df[['ofns_desc']])
df.ofns_desc.value_counts()
## create dataframe with mapping
df_mapping_ofns_desc = pd.DataFrame(enc.categories_[0], columns=['ofns_desc'])
df_mapping_ofns_desc['ofns_desc_ordinal'] = df_mapping_ofns_desc.index
df_mapping_ofns_desc.head(5)
# save mapping to csv
df_mapping_ofns_desc.to_csv('model_dev2/data/processed/mapping_ofns_desc.csv', index=False)




## law_cat_cd --> will need to encode this
df.law_cat_cd.value_counts()
## get count of missing for law_cat_cd
df.law_cat_cd.isna().sum()
## replace isna with 'Not Reported'
df.law_cat_cd.fillna('Not Reported', inplace=True)
## drop row if cat_cd is equal to 9 or I
df = df[df['law_cat_cd'] != '9']
df = df[df['law_cat_cd'] != 'I']
df.law_cat_cd.value_counts()
## perform ordinal encoding on law_cat_cd
enc = OrdinalEncoder()
enc.fit(df[['law_cat_cd']])
df['law_cat_cd'] = enc.transform(df[['law_cat_cd']])
## create dataframe with mapping
df_mapping_law_cat_cd = pd.DataFrame(enc.categories_[0], columns=['law_cat_cd'])
df_mapping_law_cat_cd['law_cat_cd_ordinal'] = df_mapping_law_cat_cd.index
df_mapping_law_cat_cd.head(5)
# save mapping to csv
df_mapping_law_cat_cd.to_csv('model_dev2/data/processed/mapping_law_cat_cd.csv', index=False)



## arrest_boro --> will need to encode this
df.arrest_boro.value_counts()
## get count of missing for arrest_boro
df.arrest_boro.isna().sum()
## perform ordinal encoding on arrest_boro
enc = OrdinalEncoder()
enc.fit(df[['arrest_boro']])
df['arrest_boro'] = enc.transform(df[['arrest_boro']])
## create dataframe with mapping
df_mapping_arrest_boro = pd.DataFrame(enc.categories_[0], columns=['arrest_boro'])
df_mapping_arrest_boro['arrest_boro_ordinal'] = df_mapping_arrest_boro.index
df_mapping_arrest_boro.head(5)
# save mapping to csv
df_mapping_arrest_boro.to_csv('model_dev2/data/processed/mapping_arrest_boro.csv', index=False)



## age_group --> will need to encode this
df.age_group.value_counts()
## get count of missing for perp
df.age_group.isna().sum()
## perform ordinal encoding on age_group
enc = OrdinalEncoder()
enc.fit(df[['age_group']])
df['age_group'] = enc.transform(df[['age_group']])
## create dataframe with mapping
df_mapping_age_group = pd.DataFrame(enc.categories_[0], columns=['age_group'])
df_mapping_age_group['age_group_ordinal'] = df_mapping_age_group.index
df_mapping_age_group.head(5)
# save mapping to csv
df_mapping_age_group.to_csv('model_dev2/data/processed/mapping_age_group.csv', index=False)



## perp_sex --> will need to encode this
df.perp_sex.value_counts()
## drop row if perp_sex is equal to U
df = df[df['perp_sex'] != 'U']
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
df_mapping_perp_sex.to_csv('model_dev2/data/processed/mapping_perp_sex.csv', index=False)



## perp_race --> will need to encode this
df.perp_race.value_counts()
## drop row if race is equal to UNKOWN
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
df_mapping_perp_race.to_csv('model_dev2/data/processed/mapping_perp_race.csv', index=False)


len(df)

#### save a temporary csv file of 10000 rows to test the model
df.head(10000).to_csv('model_dev2/data/processed/nypd_arrest_data.csv', index=False)
df