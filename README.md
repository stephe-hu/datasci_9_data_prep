# datasci_9_data_prep

## Data Cleaning and Transformation Plan
### Dataset # 1: [NYPD Shooting Incident Data (Historic)](https://data.cityofnewyork.us/Public-Safety/NYPD-Shooting-Incident-Data-Historic-/833y-fsy8)
+ Located in [model_dev](/model_dev) folder.
+ This dataset contains a list of every shooting incident that occurred in NYC going back to 2006 through the end of the previous calendar year. Each record represents a shooting incident in NYC and includes information about the event, the location and time of occurrence. In addition, information related to suspect and victim demographics is also included.
+ The intended machine learning task is classification.
+ The independent variables are the occur_date, boro, location_desc, perp_age_group, perp_sex, perp_race, vic_age_group, and vic_race. The dependent variable is the vic_sex.
+ The steps needed to clean and transform the data are documented in the scripts.

### Dataset # 2: [NYPD Arrest Data (Year to Date)](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrest-Data-Year-to-Date-/uip8-fykc)
+ Located in [model_dev2](/model_dev2) folder.
+ This dataset contains every arrest effected in NYC by the NYPD during the current year. Each record represents an arrest effected in NYC by the NYPD and includes information about the type of crime, the location and time of enforcement.
+ The intended machine learning task is classification.
+ The independent variables are the arrest_date, pd_desc, ofns_desc, law_cat_cd, arrest_boro, age_group, and perp_sex The dependent variable is the perp_race.
+ The steps needed to clean and transform the data are documented in the scripts.