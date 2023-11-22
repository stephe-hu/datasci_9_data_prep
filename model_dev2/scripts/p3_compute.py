import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier, XGBRegressor


# Import the clean random sample of 10k data
df = pd.read_csv('model_dev2/data/processed/nypd_arrest_data.csv')
len(df)

# drop rows with missing values
df.dropna(inplace=True)
len(df)

# Define the features and the target variable
X = df.drop('perp_race', axis=1)  # Features (all columns except 'perp_race')
y = df['perp_race']               # Target variable (perp_race)

# Initialize the StandardScaler
scaler = StandardScaler()
scaler.fit(X) # Fit the scaler to the features
pickle.dump(scaler, open('model_dev/models/scaler.sav', 'wb')) # Save the scaler for later use

# Fit the scaler to the features and transform
X_scaled = scaler.transform(X)

# Split the scaled data into training, validation, and testing sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check the size of each set
(X_train.shape, X_val.shape, X_test.shape)

# Pkle the X_train for later use in explanation
pickle.dump(X_train, open('model_dev2/models/X_train.sav', 'wb'))
# Pkle X.columns for later use in explanation
pickle.dump(X.columns, open('model_dev2/models/X_columns.sav', 'wb'))