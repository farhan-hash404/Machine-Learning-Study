# Bike Sharing Demand Prediction using XGBoost Regressor

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt

# Step 2: Load Dataset
# Replace 'hour.csv' with the path to your downloaded dataset
data = pd.read_csv('hour.csv')

# Step 3: Preprocess Data
# Convert date column
data['dteday'] = pd.to_datetime(data['dteday'])

# Feature engineering
data['day'] = data['dteday'].dt.day
data['month'] = data['dteday'].dt.month
data['weekday'] = data['dteday'].dt.weekday
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x>=5 else 0)

# Drop unnecessary columns and define target
X = data.drop(columns=['instant','dteday','casual','registered','cnt'])
y = data['cnt']

# Encode categorical features
categorical_features = ['season', 'weathersit', 'holiday', 'workingday', 'weekday', 'month', 'hr']
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train XGBoost Regressor
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    early_stopping_rounds=50,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print('RMSE:', rmse)
print('R2 Score:', r2)

# Step 7: Feature Importance
plot_importance(model, max_num_features=10)
plt.show()

# Step 8: Hyperparameter Tuning (Optional)
param_grid = {
    'n_estimators':[500, 1000],
    'max_depth':[4,6,8],
    'learning_rate':[0.01,0.05,0.1],
    'subsample':[0.7,0.8,1.0]
}

grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid.fit(X_train, y_train)
print('Best Parameters:', grid.best_params_)

# Bike Sharing Demand Prediction using XGBoost Regressor

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt

# Step 2: Load Dataset
# Replace 'hour.csv' with the path to your downloaded dataset
data = pd.read_csv('hour.csv')

# Step 3: Preprocess Data
# Convert date column
data['dteday'] = pd.to_datetime(data['dteday'])

# Feature engineering
data['day'] = data['dteday'].dt.day
data['month'] = data['dteday'].dt.month
data['weekday'] = data['dteday'].dt.weekday
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x>=5 else 0)

# Drop unnecessary columns and define target
X = data.drop(columns=['instant','dteday','casual','registered','cnt'])
y = data['cnt']

# Encode categorical features
categorical_features = ['season', 'weathersit', 'holiday', 'workingday', 'weekday', 'month', 'hr']
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train XGBoost Regressor
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    early_stopping_rounds=50,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print('RMSE:', rmse)
print('R2 Score:', r2)

# Step 7: Feature Importance
plot_importance(model, max_num_features=10)
plt.show()

# Step 8: Hyperparameter Tuning (Optional)
param_grid = {
    'n_estimators':[500, 1000],
    'max_depth':[4,6,8],
    'learning_rate':[0.01,0.05,0.1],
    'subsample':[0.7,0.8,1.0]
}

grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid.fit(X_train, y_train)
print('Best Parameters:', grid.best_params_)