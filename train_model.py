import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

print('Loading dataset...')
df = pd.read_csv('calories (3).csv')
print(f'Dataset shape: {df.shape}')

print('\nRemoving outliers using Z-Score...')
z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
df_clean = df[(z_scores < 3).all(axis=1)]
print(f'Before: {df.shape}, After: {df_clean.shape}')

df_clean['Age'] = df_clean['Age'].clip(lower=0, upper=120)
df_clean['Body_Temp'] = df_clean['Body_Temp'].clip(lower=35, upper=40)
df_clean['Heart_Rate'] = df_clean['Heart_Rate'].clip(lower=30, upper=220)
df_clean['Weight'] = df_clean['Weight'].clip(lower=0, upper=200)
df_clean['Height'] = df_clean['Height'].clip(lower=0, upper=250)
df_clean['Duration'] = df_clean['Duration'].clip(lower=0, upper=120)

numeric_cols = df_clean.select_dtypes(include=np.number)
z_scores2 = np.abs(stats.zscore(numeric_cols))
outliers2 = (z_scores2 > 3)
df_clean = df_clean[(~outliers2).all(axis=1)]
print(f'After second outlier pass: {df_clean.shape}')

if 'User_ID' in df_clean.columns:
    df_clean.drop('User_ID', inplace=True, axis=1)

print('\nEncoding Gender...')
le = LabelEncoder()
df_clean['Gender'] = le.fit_transform(df_clean['Gender'])

X = df_clean.drop('Calories', axis=1)
y = df_clean['Calories']

print(f'\nFeatures: {list(X.columns)}')
print(f'X shape: {X.shape}, y shape: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('\nFinding optimal K...')
k_values = range(1, 21)
cv_scores = []
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

best_k = k_values[int(np.argmin(cv_scores))]
print(f'Best K: {best_k}')

print(f'\nTraining KNN Regressor with K={best_k}...')
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE : {mse:.4f}')
print(f'RMSE: {np.sqrt(mse):.4f}')
print(f'R2  : {r2:.4f}')

with open('model.pkl', 'wb') as f:
    pickle.dump(knn, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print('\nmodel.pkl and scaler.pkl saved successfully!')
print('Feature order for inference:', list(X.columns))
