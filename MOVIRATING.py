import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with appropriate encoding
file_path = 'IMDb Movies India.csv'
movies_df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Drop rows with missing ratings
movies_df = movies_df.dropna(subset=['Rating'])

# Fill missing values in categorical columns
categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
movies_df[categorical_cols] = movies_df[categorical_cols].fillna('Unknown')

# Extract and fill year, duration, and votes columns
movies_df['Year'] = movies_df['Year'].str.extract(r'(\d{4})')
movies_df['Year'] = movies_df['Year'].fillna(0).astype(int)
movies_df['Duration'] = movies_df['Duration'].str.extract(r'(\d+)').fillna(0).astype(int)
movies_df['Votes'] = movies_df['Votes'].str.replace(',', '').fillna(0).astype(int)

# Convert categorical variables to dummy variables
movies_df = pd.get_dummies(movies_df, columns=categorical_cols)

# Separate features and target variable
X = movies_df.drop(['Rating', 'Name'], axis=1)
y = movies_df['Rating']

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Reduced hyperparameter tuning grid for Random Forest and Gradient Boosting
param_grid_rf = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

param_grid_gb = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [3, 4],
    'subsample': [1.0]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='neg_mean_squared_error')
grid_search_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=3, scoring='neg_mean_squared_error')

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    if name == 'Random Forest':
        grid_search_rf.fit(X_train, y_train)
        best_model_rf = grid_search_rf.best_estimator_
        y_pred = best_model_rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse
        print(f"Best Random Forest params: {grid_search_rf.best_params_}")
    elif name == 'Gradient Boosting':
        grid_search_gb.fit(X_train, y_train)
        best_model_gb = grid_search_gb.best_estimator_
        y_pred = best_model_gb.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse
        print(f"Best Gradient Boosting params: {grid_search_gb.best_params_}")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse

# Print results
for name, mse in results.items():
    print(f'{name} Mean Squared Error: {mse}')

# Feature importance analysis for the best model (Gradient Boosting)
if 'Gradient Boosting' in results:
    feature_importances = best_model_gb.feature_importances_
    features = movies_df.drop(['Rating', 'Name'], axis=1).columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Feature Importances (Gradient Boosting)')
    plt.show()
else:
    print("Gradient Boosting model was not trained successfully.")
