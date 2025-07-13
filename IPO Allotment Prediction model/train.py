
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from xgboost import XGBClassifier
from fuzzywuzzy import process
import joblib

# Step 2: Load and Explore Data
# Load Datasets
ipo = pd.read_csv("/Users/adi/Desktop/Mission/IPO Allotment Prediction model/ipo.csv", header=1)
cleaned_ipo = pd.read_csv("/Users/adi/Desktop/Mission/IPO Allotment Prediction model/cleaned_ipo_data 2022-25.csv")

# Display First Few Rows
print(ipo.head())
print(cleaned_ipo.head(380))

# Check for Missing Values
print(ipo.isnull().sum())
print(cleaned_ipo.isnull().sum())

# Combine datasets
# Fuzzy Match Company Names
def fuzzy_merge(df1, df2, key1, key2, threshold=80):
    matches = []
    for name in df1[key1]:
        result = process.extractOne(name, df2[key2])
        if result:
            match, score = result[0], result[1]
            if score >= threshold:
                matches.append(match)
            else:
                matches.append(None)
        else:
            matches.append(None)
    df1['matched_name'] = matches
    return df1.merge(df2, left_on='matched_name', right_on=key2, how='inner')

combined = fuzzy_merge(ipo, cleaned_ipo, 'IPO Name', 'Name')

# Display Combined Dataset
print(combined.head())

# preprocess the data.
# Target Variable (Binary)
combined['Target'] = ((combined['Listing Gains(%)'].fillna(0) > 0) | (combined['Returns'].fillna(0) > 0)).astype(int)

# Log Transform Skewed Features
# The column 'Issue Size  (in crores)' has a space at the end.
combined['Log_Issue_Size'] = np.log1p(combined['Issue Size  (in crores)'])

# Investor Category (Categorical)
combined['Investor_Category'] = pd.cut(
    combined['RII'], 
    bins=[-np.inf, 50, 100, np.inf], 
    labels=['Retail', 'HNI', 'Institutional']
)

# Drop Unnecessary Columns
combined = combined[['Log_Issue_Size', 'Total', 'Investor_Category', 'Target']]

# Step 5: Train-Test Split
# Define Features and Target
X = combined.drop(columns=['Target'])
y = combined['Target']

# Calculate scale_pos_weight for handling class imbalance
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Build Pipeline
# Preprocessing Pipelines
numerical_features = ['Log_Issue_Size', 'Total']
categorical_features = ['Investor_Category']

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Full Pipeline with Model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, reg_alpha=0.1))
])

# Step 7: Hyperparameter Tuning
# Hyperparameter Grid
param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [3, 4, 5],
    'classifier__learning_rate': [0.05, 0.1, 0.15],
    'classifier__gamma': [0, 0.1, 0.2]
}

# Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Best Parameters
print("Best Parameters:", grid_search.best_params_)

# Step 8: Evaluate the Model
# Predict Probabilities
y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.legend()
plt.show()

# Classification Report
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 9: Deploy the Model
# Save Model
joblib.dump(grid_search, "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/ipo_model.pkl")
