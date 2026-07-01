# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
import joblib

def create_features(df):
    """Creates features for the model."""
    # Target Variable (Binary)
    df['Target'] = ((df['Listing Gains(%)'].fillna(0) > 0)).astype(int)

    # Log Transform Skewed Features
    if 'Issue Size  (in crores)' in df.columns:
        df['Log_Issue_Size'] = np.log1p(df['Issue Size  (in crores)'])

    # Investor Category (Categorical)
    df['Investor_Category'] = pd.cut(
        df['RII'],
        bins=[-np.inf, 50, 100, np.inf],
        labels=['Retail', 'HNI', 'Institutional']
    )
    
    # --- Add Placeholders for New, More Powerful Features ---
    df['gmp_percentage'] = 0.0  # Grey Market Premium as a percentage of issue price
    df['revenue_crores'] = 0.0  # Company's latest annual revenue
    df['profit_margin'] = 0.0    # Company's net profit margin

    df['Inverse_Subscription'] = 1 / (df['Total'] + 1e-6)

    return df

def main():
    """Main function to train and evaluate models."""
    data_path = "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/ipo.csv"
    df = pd.read_csv(data_path, header=1)
    df = create_features(df)

    features = ['Log_Issue_Size', 'Inverse_Subscription', 'Investor_Category', 'gmp_percentage', 'revenue_crores', 'profit_margin']
    target = 'Target'
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numerical_features = ['Log_Issue_Size', 'Inverse_Subscription', 'gmp_percentage', 'revenue_crores', 'profit_margin']
    categorical_features = ['Investor_Category']

    numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    # --- MODIFIED: Model Training, Selection, and Hyperparameter Tuning ---
    
    # Define models to be tuned
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=10000)
    }

    # Define hyperparameter grids for each model
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        },
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        }
    }

    best_model_pipeline = None
    best_score = 0
    best_model_name = ""

    # Loop through each model and perform GridSearchCV
    for name, model in models.items():
        print(f"--- Tuning {name} ---")
        
        # NOTE: GridSearchCV will be performed on the preprocessed and resampled data
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate the best estimator from the grid search on the test set
        score = roc_auc_score(y_test, grid_search.predict_proba(X_test_processed)[:, 1])
        print(f"Best {name} Tuned Test ROC-AUC: {score:.4f}")
        print(f"Best parameters: {grid_search.best_params_}\n")

        if score > best_score:
            best_score = score
            best_model_name = name
            # Re-create the final pipeline with the best tuned model
            best_model_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', grid_search.best_estimator_) # Use the best model found
            ])

        # Save each tuned model
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', grid_search.best_estimator_)
        ])
        calibrated_model = CalibratedClassifierCV(model_pipeline, method='isotonic', cv=5)
        calibrated_model.fit(X_train, y_train)
        joblib.dump(calibrated_model, f"/Users/adi/Desktop/Mission/IPO Allotment Prediction model/{name.lower().replace(' ', '_')}_model.pkl")
        print(f"Saved calibrated {name} model.")

    print(f"\nBest overall model is: {best_model_name} with ROC-AUC of {best_score:.4f}")

    # --- Calibrate and Save the Best Tuned Model ---
    # Calibrate the entire pipeline which now includes the best tuned classifier
    calibrated_pipeline = CalibratedClassifierCV(best_model_pipeline, method='isotonic', cv=5)
    
    # We fit the calibration on the original (non-resampled) training data
    # as this often gives better calibrated probabilities.
    calibrated_pipeline.fit(X_train, y_train)
        # Model evaluation: check for overfitting/underfitting
    y_train_pred = calibratedpipeline.predict_proba(X_train)[:, 1]
    y_test_pred = calibratedpipeline.predict_proba(X_test)[:, 1]

    train_roc_auc = roc_auc_score(y_train, y_train_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred)

    print(f"Train ROC-AUC: {train_roc_auc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(classification_report(y_test, calibratedpipeline.predict(X_test)))

    joblib.dump(calibrated_pipeline, "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/best_model.pkl")
    print("\nSuccessfully trained, tuned, and saved the calibrated best model.")

    # --- Final Evaluation ---
    y_pred = calibrated_pipeline.predict(X_test)
    print("\nFinal Calibrated Model Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()