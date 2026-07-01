
import pandas as pd
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


def load_processed_data(path):
    """Loads the pre-processed and featured data."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {path}. Please run data_processing.py first.")
        return None

def main():
    """Main function to train and evaluate models."""
    # Load the processed data
    data_path = "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/featured_ipo_data.csv"
    df = load_processed_data(data_path)
    if df is None:
        return

    # Define features and target
    # We now include the new placeholder features
    features = ['Log_Issue_Size', 'Total', 'Investor_Category', 'gmp_percentage', 'revenue_crores', 'profit_margin']
    target = 'Target'
    
    X = df[features]
    y = df[target]

    # Initial Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define preprocessing steps
    numerical_features = ['Log_Issue_Size', 'Total', 'gmp_percentage', 'revenue_crores', 'profit_margin']
    categorical_features = ['Investor_Category']

    numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Apply preprocessing and SMOTE
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    # --- Model Training and Selection ---
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42)
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        score = roc_auc_score(y_test, model.predict_proba(X_test_processed)[:, 1])
        print(f"{name} Test ROC-AUC: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model

    print(f"\nBest model is: {type(best_model).__name__}")

    # --- Calibrate and Save the Best Model ---
    # We create a final pipeline that includes preprocessing and the best model
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])

    # Calibrate the entire pipeline
    calibrated_pipeline = CalibratedClassifierCV(final_pipeline, method='isotonic', cv=5)
    calibrated_pipeline.fit(X_train, y_train) # Calibrate on the original, non-resampled data

    # Save the calibrated model
    joblib.dump(calibrated_pipeline, "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/best_model.pkl")
    print("\nSuccessfully trained and saved the calibrated best model.")

    # --- Final Evaluation ---
    y_pred = calibrated_pipeline.predict(X_test)
    print("\nFinal Calibrated Model Classification Report on Test Set:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
