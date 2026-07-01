import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
import io
import contextlib

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

def train_model():
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

    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        score = roc_auc_score(y_test, model.predict_proba(X_test_processed)[:, 1])
        
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        calibrated_pipeline = CalibratedClassifierCV(final_pipeline, method='isotonic', cv=5)
        calibrated_pipeline.fit(X_train, y_train)

        joblib.dump(calibrated_pipeline, f"/Users/adi/Desktop/Mission/IPO Allotment Prediction model/{name.lower().replace(' ', '_')}_model.pkl")

st.title("IPO Allotment Prediction")

st.sidebar.header("Controls")
if st.sidebar.button("Run Training"):
    st.sidebar.text("Training in progress...")
    train_model()
    st.sidebar.text("Training complete.")

st.header("Make a Prediction")

# --- Model Selection ---
model_choice = st.selectbox(
    "Choose the prediction model:",
    ("Random Forest", "XGBoost", "Logistic Regression")
)

# Define model paths
model_paths = {
    "Random Forest": "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/random_forest_model.pkl",
    "XGBoost": "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/xgboost_model.pkl",
    "Logistic Regression": "/Users/adi/Desktop/Mission/IPO Allotment Prediction model/logistic_regression_model.pkl"
}

# Load the selected model
selected_model_path = model_paths[model_choice]

try:
    model = joblib.load(selected_model_path)
except FileNotFoundError:
    st.error(f"Error: Model file not found at {selected_model_path}. Please run the training first.")
    st.stop()

# --- User Input Fields ---
oversubscription = st.number_input("Total Oversubscription Rate (e.g., 24.23 for 24.23x)", min_value=0.1, value=10.0, step=1.0)
category = st.selectbox("Your Investor Category", ["Retail", "HNI", "Institutional"])
issue_size = st.number_input("Total Issue Size (in Crores INR)", min_value=1, value=500, step=50)


# --- Prediction Logic ---
if st.button("Predict Allotment Chance"):
    # Preprocess Inputs
    input_data = pd.DataFrame({
        'Log_Issue_Size': [np.log1p(issue_size)],
        'Inverse_Subscription': [1 / (oversubscription + 1e-6)],
        'Investor_Category': [category],
        'gmp_percentage': [0],  # Placeholder
        'revenue_crores': [0],    # Placeholder
        'profit_margin': [0]      # Placeholder
    })

    try:
        # Predict probability
        probability = model.predict_proba(input_data)[0][1]
        
        st.subheader("Prediction Result")
        st.write(f"The estimated probability of a positive listing gain is:")
        
        # Display probability with a progress bar
        st.progress(float(probability))
        st.markdown(f"<h2 style='text-align: center; color: #28a745;'>{probability:.2%}</h2>", unsafe_allow_html=True)
        
        st.info("**Disclaimer:** This is a probabilistic estimate based on historical data and does not guarantee allotment or listing gains. The model's accuracy is limited by the available data.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
