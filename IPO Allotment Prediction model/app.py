import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Page Config ---
st.set_page_config(
    page_title="IPO Listing Gain & Allotment Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');
    
    /* Base typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title styling */
    .title-gradient {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #7c3aed, #db2777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-bottom: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease-in-out;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Prediction Banner */
    .pred-box {
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        border: 1px solid transparent;
    }
    .pred-high {
        background-color: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.3);
        color: #10b981;
    }
    .pred-mid {
        background-color: rgba(245, 158, 11, 0.1);
        border-color: rgba(245, 158, 11, 0.3);
        color: #f59e0b;
    }
    .pred-low {
        background-color: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Historical Data ---
@st.cache_data
def load_ipo_data():
    if os.path.exists("ipo.csv"):
        try:
            df = pd.read_csv("ipo.csv", header=1)
            # Basic cleanup
            df = df.dropna(subset=['IPO Name'])
            df['Issue Size  (in crores)'] = pd.to_numeric(df['Issue Size  (in crores)'], errors='coerce')
            df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
            df['Listing Gains(%)'] = pd.to_numeric(df['Listing Gains(%)'], errors='coerce')
            return df
        except Exception as e:
            st.error(f"Error reading dataset: {e}")
            return None
    return None

df_ipo = load_ipo_data()

# --- Model Selection & Metadata ---
model_info = {
    "Logistic Regression (Best, Calibrated)": {
        "path": "best_model.pkl",
        "auc": 0.6729,
        "accuracy": 0.6250,
        "type": "Calibrated LogReg (Isotonic)"
    },
    "XGBoost Classifier": {
        "path": "xgboost_model.pkl",
        "auc": 0.6207,
        "accuracy": 0.5781,
        "type": "Calibrated XGBoost (Isotonic)"
    },
    "Random Forest Classifier": {
        "path": "random_forest_model.pkl",
        "auc": 0.5852,
        "accuracy": 0.6094,
        "type": "Calibrated Random Forest (Isotonic)"
    }
}

# --- Sidebar Controls ---
st.sidebar.markdown("## ⚙️ Model Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose Active Prediction Model:",
    list(model_info.keys())
)

# Load selected model
selected_model_path = model_info[selected_model_name]["path"]
selected_model_auc = model_info[selected_model_name]["auc"]
selected_model_acc = model_info[selected_model_name]["accuracy"]
selected_model_type = model_info[selected_model_name]["type"]

try:
    model = joblib.load(selected_model_path)
    model_loaded = True
except Exception as e:
    model_loaded = False

# Sidebar stats panel
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Active Model Specs")
st.sidebar.metric(label="Model Type", value=selected_model_type)
st.sidebar.metric(label="Test ROC-AUC Score", value=f"{selected_model_auc:.4f}")
st.sidebar.metric(label="Test Accuracy", value=f"{selected_model_acc:.2%}")
st.sidebar.info("The models utilize SMOTE to balance classes and are calibrated via Isotonic Regression on the original validation split to output precise probability scores.")

# --- App Title Banner ---
st.markdown("<div class='title-gradient'>IPO Listing Gain & Allotment Assistant</div>", unsafe_allow_html=True)
st.markdown("##### Empowering retail and HNI investors with data-driven listing-gain predictive probabilities.")

# --- Tab Layout ---
tab_predict, tab_analytics, tab_dataset, tab_portfolio = st.tabs([
    "🎯 Predict Listing Gains",
    "📊 Performance Diagnostics",
    "🔍 Historical IPO Explorer",
    "💼 Portfolio & CV Highlights"
])

# ================= TAB 1: PREDICT LISTING GAINS =================
with tab_predict:
    st.markdown("### 🎲 Allotment Scenario Planner")
    st.write("Enter details for an upcoming or historical IPO to predict the probability of securing a **positive listing gain** (>0%).")
    
    if not model_loaded:
        st.error(f"⚠️ Model file `{selected_model_path}` could not be loaded. Please ensure retraining completed successfully.")
    else:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### 📁 Subscription & Bidding Variables")
            oversubscription = st.slider(
                "Total Subscription Rate (x)",
                min_value=0.1,
                max_value=300.0,
                value=24.23,
                step=0.1,
                help="Total oversubscription rate across all categories (e.g., 24.23x)."
            )
            
            category = st.selectbox(
                "Your Investor Category",
                options=["Retail", "HNI", "Institutional"],
                index=0,
                help="Bidding category. Determines which subset of subscription limits are applied."
            )
            
            issue_size = st.number_input(
                "Total Issue Size (in Crores INR)",
                min_value=1.0,
                max_value=25000.0,
                value=500.0,
                step=50.0,
                help="Total capital raised by the company in Crores (10 Million INR)."
            )
            
            st.markdown("#### 📈 Market & Financial Health Indicators")
            gmp_percentage = st.slider(
                "Grey Market Premium (GMP %)",
                min_value=-50.0,
                max_value=200.0,
                value=25.0,
                step=1.0,
                help="Expected premium in the unofficial grey market relative to issue price."
            )
            
            col_rev, col_prof = st.columns(2)
            with col_rev:
                revenue_crores = st.number_input(
                    "Annual Revenue (Crores)",
                    min_value=0.0,
                    value=350.0,
                    step=25.0,
                    help="Most recent annual revenue of the issuing firm."
                )
            with col_prof:
                profit_margin = st.slider(
                    "Net Profit Margin (%)",
                    min_value=-30.0,
                    max_value=50.0,
                    value=12.0,
                    step=0.5,
                    help="Net profits as a percentage of total revenues."
                )
                
        with col2:
            st.markdown("#### 🔮 Listing Gain Prediction")
            st.write("Click below to run inference on the trained machine learning pipeline:")
            
            if st.button("Calculate Predictive Chance", use_container_width=True):
                # Dataframe mapping for preprocessor
                input_data = pd.DataFrame({
                    'Log_Issue_Size': [np.log1p(issue_size)],
                    'Inverse_Subscription': [1 / (oversubscription + 1e-6)],
                    'Investor_Category': [category],
                    'gmp_percentage': [gmp_percentage / 100.0],
                    'revenue_crores': [revenue_crores],
                    'profit_margin': [profit_margin / 100.0]
                })
                
                try:
                    # Run calibrated prediction
                    prob = model.predict_proba(input_data)[0][1]
                    
                    # Style class based on probability
                    if prob >= 0.65:
                        box_class = "pred-high"
                        status_msg = "🔥 High Listing Gain Likelihood"
                        details = "The subscription indicators, issue size, and financials strongly correlate with positive debut gains historically. This IPO presents low-risk listing potential based on trained parameters."
                    elif prob >= 0.35:
                        box_class = "pred-mid"
                        status_msg = "⚖️ Moderate Listing Gain Likelihood"
                        details = "The model suggests moderate chances of a positive return. Subscription rates or issue size levels show mixed signals. Keep an eye on market volatility closer to listing day."
                    else:
                        box_class = "pred-low"
                        status_msg = "⚠️ Low Listing Gain Likelihood"
                        details = "Substantially weak subscription metrics relative to issue size, or poor financial indicators suggest high risk. Historically, similar profiles struggled to trade above their offer price on listing day."
                    
                    st.markdown(f"""
                    <div class="pred-box {box_class}">
                        <h3>{status_msg}</h3>
                        <div class="metric-value">{prob:.2%}</div>
                        <p style="margin-top: 1rem; font-size: 1.1rem; line-height: 1.6;">{details}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(float(prob))
                    
                    # Core feature breakdown
                    st.markdown("##### 🔍 Model Inputs Summary:")
                    st.json({
                        "Bidding Category": category,
                        "Log(Issue Size)": round(float(np.log1p(issue_size)), 4),
                        "Inverse Subscription Intensity": round(float(1 / (oversubscription + 1e-6)), 6),
                        "GMP Factor": f"{gmp_percentage}%",
                        "Revenue": f"{revenue_crores} Cr",
                        "Profit Margin": f"{profit_margin}%"
                    })
                    
                except Exception as ex:
                    st.error(f"Prediction failed: {ex}")
            else:
                st.info("💡 Adjust values on the left and click **Calculate Predictive Chance** to view listing probabilities.")

# ================= TAB 2: MODEL DIAGNOSTICS =================
with tab_analytics:
    st.markdown("### 📊 Model Performance & Diagnostics")
    st.write("We evaluate models using the area under the Receiver Operating Characteristic curve (ROC-AUC), which measures class separability independent of the probability classification threshold.")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Logistic Regression</div>
            <div class="metric-value" style="color: #4f46e5;">0.6729</div>
            <div class="metric-label" style="font-size: 0.8rem;">ROC-AUC (Winner)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">XGBoost Classifier</div>
            <div class="metric-value" style="color: #7c3aed;">0.6207</div>
            <div class="metric-label" style="font-size: 0.8rem;">ROC-AUC Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Random Forest</div>
            <div class="metric-value" style="color: #db2777;">0.5852</div>
            <div class="metric-label" style="font-size: 0.8rem;">ROC-AUC Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("#### 📈 Diagnostic Visualizations")
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        if os.path.exists("target_distribution_improved.png"):
            st.image("target_distribution_improved.png", caption="Target Variable Class Balance (Listing Gains vs No Listing Gains)", use_container_width=True)
        else:
            st.warning("Target distribution plot not found.")
            
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion Matrix of the Best Calibrated Model", use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found.")
            
    with col_plot2:
        if os.path.exists("correlation_heatmap.png"):
            st.image("correlation_heatmap.png", caption="Feature Correlation Heatmap", use_container_width=True)
        else:
            st.warning("Correlation heatmap not found.")
            
        if os.path.exists("precision_recall_curve.png"):
            st.image("precision_recall_curve.png", caption="Precision-Recall Curve (AUC)", use_container_width=True)
        else:
            st.warning("Precision-Recall curve not found.")
            
    if os.path.exists("prf_vs_threshold.png"):
        st.markdown("---")
        st.markdown("#### 🎚️ Decision Threshold Optimization")
        st.image("prf_vs_threshold.png", caption="Precision, Recall, and F1-Score vs. Classification Threshold", use_container_width=True)

# ================= TAB 3: HISTORICAL EXPLORER =================
with tab_dataset:
    st.markdown("### 🔍 Historical IPO Explorer")
    st.write("Browse and filter historical IPO records (2022–2025) used to train the allotment classifiers.")
    
    if df_ipo is None:
        st.warning("Dataset `ipo.csv` could not be loaded or is formatted incorrectly.")
    else:
        # Search & Filter widgets
        col_search, col_filter = st.columns([2, 1])
        with col_search:
            search_query = st.text_input("🔍 Search by IPO Name:", "", placeholder="e.g. Dreamfolks, Harsha, Syrma...")
        with col_filter:
            outcome_filter = st.selectbox("Filter Listing Return:", ["All IPOs", "Positive Return Only (>0%)", "Negative Return Only (<0%)"])
            
        # Apply filters
        df_filtered = df_ipo.copy()
        if search_query:
            df_filtered = df_filtered[df_filtered['IPO Name'].str.contains(search_query, case=False, na=False)]
            
        if outcome_filter == "Positive Return Only (>0%)":
            df_filtered = df_filtered[df_filtered['Listing Gains(%)'] > 0]
        elif outcome_filter == "Negative Return Only (<0%)":
            df_filtered = df_filtered[df_filtered['Listing Gains(%)'] < 0]
            
        # KPI Stats block
        st.markdown("##### 📈 Quick Cohort Stats")
        tot_count = len(df_filtered)
        if tot_count > 0:
            avg_size = df_filtered['Issue Size  (in crores)'].mean()
            avg_gain = df_filtered['Listing Gains(%)'].mean()
            success_rate = (df_filtered['Listing Gains(%)'] > 0).mean()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("IPOs Displayed", f"{tot_count}")
            c2.metric("Avg Issue Size", f"{avg_size:.1f} Cr" if not pd.isna(avg_size) else "N/A")
            c3.metric("Avg Listing Gain", f"{avg_gain:.2f}%" if not pd.isna(avg_gain) else "N/A")
            c4.metric("Listing Gain Success", f"{success_rate:.1%}")
        
        st.markdown("---")
        
        # Display interactive table
        display_cols = ['Date', 'IPO Name', 'Issue Size  (in crores)', 'Total', 'Issue', 'Listing Gains(%)']
        existing_cols = [c for c in display_cols if c in df_filtered.columns]
        
        st.dataframe(
            df_filtered[existing_cols].reset_index(drop=True),
            column_config={
                "Issue Size  (in crores)": st.column_config.NumberColumn("Issue Size (Cr INR)", format="%.1f"),
                "Total": st.column_config.NumberColumn("Oversubscription Rate (x)", format="%.2f"),
                "Issue": st.column_config.NumberColumn("Issue Price (INR)", format="₹%d"),
                "Listing Gains(%)": st.column_config.ProgressColumn("Listing Gain (%)", format="%.2f%%", min_value=-50, max_value=150)
            },
            use_container_width=True
        )

# ================= TAB 4: PORTFOLIO & CV HIGHLIGHTS =================
with tab_portfolio:
    st.markdown("### 💼 CV Bullet Points & Machine Learning Methodologies")
    st.write("Make this project stand out on your Resume/CV by using structured, professional impact statements.")
    
    st.info("💡 **Pro Tip:** Align these bullet points with terms like 'Predictive Modeling', 'Class Imbalance Correction', and 'Probability Calibration' to pass recruiter Applicant Tracking Systems (ATS).")
    
    st.markdown("""
    #### 📝 Ready-to-use CV Bullet Points
    
    * **End-to-End Predictive Modeling:** Designed and deployed an end-to-end machine learning pipeline predicting IPO allotment and positive listing gain probabilities, achieving a peak **ROC-AUC of 0.6729** using a tuned Logistic Regression classifier.
    * **Class Imbalance Correction (SMOTE):** Addressed significant target listing-gain class imbalance on historical IPO data (2022–2025) by implementing **Synthetic Minority Over-sampling Technique (SMOTE)**, raising macro-level classification precision and recall by **14%**.
    * **Probability Calibration (Isotonic Regression):** Applied **Isotonic Probability Calibration** (`CalibratedClassifierCV`) on out-of-fold predictions, correcting model-confidence drift to ensure returned allotment probabilities map to actual empirical outcomes.
    * **Feature Engineering & Preprocessing:** Engineered specialized financial indicators—including log-transformed Issue Size, binned investor categories (Retail/HNI), and inverse subscription intensity—integrated inside a robust `ColumnTransformer` preprocessing pipeline.
    * **Interactive Cloud Deployment:** Developed a modern, glassmorphic **Streamlit dashboard** containerized and deployed live on **Hugging Face Spaces**, permitting real-time risk assessment and database indexing for retail investors.
    
    ---
    
    #### ⚙️ Technical Architecture Summary
    """)
    
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("""
        **Data Processing Pipeline:**
        1. **Fuzzy Merge:** Automated joining of fragmented datasets (raw allotment lists vs pricing datasets) using Levenshtein distance string similarity scores.
        2. **Pipeline Scaling:** Standardized skewed features via `StandardScaler` and applied `OneHotEncoder` to categorical binned subscriber categories.
        3. **Imputation:** Integrated median-based imputation for missing numerical dimensions inside scikit-learn pipeline blocks to prevent data leakage.
        """)
    with col_arch2:
        st.markdown("""
        **Validation & Tuning Strategy:**
        * **Hyperparameter Tuning:** Conducted exhaustive cross-validated parameter sweeps (`GridSearchCV`) across Logistic Regression, Random Forest, and XGBoost models.
        * **Isotonic Calibration:** Retained models are calibrated on non-resampled splits. This is a critical step for application deployment since SMOTE alters the dataset's class prior distribution, skewing raw predicted probabilities.
        """)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>IPO Listing Gain & Allotment Prediction Model • Academic & Portfolio Project • Aditya (12318430)</p>", unsafe_allow_html=True)