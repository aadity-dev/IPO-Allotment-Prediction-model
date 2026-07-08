---
title: IPO Allotment & Listing Gain Prediction Model
emoji: 📈
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: false
license: mit
---

# 📈 IPO Allotment & Listing Gain Predictor

An end-to-end Machine Learning project to predict listing gain probabilities and allotment potential for Initial Public Offerings (IPOs) based on historical dataset records (2022–2025). The app is built with **Streamlit** and incorporates predictive modeling pipelines trained on calibrated classifiers.

*Developed by Aditya (12318430) as an Academic & Portfolio Project under the guidance of Mahipal Sir.*

---

## 🚀 Live App Features

1. **Interactive Allotment Planner:** Input parameters like oversubscription rates, investor category (Retail/HNI/Institutional), issue size, Grey Market Premium (GMP %), revenue, and net profit margins to calculate listing probabilities.
2. **Model Performance & Diagnostics:** Explore evaluation metrics (ROC-AUC, confusion matrices, and precision-recall graphs) comparing calibrated Logistic Regression, Random Forest, and XGBoost models.
3. **Historical Explorer:** Full-text search and filter historical IPO records to analyze performance trends.
4. **CV & Portfolio Hub:** Ready-to-use resume bullet points and detailed technical architecture.

---

## 🛠️ Machine Learning Pipeline

- **Fuzzy String Merging:** Fragmented datasets (allotment subscriber lists and stock exchange pricing charts) are matched using Levenshtein distance string similarity scores.
- **Preprocessing Pipeline:** Outlier log-transforms (`Log_Issue_Size`), scaling, median imputation, and categorical One-Hot encoding are structured into a scikit-learn `ColumnTransformer`.
- **Handling Class Imbalance:** Corrected listing gain class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) to raise macro-level recall.
- **Probability Calibration:** Models are calibrated using **Isotonic Regression** (`CalibratedClassifierCV`) on the original class ratios. This ensures predicted probabilities accurately map to real-world listing gain rates.
- **Tuned Model Scores (ROC-AUC):**
  - **Logistic Regression (Winner):** `0.6729` (Test Accuracy: `62.50%`)
  - **XGBoost Classifier:** `0.6207`
  - **Random Forest Classifier:** `0.5852`

---

## 💻 Running the App Locally

### 1. Set Up Environment
Ensure you have Python 3.11+ installed. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
Install the required packages from the requirements list:
```bash
pip install -r requirements.txt
```

### 3. Run Retraining & Diagnostics (Optional)
If you want to train and tune the models locally to generate new serialized pipeline pickles:
```bash
python train1.py
python evaluation.py
```

### 4. Run the Streamlit Dashboard
Launch the local web application:
```bash
streamlit run app.py
```

---

## ☁️ Deployment Guide: Hugging Face Spaces

Deploying this app live to **Hugging Face Spaces** takes less than 3 minutes. Hugging Face will automatically read the YAML metadata at the top of this `README.md` and spin up a secure Streamlit instance!

### Step 1: Create a Space on Hugging Face
1. Log in to [Hugging Face](https://huggingface.co/) (create a free account if you don't have one).
2. Click on your profile icon in the top right and select **New Space** (or go to `huggingface.co/new-space`).
3. Set your **Space Name** (e.g., `ipo-allotment-predictor`).
4. Select **Streamlit** as the SDK.
5. Keep it **Public** (recommended for portfolio showcases) or set to Private.
6. Click **Create Space**.

### Step 2: Upload Files to the Space
You can deploy your files using either the Hugging Face Web UI or Git.

#### Method A: Direct Upload via Web Browser (Easiest)
1. In your newly created Hugging Face Space, click on the **Files and versions** tab at the top.
2. Click **Add file** -> **Upload files**.
3. Drag and drop the following files from your project folder:
   - `app.py` (The main web app)
   - `requirements.txt` (Declares dependencies)
   - `best_model.pkl` (The winner calibrated Logistic Regression model)
   - `random_forest_model.pkl` (The calibrated Random Forest model)
   - `xgboost_model.pkl` (The calibrated XGBoost model)
   - `ipo.csv` (Historical dataset)
   - All evaluation plots (Optional, to display in the diagnostics tab):
     - `target_distribution_improved.png`
     - `confusion_matrix.png`
     - `precision_recall_curve.png`
     - `correlation_heatmap.png`
     - `prf_vs_threshold.png`
4. Write a commit message (e.g., `Initial commit`) and click **Commit changes to main**.

#### Method B: Deploy using Git (Command Line)
1. Copy the clone command from your Space's page (e.g., `git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`).
2. Run the command in your terminal to clone the space repository locally.
3. Copy the project files (listed in Method A) into the cloned directory.
4. Run git commands to commit and push:
   ```bash
   git add .
   git commit -m "Deploy IPO predictor app"
   git push
   ```

### Step 3: Wait for Build
Hugging Face will automatically detect `requirements.txt`, install all libraries, and make the app live at:
`https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
