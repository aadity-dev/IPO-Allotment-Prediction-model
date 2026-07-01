

# evaluation.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, accuracy_score

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

def plot_target_distribution(df, file_path):
    """Plots and saves a clear distribution of the target variable."""
    plt.figure(figsize=(8, 6))
    
    # Create a temporary column with meaningful labels for plotting
    plot_df = df.copy()
    plot_df['Outcome'] = plot_df['Target'].map({0: 'No Gain', 1: 'Listing Gain'})
    
    ax = sns.countplot(
        data=plot_df, 
        x='Outcome', 
        hue='Outcome', # Recommended for future seaborn versions
        order=['No Gain', 'Listing Gain'], 
        palette={'No Gain': '#E57373', 'Listing Gain': '#81C784'}, 
        legend=False
    )
    
    ax.set_title('Distribution of IPO Listing Outcomes', fontsize=16)
    ax.set_xlabel('Outcome', fontsize=12)
    ax.set_ylabel('Number of IPOs', fontsize=12)

    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points', fontsize=12)
    
    plt.savefig(file_path)
    plt.close()
    print(f"Improved target distribution plot saved to {file_path}")

def plot_confusion_matrix(y_true, y_pred, file_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Gain', 'Gain'], 
                yticklabels=['No Gain', 'Gain'])
    plt.title('Confusion Matrix of the Best Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved to {file_path}")

def plot_precision_recall_curve(y_true, y_scores, file_path):
    """Plots and saves the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve of the Best Model')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()
    print(f"Precision-Recall curve saved to {file_path}")

def plot_correlation_heatmap(df, numerical_features, file_path):
    """Plots and saves the feature correlation heatmap."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(file_path)
    plt.close()
    print(f"Feature correlation heatmap saved to {file_path}")

def plot_prf_vs_threshold(y_true, y_scores, file_path):
    """Plots Precision, Recall, and F1-Score against the decision threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold.
    # We will use the thresholds array which is one element shorter.
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = f1_scores[:-1] # align with thresholds
    precision = precision[:-1] # align with thresholds
    recall = recall[:-1] # align with thresholds

    # Find the threshold that maximizes F1 score
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]

    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, precision, 'b--', label='Precision')
    plt.plot(thresholds, recall, 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', label='F1-Score')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold (F1-Max) \n at {best_threshold:.2f}')
    plt.title('Precision, Recall, and F1-Score vs. Decision Threshold')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()
    print(f"Precision, Recall, F1 vs. Threshold plot saved to {file_path}")

def main():
    """Main function to load models and generate evaluations."""
    # Load Data and recreate the same test set
    data_path = "ipo.csv"
    df = pd.read_csv(data_path, header=1)
    df = create_features(df)

    # --- Generate and Save All Plots ---
    model_dir = "."
    
    # Create improved target distribution plot
    target_dist_path = f"{model_dir}/target_distribution_improved.png"
    plot_target_distribution(df, target_dist_path)

    numerical_features = ['Log_Issue_Size', 'Inverse_Subscription', 'gmp_percentage', 'revenue_crores', 'profit_margin']
    features = numerical_features + ['Investor_Category']
    target = 'Target'
    
    X = df[features]
    y = df[target]

    # Use the same split to get the identical test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Calculate and Display Accuracy for All Models ---
    print("--- Model Accuracy Scores ---")
    models_to_evaluate = {
        "Logistic Regression": f"{model_dir}/logistic_regression_model.pkl",
        "Random Forest": f"{model_dir}/random_forest_model.pkl",
        "XGBoost": f"{model_dir}/xgboost_model.pkl"
    }

    for name, path in models_to_evaluate.items():
        try:
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name}: {accuracy:.4f}")
        except FileNotFoundError:
            print(f"Could not find model for {name} at {path}")
        except Exception as e:
            print(f"Could not evaluate {name}. Error: {e}")

    # --- Evaluate Best Model and Generate Plots ---
    print("\n--- Best Model Evaluation ---")
    best_model_path = f"{model_dir}/best_model.pkl"
    try:
        calibrated_pipeline = joblib.load(best_model_path)
        print("Successfully loaded the best model.")
        
        y_pred_best = calibrated_pipeline.predict(X_test)
        y_pred_proba_best = calibrated_pipeline.predict_proba(X_test)[:, 1]
        best_accuracy = accuracy_score(y_test, y_pred_best)
        print(f"Best Model Accuracy: {best_accuracy:.4f}")

        # Set file paths for plots
        cm_path = f"{model_dir}/confusion_matrix.png"
        pr_curve_path = f"{model_dir}/precision_recall_curve.png"
        corr_heatmap_path = f"{model_dir}/correlation_heatmap.png"
        prf_threshold_path = f"{model_dir}/prf_vs_threshold.png"
        
        # Generate plots
        plot_confusion_matrix(y_test, y_pred_best, cm_path)
        plot_precision_recall_curve(y_test, y_pred_proba_best, pr_curve_path)
        plot_correlation_heatmap(df, numerical_features, corr_heatmap_path)
        plot_prf_vs_threshold(y_test, y_pred_proba_best, prf_threshold_path)

    except FileNotFoundError:
        print(f"Error: Best model file not found at {best_model_path}")
    except Exception as e:
        print(f"Could not evaluate best model. Error: {e}")

if __name__ == "__main__":
    main()
