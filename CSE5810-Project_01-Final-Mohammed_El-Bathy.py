import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from imblearn.over_sampling import SMOTE
from sklearn import tree

# Global variables
data = None
encoded_data = None
X_resampled = None
y_resampled = None
dt_model = None
lr_model = None
rf_model = None
lr_results = None
rf_results = None

# Function 1: Import Libraries
def import_libraries():
    return "Libraries imported successfully."

# Function 2: Load the Dataset
def load_dataset():
    global data
    file_path = 'default_of_credit_card_clients.csv'
    data = pd.read_csv(file_path, header=1)
    return data

# Function 3: Display statistical summary
def display_statistics():
    global data
    if data is not None:
        return data.describe().T
    else:
        return "Data not loaded."

# Function 4: Display correlation matrix
def display_correlation():
    global data
    if data is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix of Features')
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)  
        return Image.open(buf)
    else:
        return "Data not loaded."

# Function 5: Outlier Detection and Handling
def outlier_detection():
    global data
    if data is not None:
        numeric_columns = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
                           'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 
                           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        for column in numeric_columns:
            upper_limit = data[column].quantile(0.99)
            lower_limit = data[column].quantile(0.01)
            data[column] = np.clip(data[column], lower_limit, upper_limit)
        return "Outliers capped at 1st and 99th percentiles."
    else:
        return "Data not loaded."

# Function 6: Encoding Categorical Variables
def encoding_categorical():
    global data, encoded_data
    if data is not None:
        encoded_data = pd.get_dummies(data, columns=['SEX', 'EDUCATION', 'MARRIAGE'], 
                                      drop_first=True)
        return encoded_data
    else:
        return "Data not loaded."

# Function 7: Feature Scaling
def feature_scaling():
    global encoded_data
    if encoded_data is not None:
        continuous_columns = ['LIMIT_BAL', 'AGE', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 
                              'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        scaler = StandardScaler()
        encoded_data[continuous_columns] = scaler.fit_transform(encoded_data[continuous_columns])
        return encoded_data
    else:
        return "Encoded data not available."

# Function 8: Apply SMOTE
def apply_smote():
    global encoded_data, X_resampled, y_resampled
    if encoded_data is not None:
        X = encoded_data.drop(columns=['default payment next month'])
        y = encoded_data['default payment next month']
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return f"SMOTE applied: Resampled dataset shape: {X_resampled.shape}"
    else:
        return "Encoded data not available."

# Function 9: Calculate VIF
def calculate_vif():
    global encoded_data
    if encoded_data is not None:
        numeric_data = encoded_data.select_dtypes(include=[np.number])
        X_vif = add_constant(numeric_data)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        return vif_data
    else:
        return "Encoded data not available."

# Function 12: Recalculate VIF
def recalculate_vif():
    global encoded_data
    if encoded_data is not None:
        numeric_data = encoded_data.select_dtypes(include=[np.number])
        X_vif_updated = add_constant(numeric_data)
        vif_data_updated = pd.DataFrame()
        vif_data_updated["Feature"] = X_vif_updated.columns
        vif_data_updated["VIF"] = [variance_inflation_factor(X_vif_updated.values, i) for i in range(X_vif_updated.shape[1])]
        return vif_data_updated
    else:
        return "Encoded data not available."

# Function 13: Visualize Distribution
def visualize_distribution():
    global data, encoded_data
    if data is not None and encoded_data is not None:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        sns.histplot(data['LIMIT_BAL'], bins=30, kde=True, ax=axes[0, 0]).set_title('Distribution of Credit Limit')
        sns.histplot(data['AGE'], bins=30, kde=True, ax=axes[0, 1]).set_title('Distribution of Age')
        sns.countplot(x='SEX_2', data=encoded_data, ax=axes[0, 2]).set_title('Distribution of Gender')
        sns.countplot(x='EDUCATION_2', data=encoded_data, ax=axes[1, 0]).set_title('Distribution of Education')
        sns.countplot(x='MARRIAGE_1', data=encoded_data, ax=axes[1, 1]).set_title('Distribution of Marital Status')
        sns.countplot(x='default payment next month', data=data, ax=axes[1, 2]).set_title('Default vs Non-default')
        plt.tight_layout()
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)
    else:
        return "Data not loaded."

# Function 14: Initialize and Train Decision Tree
def initialize_decision_tree():
    global dt_model, X_resampled, y_resampled
    if X_resampled is not None and y_resampled is not None:
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_resampled, y_resampled)
        return "Decision Tree Classifier initialized and trained."
    else:
        return "SMOTE resampled data not available."

# Function 15: K-Fold Cross-Validation for Decision Tree
def cross_validation_decision_tree():
    global dt_model, X_resampled, y_resampled
    if dt_model is not None and X_resampled is not None:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_scores = cross_val_score(dt_model, X_resampled, y_resampled, cv=kf, scoring='accuracy')
        roc_auc_scores = cross_val_score(dt_model, X_resampled, y_resampled, cv=kf, scoring='roc_auc')
        return f"Mean Accuracy: {accuracy_scores.mean():.4f}, Std Dev: {accuracy_scores.std():.4f}\n" \
               f"Mean ROC-AUC: {roc_auc_scores.mean():.4f}, Std Dev: {roc_auc_scores.std():.4f}"
    else:
        return "Decision Tree or resampled data not available."

# Function 16: Visualize Decision Tree Structure
def visualize_decision_tree():
    global dt_model, X_resampled
    if dt_model is not None and X_resampled is not None:
        plt.figure(figsize=(20, 10))
        tree.plot_tree(dt_model, feature_names=X_resampled.columns, class_names=['No Default', 'Default'], filled=True)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return Image.open(buf)
    else:
        return "Decision Tree model not trained."

# Function 17: Confusion Matrix Visualization for Decision Tree
def visualize_confusion_matrix_decision_tree():
    global dt_model, X_resampled, y_resampled
    if dt_model is not None and X_resampled is not None:
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_matrix(y_resampled, dt_model.predict(X_resampled)), annot=True, fmt='d', cmap='Blues')
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return Image.open(buf)
    else:
        return "Decision Tree or resampled data not available."

# Function 18: Classification Report for Decision Tree
def classification_report_dt():
    global dt_model, X_resampled, y_resampled
    if dt_model is not None and X_resampled is not None:
        y_pred = dt_model.predict(X_resampled)
        report = classification_report(y_resampled, y_pred)
        return report
    else:
        return "Decision Tree or resampled data not available."

# Function 19: Initialize and Train Logistic Regression
def initialize_logistic_regression():
    global lr_model, X_resampled, y_resampled
    if X_resampled is not None and y_resampled is not None:
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_resampled, y_resampled)
        return "Logistic Regression model initialized and trained."
    else:
        return "SMOTE resampled data not available."

# Function 20: Logistic Regression Evaluation
def evaluate_logistic_regression():
    global lr_model, X_resampled, y_resampled
    if lr_model is not None and X_resampled is not None:
        y_pred = lr_model.predict(X_resampled)
        report = classification_report(y_resampled, y_pred)
        auc = roc_auc_score(y_resampled, lr_model.predict_proba(X_resampled)[:, 1])
        return f"Classification Report:\n{report}\nROC-AUC Score: {auc:.4f}"
    else:
        return "Logistic Regression model or resampled data not available."

# Function 21: Initialize and Train Random Forest with Hyperparameter Tuning
def initialize_random_forest():
    global rf_model, X_resampled, y_resampled, rf_results
    if X_resampled is not None and y_resampled is not None:
        rf_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_resampled, y_resampled)
        
        rf_model = grid_search.best_estimator_
        rf_results = {
            "Best Parameters": grid_search.best_params_,
            "Best ROC-AUC": grid_search.best_score_
        }
        return f"Random Forest model initialized and trained with tuning. Best ROC-AUC: {rf_results['Best ROC-AUC']:.4f}"
    else:
        return "SMOTE resampled data not available."

# Function 22: Random Forest Evaluation
def evaluate_random_forest():
    global rf_model, X_resampled, y_resampled
    if rf_model is not None and X_resampled is not None:
        y_pred = rf_model.predict(X_resampled)
        report = classification_report(y_resampled, y_pred)
        auc = roc_auc_score(y_resampled, rf_model.predict_proba(X_resampled)[:, 1])
        return f"Classification Report:\n{report}\nROC-AUC Score: {auc:.4f}"
    else:
        return "Random Forest model or resampled data not available."

# Gradio Interface Setup
with gr.Blocks() as credit_default_interface:
    with gr.Tab("Import Libraries"):
        gr.Interface(fn=import_libraries, inputs=[], outputs="text").render()
    with gr.Tab("Load Dataset"):
        gr.Interface(fn=load_dataset, inputs=[], outputs="dataframe").render()
    with gr.Tab("Display Statistics"):
        gr.Interface(fn=display_statistics, inputs=[], outputs="dataframe").render()
    with gr.Tab("Display Correlation"):
        gr.Interface(fn=display_correlation, inputs=[], outputs=gr.Image(type="pil")).render()
    with gr.Tab("Outlier Detection"):
        gr.Interface(fn=outlier_detection, inputs=[], outputs="text").render()
    with gr.Tab("Encoding Categorical Variables"):
        gr.Interface(fn=encoding_categorical, inputs=[], outputs="dataframe").render()
    with gr.Tab("Feature Scaling"):
        gr.Interface(fn=feature_scaling, inputs=[], outputs="dataframe").render()
    with gr.Tab("Apply SMOTE"):
        gr.Interface(fn=apply_smote, inputs=[], outputs="text").render()
    with gr.Tab("Calculate VIF"):
        gr.Interface(fn=calculate_vif, inputs=[], outputs="dataframe").render()
    with gr.Tab("Recalculate VIF"):
        gr.Interface(fn=recalculate_vif, inputs=[], outputs="dataframe").render()
    with gr.Tab("Visualize Distribution"):
        gr.Interface(fn=visualize_distribution, inputs=[], outputs=gr.Image(type="pil")).render()
    with gr.Tab("Initialize and Train Decision Tree"):
        gr.Interface(fn=initialize_decision_tree, inputs=[], outputs="text").render()
    with gr.Tab("Decision Tree Cross-Validation"):
        gr.Interface(fn=cross_validation_decision_tree, inputs=[], outputs="text").render()
    with gr.Tab("Visualize Decision Tree"):
        gr.Interface(fn=visualize_decision_tree, inputs=[], outputs=gr.Image(type="pil")).render()
    with gr.Tab("Decision Tree Confusion Matrix"):
        gr.Interface(fn=visualize_confusion_matrix_decision_tree, inputs=[], outputs=gr.Image(type="pil")).render()
    with gr.Tab("Decision Tree Classification Report"):
        gr.Interface(fn=classification_report_dt, inputs=[], outputs="text").render()
    with gr.Tab("Initialize and Train Logistic Regression"):
        gr.Interface(fn=initialize_logistic_regression, inputs=[], outputs="text").render()
    with gr.Tab("Evaluate Logistic Regression"):
        gr.Interface(fn=evaluate_logistic_regression, inputs=[], outputs="text").render()
    with gr.Tab("Initialize and Train Random Forest"):
        gr.Interface(fn=initialize_random_forest, inputs=[], outputs="text").render()
    with gr.Tab("Evaluate Random Forest"):
        gr.Interface(fn=evaluate_random_forest, inputs=[], outputs="text").render()

# Launch Interface
credit_default_interface.launch(share=True)
