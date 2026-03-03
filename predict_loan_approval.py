import pandas as pd
import numpy as np
import random
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE # Used for handling class imbalance

# Synthetic Data Generation

def generate_synthetic_data(n_samples = 614):
    np.random.seed(42)
    data = {
        'Loan_ID': [f'LP00{i}' for i in range(101, n_samples + 101)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p = [0.8, 0.2]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p = [0.65, 0.35]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p = [0.58, 0.16, 0.16, 0.10]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p = [0.78, 0.22]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p = [0.14, 0.86]),
        'ApplicantIncome': np.random.randint(1500, 10000, n_samples),
        'CoapplicantIncome': np.random.randint(0, 5000, n_samples),
        'LoanAmount': np.random.randint(9, 700, n_samples) * 10,
        'Loan_Amount_Term': np.random.choice([12, 36, 60, 180, 360, 480], n_samples),
        'Credit_History': np.random.choice([1.0, 0.0], n_samples, p = [0.85, 0.15]), # Crucial feature
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p = [0.35, 0.40, 0.25]),
        'Loan_Status': np.random.choice(['Y', 'N'], n_samples, p = [0.70, 0.30]) # Target variable
    }

    data = pd.DataFrame(data)

    # Introduce NaNs and outliers for demonstration
    for col in ['Gender', 'Married', 'Dependents', 'LoanAmount', 'Credit_History']:
        data.loc[data.sample(frac = 0.05).index, col] = np.nan

    # Adjust LoanAmount based on Income to make it slightly more realistic
    data['LoanAmount'] = data.apply(lambda row: row['LoanAmount'] if row['LoanAmount'] < (row['ApplicantIncome'] / 4) else row['LoanAmount'] / 2, axis = 1)
    data.loc[data['Credit_History'] == 0.0, 'Loan_Status'] = 'N' # If poor credit, likely rejected

    return data

data = generate_synthetic_data()

print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

# Display first 10 rows
data.head(10)

# Check data types
data.info()

# EDA - Target Variable Distribution
plt.figure(figsize = (7, 5))
sns.countplot(x = 'Loan_Status', data = data, palette = 'rocket')
plt.title('Distribution of Loan Status (Target)')
plt.show()
print(f"Approval Ratio (Y/Total): {data['Loan_Status'].value_counts(normalize = True)['Y']:.2f}")

# EDA - Credit History
plt.figure(figsize = (7, 5))
sns.countplot(x = 'Credit_History', hue = 'Loan_Status', data = data, palette = 'viridis')
plt.title('Loan Status by Credit History')
plt.legend(title = 'Loan Status')
plt.show()

# EDA - Income Distribution
plt.figure(figsize = (7, 5))
sns.histplot(data['ApplicantIncome'], bins = 50, kde = True, color = '#A23B72', alpha = 0.7)
plt.title('Applicant Income Distribution')
plt.show()

# Create 'Total_Income'
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']

# Create 'Monthly_Loan_Installment_Ratio' (Proxy for DTI)
# Since LoanAmount and Term are in the data, we can create a rough monthly burden
# Using 0.08% as a simple interest factor for the term for demonstration
data['Monthly_Loan_Payment'] = data['LoanAmount'] / data['Loan_Amount_Term']
data['Income_to_Loan_Ratio'] = data['Monthly_Loan_Payment'] / data['Total_Income']
data['Income_to_Loan_Ratio'] = data['Income_to_Loan_Ratio'].replace([np.inf, -np.inf], np.nan)

# Drop original features that were combined or calculated
data = data.drop(columns = ['ApplicantIncome', 'CoapplicantIncome', 'Loan_ID', 'LoanAmount', 'Loan_Amount_Term', 'Monthly_Loan_Payment'])

# Separate x and y
x = data.drop('Loan_Status', axis = 1)
y = data['Loan_Status'].map({'Y': 1, 'N': 0}) # Map target to 0 and 1

# Define column types for the pipeline
numerical = x.select_dtypes(include = np.number).columns.tolist()
categorical = x.select_dtypes(include = 'object').columns.tolist()

# Pipelines

numerical_pipeline = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())])

categorical_pipeline = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

preprocessor = ColumnTransformer(transformers = [
        ('num', numerical_pipeline, numerical),
        ('cat', categorical_pipeline, categorical)], remainder = 'passthrough')

# Split data into train(80%) and test(20%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2, stratify = y)

# Handling Imbalanced Data (SMOTE)
x_train_prep = preprocessor.fit_transform(x_train)
x_test_prep = preprocessor.transform(x_test)

smote = SMOTE(random_state = 42)
x_train_res, y_train_res = smote.fit_resample(x_train_prep, y_train)

print(f"Original Training shape: {x_train.shape}")
print(f"Resampled Training shape: {x_train_res.shape}")
print(f"Resampled target value counts:\n{y_train_res.value_counts()}")

xgb = XGBClassifier(use_label_encoder = False, eval_metric = 'logloss', random_state = 1)

# Hyperparameter Tuning
params = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5], 'subsample': [0.7, 0.9]}

grid = GridSearchCV(estimator = xgb, param_grid = params, scoring = 'recall', cv = 3, verbose = 1, n_jobs = -1)
grid.fit(x_train_res, y_train_res)

best_model = grid.best_estimator_

print(f"\nBest Model Parameters: {grid.best_params_}")

# Model Evaluation
y_pred = best_model.predict(x_test_prep)
y_proba = best_model.predict_proba(x_test_prep)[:, 1]

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (7, 5))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Reds', cbar = False,
            xticklabels = ['Rejected (0)', 'Approved (1)'],
            yticklabels = ['Actual Rejected (0)', 'Actual Approved (1)'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# AUC Score
auc_score = roc_auc_score(y_test, y_proba)
print(f"AUC Score: {auc_score:.2f}")

# Final pipeline
final_pipeline = Pipeline(steps = [("preprocessor", preprocessor), ("model", best_model)])
final_pipeline.fit(x_train, y_train)

# Model saving
joblib.dump(final_pipeline, "PLA_model.pkl")
print("The End!")
