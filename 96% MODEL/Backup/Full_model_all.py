# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Handle missing values
columns_with_zeroes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='median')
X[columns_with_zeroes] = imputer.fit_transform(X[columns_with_zeroes])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gb = gb.predict(X_test)

# Feature importance
rf_feature_importance = rf.feature_importances_
gb_feature_importance = gb.feature_importances_
feature_names = X.columns

importance_df_rf = pd.DataFrame({'Feature': feature_names, 'Importance': rf_feature_importance}).sort_values(by='Importance', ascending=False)
importance_df_gb = pd.DataFrame({'Feature': feature_names, 'Importance': gb_feature_importance}).sort_values(by='Importance', ascending=False)

# Plot functions
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_feature_importance(importance_df, model_name):
    plt.figure(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Feature Importance - {model_name}')
    plt.show()

def plot_combined_roc(models, model_names, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for model, name in zip(models, model_names):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison for All Models')
    plt.legend(loc="lower right")
    plt.show()

def plot_cross_validation(models, model_names, X_train, y_train):
    scores_dict = {}
    for model, name in zip(models, model_names):
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        scores_dict[name] = scores
    scores_df = pd.DataFrame(scores_dict)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=scores_df)
    plt.title('Cross-Validation Accuracy Scores for All Models')
    plt.ylabel('Accuracy')
    plt.show()

def plot_learning_curve(model, model_name, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation Score')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.show()

# Confusion matrices
plot_confusion_matrix(y_test, y_pred_lr, 'Logistic Regression')
plot_confusion_matrix(y_test, y_pred_dt, 'Decision Tree')
plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest')
plot_confusion_matrix(y_test, y_pred_gb, 'Gradient Boosting')

# Feature importance
plot_feature_importance(importance_df_rf, 'Random Forest')
plot_feature_importance(importance_df_gb, 'Gradient Boosting')

# ROC curves for all models
models = [lr, dt, rf, gb]
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
plot_combined_roc(models, model_names, X_test, y_test)

# Cross-validation boxplot
plot_cross_validation(models, model_names, X_train, y_train)

# Learning curves for Random Forest and Gradient Boosting
plot_learning_curve(rf, 'Random Forest', X_scaled, y)
plot_learning_curve(gb, 'Gradient Boosting', X_scaled, y)
