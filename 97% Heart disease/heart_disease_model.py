# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# Load the dataset
df = pd.read_csv('Medicaldataset.csv')

# Data preprocessing
# Encode the target variable ('Result') - converting 'positive' and 'negative' to 1 and 0
label_encoder = LabelEncoder()
df['Result'] = label_encoder.fit_transform(df['Result'])  # 'positive' -> 1, 'negative' -> 0

# Separate features (X) and target (y)
X = df.drop('Result', axis=1)  # Features
y = df['Result']               # Target

# Split the data into training and test sets (70% training, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature values (scale data for better model performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# List of models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
}

# Dictionary to store the accuracy of each model
model_accuracies = {}

# Function to plot confusion matrix heatmap
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_prob, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Function to plot feature importance
def plot_feature_importance(importance_df, model_name):
    plt.figure(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Feature Importance - {model_name}')
    plt.show()

# Function to plot Learning Curve
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

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and store it
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = accuracy
    
    # Print classification report for each model
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    # Plot ROC curve
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_pred_prob, model_name)
    
    # Plot learning curve
    plot_learning_curve(model, model_name, X_train, y_train)
    
    print("-" * 60)

# Identify the best model based on accuracy
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model_name]

print(f"The best model is {best_model_name} with an accuracy of {best_accuracy * 100:.2f}%")

# Perform cross-validation for Random Forest and Logistic Regression
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation results for {model_name}:")
    print(f"Mean Accuracy: {cv_scores.mean() * 100:.2f}%")
    print(f"Standard Deviation: {cv_scores.std() * 100:.2f}%")
    print("-" * 60)

# Save the best Random Forest model as a pickle file
if best_model_name == 'Random Forest':
    joblib.dump(models['Random Forest'], 'heart_disease_model.pkl')
    print("Random Forest model saved as heart_disease_model.pkl")

# Feature importance for Random Forest
if best_model_name == 'Random Forest':
    rf = models['Random Forest']
    rf_feature_importance = rf.feature_importances_
    feature_names = X.columns
    importance_df_rf = pd.DataFrame({'Feature': feature_names, 'Importance': rf_feature_importance}).sort_values(by='Importance', ascending=False)
    plot_feature_importance(importance_df_rf, 'Random Forest')

# Overfitting/Underfitting Check for Random Forest
if best_model_name == 'Random Forest':
    rf_train_accuracy = accuracy_score(y_train, rf.predict(X_train))
    rf_test_accuracy = accuracy_score(y_test, rf.predict(X_test))
    print(f"\nRandom Forest Train Accuracy: {rf_train_accuracy * 100:.2f}%")
    print(f"Random Forest Test Accuracy: {rf_test_accuracy * 100:.2f}%")

    if rf_train_accuracy > rf_test_accuracy:
        print("The model may be overfitting.")
    else:
        print("The model generalizes well.")
