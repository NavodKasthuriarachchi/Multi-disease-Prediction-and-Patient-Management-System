# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pickle

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Handle missing values (replace zero values in columns with median values, except for 'Pregnancies' and 'Outcome')
columns_with_zeroes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='median')
X[columns_with_zeroes] = imputer.fit_transform(X[columns_with_zeroes])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred_gb)

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

# Function to perform cross-validation
def perform_cross_validation(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean(), scores.std()

# 1. Logistic Regression
print("Logistic Regression Performance Metrics:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# ROC and AUC for Logistic Regression
y_pred_prob_lr = lr.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_pred_prob_lr, 'Logistic Regression')

# Cross-validation for Logistic Regression
lr_cv_mean, lr_cv_std = perform_cross_validation(lr, X_train, y_train)

# 2. Decision Tree
print("\nDecision Tree Performance Metrics:")
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# ROC and AUC for Decision Tree
y_pred_prob_dt = dt.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_pred_prob_dt, 'Decision Tree')

# Cross-validation for Decision Tree
dt_cv_mean, dt_cv_std = perform_cross_validation(dt, X_train, y_train)

# 3. Random Forest
print("\nRandom Forest Performance Metrics:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# ROC and AUC for Random Forest
y_pred_prob_rf = rf.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_pred_prob_rf, 'Random Forest')

# Cross-validation for Random Forest
rf_cv_mean, rf_cv_std = perform_cross_validation(rf, X_train, y_train)

# Feature Importance for Random Forest
rf_feature_importance = rf.feature_importances_
feature_names = X.columns
importance_df_rf = pd.DataFrame({'Feature': feature_names, 'Importance': rf_feature_importance}).sort_values(by='Importance', ascending=False)

# 4. Gradient Boosting
print("\nGradient Boosting Performance Metrics:")
print(classification_report(y_test, y_pred_gb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))

# ROC and AUC for Gradient Boosting
y_pred_prob_gb = gb.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_pred_prob_gb, 'Gradient Boosting')

# Cross-validation for Gradient Boosting
gb_cv_mean, gb_cv_std = perform_cross_validation(gb, X_train, y_train)

# Feature Importance for Gradient Boosting
gb_feature_importance = gb.feature_importances_
importance_df_gb = pd.DataFrame({'Feature': feature_names, 'Importance': gb_feature_importance}).sort_values(by='Importance', ascending=False)

# Display Feature Importance for Random Forest
print("\nRandom Forest Feature Importance:")
print(importance_df_rf)

# Display Feature Importance for Gradient Boosting
print("\nGradient Boosting Feature Importance:")
print(importance_df_gb)

# Overfitting/Underfitting Check - Random Forest
rf_train_accuracy = accuracy_score(y_train, rf.predict(X_train))
rf_test_accuracy = rf_accuracy

print(f"\nRandom Forest Train Accuracy: {rf_train_accuracy}")
print(f"Random Forest Test Accuracy: {rf_test_accuracy}")

# Save the best model (Random Forest) and the scaler as pickle files
with open('diabetes_rf_model_tuned.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Tuned model and scaler saved as pickle files!")
