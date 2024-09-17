import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'Medicaldataset.csv'
df = pd.read_csv(file_path)

# Preprocessing the dataset

# Encoding the target variable (Result)
label_encoder = LabelEncoder()
df['Result'] = label_encoder.fit_transform(df['Result'])  # 0 for negative, 1 for positive

# Separating features and target variable
X = df.drop('Result', axis=1)
y = df['Result']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest classifier to determine feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Extracting feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display feature importance
print("Feature Importance:")
print(feature_importance)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

import joblib

# Save the model to a file
model_filename = 'heart_disease_model.pkl'
joblib.dump(rf, model_filename)