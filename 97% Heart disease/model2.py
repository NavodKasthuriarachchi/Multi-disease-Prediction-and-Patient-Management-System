# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    print("-" * 60)

# Identify the best model based on accuracy
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model_name]

print(f"The best model is {best_model_name} with an accuracy of {best_accuracy * 100:.2f}%")

# Save the best Random Forest model as a pickle file
if best_model_name == 'Random Forest':
    joblib.dump(models['Random Forest'], 'heart_disease_model.pkl')
    print("Random Forest model saved as heart_disease_model.pkl")


