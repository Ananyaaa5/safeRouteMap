import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load data
df = pd.read_csv("features_for_model_final.csv")

# Create binary target based on crime count threshold
df['unsafe'] = (df['crime_count'] > 3).astype(int)

# Select features avoiding leakage
features = [
    'Latitude', 'Longitude', 'hour_of_day', 'day_of_week',
    'time_slot_morning', 'time_slot_afternoon', 'time_slot_evening', 'time_slot_night',
    'primary_type_encoded', 'time_of_day_encoded'
]
X = df[features]
y = df['unsafe']

# Train/test split stratified
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print('Logistic Regression Report')
print(classification_report(y_test, y_pred, target_names=['Safe', 'Unsafe']))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_proba))

# Save model and scaler
joblib.dump(model, 'logreg_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
