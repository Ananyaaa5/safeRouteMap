import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Train random forest (no scaling needed)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print('Random Forest Report')
print(classification_report(y_test, y_pred, target_names=['Safe', 'Unsafe']))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_proba))

# Save model
joblib.dump(model, 'random_forest_model.pkl')
