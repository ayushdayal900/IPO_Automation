import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# Replace 'your_dataset.csv' with your actual file path
df = pd.read_csv('../data.csv')

# Display first few rows
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Replace NA values with 0
df = df.replace('NA', 0)
df = df.fillna(0)

# Define FOCUSED feature columns (only the specified features)
feature_columns = [
    'P/E',
    'Mar Capitalization Rs.Cr.',
    'Net Profit of last quarter Rs. Cr.',
    'Quarterly Profit Variation %',
    'Quarterly Sales Rs.Cr.',
    'Quarterly Sales Variation %',
    'Issue Price (Rs)',
    'ROCE %'
]

print(f"\nUsing {len(feature_columns)} focused features for better accuracy")

# Create a copy of original dataframe for later use
df_original = df.copy()

# Extract features and target
X = df[feature_columns].copy()
y = df['Classification'].copy()

# Convert features to numeric (in case some values are still strings)
for col in feature_columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

# Convert target to string to handle mixed types
y = y.astype(str).str.strip()

# Remove any rows where Classification is not S, F, or N
valid_classes = ['S', 'F', 'N']
mask = y.isin(valid_classes)

# Filter and reset index
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df_filtered = df[mask].reset_index(drop=True)

print(f"\nFiltered dataset - keeping only S, F, N classifications")
print(f"Rows after filtering: {len(y)}")

print("\n" + "="*60)
print("Features shape:", X.shape)
print("Target distribution:")
print(y.value_counts())
print("="*60)

# Custom scoring function: N predicted as S is considered correct
def custom_accuracy(y_true, y_pred):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
        elif true == 'N' and pred == 'S':  # N predicted as S is correct
            correct += 1
    return correct / len(y_true)

# Use RobustScaler instead of StandardScaler (better for outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

# Split the data into training and testing sets (75-25 split for better test set)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Handle class imbalance using SMOTE
print("\n" + "="*60)
print("Applying SMOTE for Class Imbalance...")
print("="*60)
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Training samples after SMOTE: {len(X_train_balanced)}")
print("Balanced class distribution:")
print(pd.Series(y_train_balanced).value_counts())

print("\n" + "="*60)
print("Training Optimized AdaBoost Model...")
print("="*60)

# Use optimized parameters with regularization to prevent overfitting
weak_learner = DecisionTreeClassifier(
    max_depth=3,  # Reduced from 5 to prevent overfitting
    min_samples_split=10,  # Increased to prevent overfitting
    min_samples_leaf=5,  # Increased to prevent overfitting
    max_features='sqrt',  # Use only subset of features
    random_state=42
)

# Create AdaBoost Classifier with stronger regularization
adaboost_model = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=100,  # Reduced from 150
    learning_rate=0.5,  # Reduced from 0.8 for better generalization
    random_state=42,
    algorithm='SAMME'
)

# Train the model on balanced data
adaboost_model.fit(X_train_balanced, y_train_balanced)

print("\n" + "="*60)
print("Optimized Parameters Used:")
print("="*60)
print(f"n_estimators: 100")
print(f"learning_rate: 0.5")
print(f"max_depth: 3")
print(f"min_samples_split: 10")
print(f"min_samples_leaf: 5")
print(f"max_features: sqrt")
print(f"SMOTE applied: Yes")
print(f"Scaler: RobustScaler")

print("\n" + "="*60)
print("Training completed with optimized parameters!")
print("="*60)

# Perform cross-validation on training set
cv_scores = cross_val_score(adaboost_model, X_train_balanced, y_train_balanced, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Make predictions
y_train_pred = adaboost_model.predict(X_train)
y_test_pred = adaboost_model.predict(X_test)

# Make predictions on the ENTIRE filtered dataset
y_all_pred = adaboost_model.predict(X_scaled)

# Calculate standard accuracies
train_accuracy_standard = accuracy_score(y_train, y_train_pred)
test_accuracy_standard = accuracy_score(y_test, y_test_pred)

# Calculate custom accuracies (N->S is correct)
train_accuracy_custom = custom_accuracy(y_train, y_train_pred)
test_accuracy_custom = custom_accuracy(y_test, y_test_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Training Accuracy (Standard): {train_accuracy_standard * 100:.2f}%")
print(f"Testing Accuracy (Standard): {test_accuracy_standard * 100:.2f}%")
print("\n--- With Custom Rule (N→S considered correct) ---")
print(f"Training Accuracy (Custom): {train_accuracy_custom * 100:.2f}%")
print(f"Testing Accuracy (Custom): {test_accuracy_custom * 100:.2f}%")

# Check for overfitting
overfitting_gap = train_accuracy_standard - test_accuracy_standard
print(f"\nOverfitting Gap: {overfitting_gap * 100:.2f}%")
if overfitting_gap > 0.15:
    print("⚠️  Model shows some overfitting, but regularization applied")
else:
    print("✓ Good generalization - minimal overfitting")

# Count N->S predictions
n_to_s_count = sum(1 for true, pred in zip(y_test, y_test_pred) if true == 'N' and pred == 'S')
print(f"\nN predicted as S (bonus correct): {n_to_s_count} cases")

# Confusion Matrix
print("\n" + "="*60)
print("CONFUSION MATRIX (Test Set)")
print("="*60)
cm = confusion_matrix(y_test, y_test_pred, labels=['S', 'F', 'N'])
print("\n          Predicted")
print("           S    F    N")
print("-" * 25)
classes = ['S', 'F', 'N']
for i, actual_class in enumerate(classes):
    print(f"Actual {actual_class} |", end="")
    for j in range(len(classes)):
        print(f"{cm[i][j]:4}", end=" ")
    print()

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT (Standard)")
print("="*60)
print(classification_report(y_test, y_test_pred, target_names=['Success (S)', 'Fail (F)', 'Normal (N)'], zero_division=0))

# Calculate custom metrics for each class
print("\n" + "="*60)
print("CUSTOM METRICS (N→S considered correct)")
print("="*60)

for cls in ['S', 'F', 'N']:
    true_positives = sum(1 for t, p in zip(y_test, y_test_pred) if t == cls and p == cls)
    if cls == 'N':
        # For N, also count N->S as correct
        true_positives += sum(1 for t, p in zip(y_test, y_test_pred) if t == 'N' and p == 'S')
    
    actual_count = sum(1 for t in y_test if t == cls)
    predicted_count = sum(1 for p in y_test_pred if p == cls)
    
    recall = true_positives / actual_count if actual_count > 0 else 0
    precision = true_positives / predicted_count if predicted_count > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nClass {cls}:")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1-Score: {f1:.2f}")
    print(f"  Support: {actual_count}")

# Feature Importance (from the AdaBoost model)
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)
feature_importance = adaboost_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(feature_importance_df.to_string(index=False))

# Sample predictions
print("\n" + "="*60)
print("SAMPLE PREDICTIONS (First 15 Test Samples)")
print("="*60)
sample_results = pd.DataFrame({
    'Actual': y_test.values[:15],
    'Predicted': y_test_pred[:15],
    'Standard_Correct': y_test.values[:15] == y_test_pred[:15],
    'Custom_Correct': [True if (t == p or (t == 'N' and p == 'S')) else False 
                       for t, p in zip(y_test.values[:15], y_test_pred[:15])]
})
print(sample_results.to_string(index=False))

# Determine which samples are in train/test set
train_indices = X_train.index.tolist()
test_indices = X_test.index.tolist()
dataset_labels = ['Train' if i in train_indices else 'Test' for i in range(len(y))]

# Calculate custom correctness for all predictions
custom_correct = [True if (t == p or (t == 'N' and p == 'S')) else False 
                  for t, p in zip(y.values, y_all_pred)]

# Calculate confidence scores for each prediction
prediction_proba = adaboost_model.predict_proba(X_scaled)
confidence_scores = [max(proba) for proba in prediction_proba]

# Save ALL predictions to CSV for the filtered dataset
all_predictions = pd.DataFrame({
    'S.No.': df_filtered['S.No.'].values if 'S.No.' in df_filtered.columns else range(1, len(y) + 1),
    'Name': df_filtered['Name'].values if 'Name' in df_filtered.columns else [''] * len(y),
    'Actual': y.values,
    'Predicted': y_all_pred,
    'Confidence': [f"{conf:.2%}" for conf in confidence_scores],
    'Standard_Correct': y.values == y_all_pred,
    'Custom_Correct': custom_correct,
    'Dataset': dataset_labels
})
all_predictions.to_csv('adaboost_predictions.csv', index=False)
print(f"\nAll {len(y)} predictions saved to 'adaboost_predictions.csv'")

# Model parameters
print("\n" + "="*60)
print("FINAL MODEL PARAMETERS")
print("="*60)
print(f"Number of estimators: {adaboost_model.n_estimators}")
print(f"Weak learner: Decision Tree")
print(f"Max depth of trees: {adaboost_model.estimator_.max_depth}")
print(f"Min samples split: {adaboost_model.estimator_.min_samples_split}")
print(f"Min samples leaf: {adaboost_model.estimator_.min_samples_leaf}")
print(f"Max features: {adaboost_model.estimator_.max_features}")
print(f"Learning rate: {adaboost_model.learning_rate}")
print(f"Algorithm: {adaboost_model.algorithm}")
print(f"Feature Scaling: RobustScaler applied")
print(f"Class Balancing: SMOTE applied")
print(f"Number of features used: {len(feature_columns)}")

# Save the trained model and scaler
import pickle
with open('adaboost_model.pkl', 'wb') as f:
    pickle.dump(adaboost_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\nModel saved to 'adaboost_model.pkl'")
print("Scaler saved to 'scaler.pkl'")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total predictions: {len(y)}")
print(f"Standard accuracy (train): {train_accuracy_standard * 100:.2f}%")
print(f"Standard accuracy (test): {test_accuracy_standard * 100:.2f}%")
print(f"Custom accuracy (train): {train_accuracy_custom * 100:.2f}%")
print(f"Custom accuracy (test): {test_accuracy_custom * 100:.2f}%")
print(f"Accuracy improvement: +{(test_accuracy_custom - test_accuracy_standard) * 100:.2f}%")
print(f"N→S cases treated as correct: {n_to_s_count}")
print(f"Cross-validation score: {cv_scores.mean():.2f}")

print("\n" + "="*60)
print("PROCESS COMPLETED!")
print("="*60)