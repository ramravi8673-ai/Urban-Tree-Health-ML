import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import joblib

# -------------------------------------------------
# 1️⃣ Load Dataset
# -------------------------------------------------

print("📂 Loading dataset...")
df = pd.read_csv("tree_census_processed.csv", nrows=50000)

required_columns = ['tree_dbh', 'stump_diam', 'health']
df = df[required_columns].dropna()

print("Dataset Shape After Cleaning:", df.shape)

# -------------------------------------------------
# 2️⃣ Encode Target Variable
# -------------------------------------------------

le = LabelEncoder()
df['health'] = le.fit_transform(df['health'])

X = df[['tree_dbh', 'stump_diam']]
y = df['health']

# -------------------------------------------------
# 3️⃣ Train-Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# 4️⃣ Logistic Regression (Baseline)
# -------------------------------------------------

print("📘 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_accuracy)

# -------------------------------------------------
# 5️⃣ Random Forest with Hyperparameter Tuning
# -------------------------------------------------

print("🌲 Performing Hyperparameter Tuning for Random Forest...")

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

rf_pred = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Tuned Random Forest Accuracy:", rf_accuracy)

# -------------------------------------------------
# 6️⃣ Save Model Comparison
# -------------------------------------------------

comparison_df = pd.DataFrame({
    "Model": ["Random Forest (Tuned)", "Logistic Regression"],
    "Accuracy": [rf_accuracy, lr_accuracy]
})

comparison_df.to_csv("model_comparison.csv", index=False)

# Save best model accuracy
with open("model_accuracy.txt", "w") as f:
    f.write(str(rf_accuracy))

# -------------------------------------------------
# 7️⃣ Save Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(y_test, rf_pred)
pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)

# -------------------------------------------------
# 8️⃣ Save Feature Importance
# -------------------------------------------------

importances = best_rf.feature_importances_
features = ['tree_dbh', 'stump_diam']

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})

importance_df.to_csv("feature_importance.csv", index=False)

# -------------------------------------------------
# 9️⃣ Save ROC Curve Data (Multiclass)
# -------------------------------------------------

print("📈 Generating ROC Data...")

y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

y_score = best_rf.predict_proba(X_test)

roc_data = []

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)

    temp_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr
    })

    temp_df.to_csv(f"roc_curve_class_{i}.csv", index=False)

    roc_data.append({
        "Class": i,
        "AUC": roc_auc
    })

roc_summary = pd.DataFrame(roc_data)
roc_summary.to_csv("roc_summary.csv", index=False)

print("✅ ROC Data Saved")

# -------------------------------------------------
# 🔟 Save Best Model & Encoder
# -------------------------------------------------

joblib.dump(best_rf, "tree_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Best Model Saved Successfully")
print("🚀 FULL TRAINING PIPELINE COMPLETED")
