# XGBoost Web Metrics Success Prediction with Risk Analysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve
)

# === Paths ===
file_path = "Dataset/web_metrics_standardized_v3.csv"
model_path = "SEMrush CodeFiles/pickel/xgboost_web_model.pkl"

# Plot saving directory
plot_dir = os.path.join(os.getcwd(), "SEMrush CodeFiles/plots/XGboost Plots")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# === Load dataset ===
df = pd.read_csv(file_path)

# Clean numeric columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '', regex=True)
        df[col] = df[col].str.replace('%', '', regex=True)
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass

df = df.fillna(0)
df = df.drop(columns=[
    'Organization Name', 'IPO Status', 'Acquisition Status',
    'Founded Date', 'Founded Date Precision', 'Announced Date', 'Last Funding Date'
], errors='ignore')

# === Balance the data ===
success_df = df[df['success'] == 1]
failure_df = df[df['success'] == 0]

failure_sample = failure_df.sample(n=len(success_df)*2, random_state=42)
balanced_df = pd.concat([success_df, success_df, failure_sample])

# === Split ===
train, test = train_test_split(df, test_size=0.3, stratify=df['success'], random_state=42)
X_test = test.drop(columns='success')
y_test = test['success']

train_X = balanced_df.drop(columns='success')
train_Y = balanced_df['success']

# === Train XGBoost model ===
model = xgb.XGBClassifier(
    max_depth=8,
    learning_rate=0.05,
    n_estimators=500,
    objective='binary:logistic',
    eval_metric='logloss',
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    random_state=42
)

model.fit(train_X, train_Y)

# === Save model ===
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Model saved to {model_path}")

# === Predict and Threshold Optimization ===
pred_proba = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
scores = [(t, f1_score(y_test, (pred_proba >= t).astype(int)),
              precision_score(y_test, (pred_proba >= t).astype(int)),
              recall_score(y_test, (pred_proba >= t).astype(int)))
          for t in thresholds]

best_threshold, best_f1, best_prec, best_rec = max(scores, key=lambda x: x[1])
y_pred = (pred_proba >= best_threshold).astype(int)

print(f"\n🔍 Best Threshold: {best_threshold:.2f} | F1: {best_f1:.4f} | Precision: {best_prec:.4f} | Recall: {best_rec:.4f}")
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# === Confusion Matrix ===
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_confusion_matrix.png"))
plt.close()

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, pred_proba)
roc_auc = roc_auc_score(y_test, pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("XGBoost ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_roc_curve.png"))
plt.close()

# === Feature Importance ===
importances = model.feature_importances_
features = train_X.columns
imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=imp_df)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_feature_importance.png"))
plt.close()

# === SHAP Analysis ===
explainer = shap.Explainer(model)
sample = X_test.sample(min(1000, len(X_test)), random_state=42)
shap_values = explainer(sample)

# SHAP bar plot
plt.figure()
shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_shap_bar.png"))
plt.close()

# SHAP dot plot
plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_shap_dot.png"))
plt.close()

# === Risk Analysis ===
print("\n🔎 Risk Level Analysis Based on Predicted Success Probability")

risk_df = pd.DataFrame({
    'Probability': pred_proba,
    'Prediction': y_pred
})

def assign_risk(prob):
    if prob >= 0.8:
        return 'Low Risk'
    elif prob >= 0.6:
        return 'Moderate Risk'
    elif prob >= 0.4:
        return 'High Risk'
    else:
        return 'Very High Risk'

risk_df['Risk Level'] = risk_df['Probability'].apply(assign_risk)
risk_counts = risk_df['Risk Level'].value_counts().reindex(
    ['Very High Risk', 'High Risk', 'Moderate Risk', 'Low Risk'], fill_value=0
)

print("\n🧾 Risk Level Summary:")
print(risk_counts)

# Barplot with fix
risk_plot_df = pd.DataFrame({
    "Risk Level": risk_counts.index,
    "Count": risk_counts.values
})

colors = {
    'Very High Risk': '#d73027',
    'High Risk': '#fc8d59',
    'Moderate Risk': '#fee08b',
    'Low Risk': '#1a9850'
}
bar_colors = [colors[r] for r in risk_plot_df['Risk Level']]

plt.figure(figsize=(8, 6))
sns.barplot(data=risk_plot_df, x='Risk Level', y='Count', palette=bar_colors)
plt.title("Startup Risk Level Distribution")
plt.ylabel("Number of Startups")
plt.xlabel("Risk Level")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_risk_analysis_distribution.png"))
plt.close()
