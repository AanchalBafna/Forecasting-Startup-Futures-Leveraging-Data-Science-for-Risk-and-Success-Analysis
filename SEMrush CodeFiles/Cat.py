# Required Libraries
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import os

# File paths
file_path = "Dataset/web_metrics_standardized_v3.csv"
model_path = "SEMrush CodeFiles/pickel/catboost_balanced_model.pkl"

# Plot saving directory
plot_dir = os.path.join(os.getcwd(), "SEMrush CodeFiles/plots/Catboost Plots")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(file_path)

# Clean columns with commas and percentage symbols in numeric fields
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '', regex=True)
        df[col] = df[col].str.replace('%', '', regex=True)
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass

# Dropping unnecessary columns
columns_to_drop = [
    'Organization Name', 'IPO Status', 'Acquisition Status',
    'Founded Date', 'Founded Date Precision', 'Announced Date', 'Last Funding Date'
]
df = df.drop(columns=columns_to_drop, errors='ignore')

# Handling missing values
df = df.fillna(0)

# Class distribution before balancing
print("Class distribution before balancing:")
print(df['success'].value_counts())
print(f"Success rate: {df['success'].mean():.2%}")

# Create a balanced dataset
success_df = df[df['success'] == 1]
failure_df = df[df['success'] == 0]

if len(success_df) < len(failure_df):
    failure_sample = failure_df.sample(n=len(success_df)*2, random_state=42)
    balanced_df = pd.concat([success_df, success_df, failure_sample])
else:
    success_sample = success_df.sample(n=len(failure_df)*2, random_state=42)
    balanced_df = pd.concat([failure_df, failure_df, success_sample])

# Class distribution after balancing
print("\nClass distribution after balancing:")
print(balanced_df['success'].value_counts())
print(f"Success rate: {balanced_df['success'].mean():.2%}")

# Split data
X = df.drop(columns=['success'])
y = df['success']
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=y)
test_X = test_df.drop(columns=['success'])
test_Y = test_df['success']

train_balanced = balanced_df.copy()
train_X_balanced = train_balanced.drop(columns=['success'])
train_Y_balanced = train_balanced['success']

# Detect categorical features
cat_features = [i for i, col in enumerate(train_X_balanced.columns) if train_X_balanced[col].dtype == 'object']

# Convert NaNs in categorical features to string "missing"
for col in train_X_balanced.columns:
    if train_X_balanced[col].dtype == 'object':
        train_X_balanced[col] = train_X_balanced[col].astype(str).fillna("missing")
        test_X[col] = test_X[col].astype(str).fillna("missing")

# Build CatBoost model
modelCatBoost = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=100,
    eval_metric='F1',
    l2_leaf_reg=3,
    bagging_temperature=1,
    one_hot_max_size=10
)

# Fit model
modelCatBoost.fit(
    train_X_balanced,
    train_Y_balanced,
    cat_features=cat_features,
    eval_set=(test_X, test_Y),
    early_stopping_rounds=50,
    verbose=100
)

# Save the trained model
with open(model_path, 'wb') as f:
    pickle.dump(modelCatBoost, f)
print(f"Model saved as {model_path}")

# Prediction
pred_proba = modelCatBoost.predict_proba(test_X)[:, 1]

# Optimal threshold
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []

for threshold in thresholds:
    pred = (pred_proba >= threshold).astype(int)
    f1 = metrics.f1_score(test_Y, pred)
    precision = metrics.precision_score(test_Y, pred)
    recall = metrics.recall_score(test_Y, pred)
    f1_scores.append((threshold, f1, precision, recall))

optimal_threshold, best_f1, best_precision, best_recall = max(f1_scores, key=lambda x: x[1])
print(f"\nOptimal threshold: {optimal_threshold:.4f}, Best F1: {best_f1:.4f}")
print(f"Precision at optimal threshold: {best_precision:.4f}")
print(f"Recall at optimal threshold: {best_recall:.4f}")

# Final predictions
pred = (pred_proba >= optimal_threshold).astype(int)

# Evaluation
accuracy = metrics.accuracy_score(test_Y, pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(metrics.classification_report(test_Y, pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = metrics.confusion_matrix(test_Y, pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix with Optimal Threshold')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
plt.close()

# ROC Curve
roc_auc = metrics.roc_auc_score(test_Y, pred_proba)
fpr, tpr, _ = metrics.roc_curve(test_Y, pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "roc_curve.png"))
plt.close()

# Feature Importance
feature_importance = modelCatBoost.get_feature_importance()
feature_names = train_X_balanced.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "catboost_feature_importance.png"))
plt.close()

# SHAP Analysis
print("\nSHAP Analysis:")
explainer = shap.TreeExplainer(modelCatBoost)
sample_size = min(1000, len(test_X))
test_sample = test_X.sample(sample_size, random_state=42)
shap_values = explainer.shap_values(test_sample)

# SHAP bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, test_sample, plot_type="bar", show=False)
plt.title("Feature Importance with SHAP Values (Bar Plot)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "shap_summary_bar.png"))
plt.close()

# SHAP dot plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, test_sample, show=False)
plt.title("SHAP Value Distribution by Feature")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "shap_summary_dot.png"))
plt.close()

print("\n✅ All plots saved in:", plot_dir)



# =====================
# RISK ANALYSIS
# =====================
print("\n🔍 Risk Level Analysis Based on Predicted Success Probability")

# Assign risk categories
risk_df = pd.DataFrame({
    'Probability': pred_proba,
    'Prediction': pred
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

# Count by risk level
risk_counts = risk_df['Risk Level'].value_counts().reindex(
    ['Very High Risk', 'High Risk', 'Moderate Risk', 'Low Risk'], fill_value=0
)

print("\n🧾 Risk Level Summary:")
print(risk_counts)

# ✅ Future-proof plotting
plot_data = pd.DataFrame({
    'Risk Level': risk_counts.index,
    'Count': risk_counts.values
})

colors = {
    'Very High Risk': '#d73027',   # red
    'High Risk': '#fc8d59',        # orange
    'Moderate Risk': '#fee08b',    # yellow
    'Low Risk': '#1a9850'          # green
}
bar_colors = [colors[r] for r in plot_data['Risk Level']]

plt.figure(figsize=(8, 6))
sns.barplot(data=plot_data, x='Risk Level', y='Count', palette=bar_colors)
plt.title("Startup Risk Level Distribution")
plt.ylabel("Number of Startups")
plt.xlabel("Risk Level")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "risk_analysis_distribution.png"))
plt.close()


