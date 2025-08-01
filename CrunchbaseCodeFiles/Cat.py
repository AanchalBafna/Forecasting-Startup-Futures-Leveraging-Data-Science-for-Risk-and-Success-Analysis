# Importing required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import shap
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle

# Paths
file_path = "Dataset/updated_startup_data.csv"
model_path = "CrunchbaseCodeFiles/pickel/cat.pkl"
plot_dir = "CrunchbaseCodeFiles/plots/Catboost"
os.makedirs(plot_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(file_path)

# Preprocessing
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')
df['founded_at'] = pd.to_datetime(df['founded_at'], errors='coerce')
df['first_funding_at'] = pd.to_datetime(df['first_funding_at'], errors='coerce')
df['founded_at'] = df['founded_at'].fillna(pd.Timestamp('1970-01-01')).apply(lambda x: x.toordinal())
df['first_funding_at'] = df['first_funding_at'].fillna(pd.Timestamp('1970-01-01')).apply(lambda x: x.toordinal())

le = LabelEncoder()
for col in ['category_list', 'country_code', 'city']:
    df[col] = le.fit_transform(df[col].astype(str))
df = df.fillna(0)

# Balance the dataset
success_df = df[df['success'] == 1]
failure_df = df[df['success'] == 0]
failure_sample = failure_df.sample(n=len(success_df)*2, random_state=42)
balanced_df = pd.concat([success_df, success_df, failure_sample])

# Train/test split
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['success'])
test_X = test.drop(columns='success')
test_Y = test['success']

train_X_balanced = balanced_df.drop(columns='success')
train_Y_balanced = balanced_df['success']

# Train CatBoost model
modelCatBoost = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    random_seed=5,
    verbose=False,
    eval_metric='F1',
    l2_leaf_reg=3,
    bagging_temperature=1,
    one_hot_max_size=10
)

cat_features = ['category_list', 'country_code', 'city']
modelCatBoost.fit(
    train_X_balanced,
    train_Y_balanced,
    cat_features=cat_features,
    eval_set=(test_X, test_Y),
    early_stopping_rounds=50,
    verbose=100
)

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(modelCatBoost, f)
print(f"✅ Model saved to {model_path}")

# Prediction & threshold tuning
pred_proba = modelCatBoost.predict_proba(test_X)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [(t, metrics.f1_score(test_Y, (pred_proba >= t).astype(int)),
              metrics.precision_score(test_Y, (pred_proba >= t).astype(int)),
              metrics.recall_score(test_Y, (pred_proba >= t).astype(int))) for t in thresholds]
optimal_threshold, best_f1, best_precision, best_recall = max(f1_scores, key=lambda x: x[1])
print(f"\n📊 Optimal threshold: {optimal_threshold:.4f}, F1: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")

# Final prediction
pred_cat = (pred_proba >= optimal_threshold).astype(int)
accuracy = metrics.accuracy_score(test_Y, pred_cat)
print(f"\n✅ Accuracy: {accuracy:.4f}\n")
print("📄 Classification Report:\n", classification_report(test_Y, pred_cat))

# Plot: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_Y, pred_cat)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix with Optimal Threshold')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
plt.close()

# SHAP Analysis
explainer = shap.TreeExplainer(modelCatBoost)
sample_size = min(1000, len(test_X))
test_sample = test_X.sample(sample_size, random_state=42)
shap_values = explainer.shap_values(test_sample)

# Plot: SHAP Bar
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, test_sample, plot_type="bar", show=False)
plt.title("Feature Importance with SHAP Values (Bar Plot)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "shap_bar_plot.png"))
plt.close()

# Plot: SHAP Dot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, test_sample, show=False)
plt.title("SHAP Value Distribution by Feature")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "shap_dot_plot.png"))
plt.close()

# Plot: Feature Importance
feature_importance = modelCatBoost.get_feature_importance()
feature_names = train_X_balanced.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "catboost_feature_importance.png"))
plt.close()

# Plot: ROC Curve
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

print(f"📁 All plots saved to: {plot_dir}")
