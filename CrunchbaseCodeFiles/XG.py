

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pickle

# Paths
file_path = r"C:\Users\aanch\Downloads\Startup\Data\updated_startup_data.csv"
model_path = r"C:\Users\aanch\Downloads\Startup\Crunch\pickel\xgboost.pkl"
plot_dir = r"C:\Users\aanch\Downloads\Startup\Crunch\plots\XG"
os.makedirs(plot_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(file_path)
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')
df['founded_at'] = pd.to_datetime(df['founded_at'], errors='coerce')
df['first_funding_at'] = pd.to_datetime(df['first_funding_at'], errors='coerce')
df['founded_at'] = df['founded_at'].fillna(pd.Timestamp('1970-01-01')).apply(lambda x: x.toordinal())
df['first_funding_at'] = df['first_funding_at'].fillna(pd.Timestamp('1970-01-01')).apply(lambda x: x.toordinal())

for col in ['category_list', 'country_code', 'city']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
df = df.fillna(0)

# Balance the dataset
success_df = df[df['success'] == 1]
failure_df = df[df['success'] == 0]
failure_sample = failure_df.sample(n=len(success_df)*2, random_state=42)
balanced_df = pd.concat([success_df, success_df, failure_sample])

train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['success'])
test_X = test.drop(columns='success')
test_Y = test['success']

train_X = balanced_df.drop(columns='success')
train_Y = balanced_df['success']

# Train XGBoost model
model = xgb.XGBClassifier(
    max_depth=8,
    learning_rate=0.05,
    n_estimators=500,
    objective='binary:logistic',
    eval_metric='auc',
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1,
    random_state=42,
    verbosity=1
)

model.fit(train_X, train_Y, eval_set=[(test_X, test_Y)], verbose=100)

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Model saved to {model_path}")

# Predictions
pred_proba = model.predict_proba(test_X)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [(t, metrics.f1_score(test_Y, (pred_proba >= t).astype(int))) for t in thresholds]
optimal_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
pred_final = (pred_proba >= optimal_threshold).astype(int)

print(f"\n📊 Optimal Threshold: {optimal_threshold:.4f}, F1: {best_f1:.4f}")

# Evaluation
print("\nClassification Report:\n", metrics.classification_report(test_Y, pred_final))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = metrics.confusion_matrix(test_Y, pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_confusion_matrix.png"))
plt.close()

# ROC Curve
roc_auc = metrics.roc_auc_score(test_Y, pred_proba)
fpr, tpr, _ = metrics.roc_curve(test_Y, pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_roc_curve.png"))
plt.close()

# SHAP
explainer = shap.TreeExplainer(model)
test_sample = test_X.sample(min(1000, len(test_X)), random_state=42)
shap_values = explainer.shap_values(test_sample)

plt.figure()
shap.summary_plot(shap_values, test_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_shap_bar.png"))
plt.close()

plt.figure()
shap.summary_plot(shap_values, test_sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "xgb_shap_dot.png"))
plt.close()

print(f"📁 All plots saved to: {plot_dir}")
