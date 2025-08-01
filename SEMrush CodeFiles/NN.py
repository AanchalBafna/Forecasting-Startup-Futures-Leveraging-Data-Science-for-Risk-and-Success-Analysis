# Neural Network for Web Metrics with Risk Analysis and Plot Saving

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import shap
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

# Set paths
file_path = "Dataset/web_metrics_standardized_v3.csv"
model_path = "SEMrush CodeFiles/pickel/nn_model.h5"

# Plot saving directory
plot_dir = os.path.join(os.getcwd(), "SEMrush CodeFiles/plots/Neural Network Plots")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Load and clean data
df = pd.read_csv(file_path)

def clean_data(df):
    float_cols = [
        'Number of Articles', 'SEMrush - Monthly Visits', 'SEMrush - Average Visits (6 months)',
        'SEMrush - Monthly Visits Growth', 'SEMrush - Visit Duration',
        'SEMrush - Visit Duration Growth', 'SEMrush - Page Views / Visit Growth',
        'SEMrush - Bounce Rate', 'SEMrush - Global Traffic Rank',
        'SEMrush - Bounce Rate Growth', 'SEMrush - Monthly Rank Change (#)',
        'SEMrush - Monthly Rank Growth', 'Money Raised Currency (in USD)',
        'Number of Investors'
    ]
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')
    df[['Founded Date', 'Last Funding Date']] = df[['Founded Date', 'Last Funding Date']].apply(pd.to_datetime, errors='coerce')
    df['Founded Year'] = df['Founded Date'].dt.year
    df['Last Funding Year'] = df['Last Funding Date'].dt.year

    return df.drop(columns=[
        'Organization Name', 'IPO Status', 'Acquisition Status',
        'Founded Date Precision', 'Announced Date',
        'Founded Date', 'Last Funding Date'
    ], errors='ignore').fillna(0)

df_clean = clean_data(df)

# Split
X = df_clean.drop(columns='success')
y = df_clean['success']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100, batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# Save model
model.save(model_path)
print(f"\n✅ Model saved to: {model_path}")

# Predict probabilities
y_pred_proba = model.predict(X_test_scaled).flatten()

# Threshold optimization
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [(t, metrics.f1_score(y_test, (y_pred_proba >= t).astype(int))) for t in thresholds]
best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
y_pred = (y_pred_proba >= best_threshold).astype(int)

# Evaluation
print(f"\n🔍 Best Threshold: {best_threshold:.2f}, F1 Score: {best_f1:.4f}")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.2%}")
print(f"AUC-ROC: {metrics.roc_auc_score(y_test, y_pred):.2%}")
print("\n📄 Classification Report:\n", metrics.classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("NN Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "nn_confusion_matrix.png"))
plt.close()

# SHAP Analysis
print("\n📊 SHAP Analysis...")
background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X_test_scaled[:100])

# SHAP dot plot
shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=X.columns, plot_type="dot", show=False)
plt.title("SHAP Summary (Dot)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "nn_shap_summary_dot.png"))
plt.close()

# SHAP bar plot
shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=X.columns, plot_type="bar", show=False)
plt.title("SHAP Summary (Bar)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "nn_shap_summary_bar.png"))
plt.close()

# ======================
# ✅ RISK ANALYSIS
# ======================
print("\n🔎 Risk Level Analysis Based on Predicted Success Probability")

risk_df = pd.DataFrame({
    'Probability': y_pred_proba,
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

print("\n🧾 Risk Level Summary:\n", risk_counts)

# Risk barplot
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
plt.title("Startup Risk Level Distribution (NN)")
plt.ylabel("Number of Startups")
plt.xlabel("Risk Level")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "nn_risk_analysis_distribution.png"))
plt.close()

print("\n✅ Neural network analysis and risk prediction complete.")
print(f"📁 All plots saved to: {plot_dir}")
