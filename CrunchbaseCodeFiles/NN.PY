# ✅ UPDATED NN.PY (Neural Network with SMOTE and Plot Saving)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

# Paths
file_path = r"C:\Users\aanch\Downloads\Startup\Data\updated_startup_data.csv"
model_path = r"C:\Users\aanch\Downloads\Startup\Crunch\pickel\nn_model.h5"
plot_dir = r"C:\Users\aanch\Downloads\Startup\Crunch\plots\NN"
os.makedirs(plot_dir, exist_ok=True)

# Load & preprocess data
df = pd.read_csv(file_path)
df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')
df['funding_rounds'] = pd.to_numeric(df['funding_rounds'], errors='coerce')
df['founded_at'] = pd.to_datetime(df['founded_at'], errors='coerce')
df['first_funding_at'] = pd.to_datetime(df['first_funding_at'], errors='coerce')
df['days_to_first_funding'] = (df['first_funding_at'] - df['founded_at']).dt.days

df['funding_per_round'] = df['funding_total_usd'] / (df['funding_rounds'] + 1e-6)
df['founded_at'] = df['founded_at'].apply(lambda x: x.toordinal() if not pd.isna(x) else 0)
df['first_funding_at'] = df['first_funding_at'].apply(lambda x: x.toordinal() if not pd.isna(x) else 0)

for col in ['category_list', 'country_code', 'city']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

num_cols = ['funding_total_usd', 'funding_rounds', 'days_to_first_funding', 'funding_per_round']
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

X = df.drop(columns='success')
y = df['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale & SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_res, y_train_res = SMOTE(sampling_strategy=0.5, random_state=42).fit_resample(X_train_scaled, y_train)

# Model
model = Sequential([
    Dense(256, activation='swish', input_shape=(X_train_res.shape[1],), kernel_regularizer=l2(0.01)),
    BatchNormalization(), Dropout(0.5),
    Dense(128, activation='swish', kernel_regularizer=l2(0.01)),
    BatchNormalization(), Dropout(0.4),
    Dense(64, activation='swish', kernel_regularizer=l2(0.01)),
    BatchNormalization(), Dropout(0.3),
    Dense(32, activation='swish'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.AUC(name='pr_auc', curve='PR')]
)

callbacks = [
    EarlyStopping(monitor='val_pr_auc', patience=20, restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint(model_path, monitor='val_pr_auc', save_best_only=True, mode='max')
]

history = model.fit(
    X_train_res, y_train_res,
    validation_data=(X_test_scaled, y_test),
    epochs=200, batch_size=128,
    callbacks=callbacks, verbose=1
)

# Save model
print(f"✅ Model saved to {model_path}")

# Evaluation
y_pred_proba = model.predict(X_test_scaled).flatten()
thresholds = np.linspace(0.1, 0.9, 50)
f1_scores = [f1_score(y_train, (model.predict(X_train_scaled).flatten() >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred = (y_pred_proba >= best_threshold).astype(int)

print(f"\n📊 Optimal threshold: {best_threshold:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Neural Network Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "nn_confusion_matrix.png"))
plt.close()

# ROC + PR Curves
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "nn_roc_pr_curves.png"))
plt.close()

# Training History
plt.figure(figsize=(12, 10))
metrics_to_plot = ['loss', 'pr_auc', 'precision', 'recall']
for i, metric in enumerate(metrics_to_plot):
    plt.subplot(2, 2, i + 1)
    plt.plot(history.history[metric], label='Train')
    plt.plot(history.history[f'val_{metric}'], label='Val')
    plt.title(metric.upper())
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "nn_training_history.png"))
plt.close()

print(f"📁 All plots saved to: {plot_dir}")
