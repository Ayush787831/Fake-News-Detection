# -*- coding: utf-8 -*-
"""
Training script for Fake News Detection models.
Trains on train.csv and saves modern final_model.sav and model.pkl.
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Download NLTK data
print("Downloading NLTK resources...", flush=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, 'train.csv')
test_path = os.path.join(base_dir, 'test.csv')
valid_path = os.path.join(base_dir, 'valid.csv')

print(f"Loading datasets from {base_dir}...", flush=True)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
valid_df = pd.read_csv(valid_path)

# Combine train and valid for stronger training set or keep separate
train_df = train_df.dropna(subset=['Statement', 'Label'])
test_df = test_df.dropna(subset=['Statement', 'Label'])
valid_df = valid_df.dropna(subset=['Statement', 'Label'])

# Normalize labels to string 'TRUE' / 'FALSE' or bool
train_df['Label'] = train_df['Label'].astype(str).str.strip().str.upper()
test_df['Label'] = test_df['Label'].astype(str).str.strip().str.upper()
valid_df['Label'] = valid_df['Label'].astype(str).str.strip().str.upper()

print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}, Valid samples: {len(valid_df)}", flush=True)
print("Class distribution in train:", train_df['Label'].value_counts().to_dict(), flush=True)

# Build Logistic Regression Pipeline (with TF-IDF n-grams 1-4)
print("\n--- Training Logistic Regression Pipeline (final_model.sav) ---", flush=True)
logr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 4), use_idf=True, smooth_idf=True, sublinear_tf=True, max_features=40000)),
    ('clf', LogisticRegression(C=1.5, max_iter=1000, random_state=42))
])

logr_pipeline.fit(train_df['Statement'], train_df['Label'])
y_pred_logr = logr_pipeline.predict(test_df['Statement'])
acc_logr = accuracy_score(test_df['Label'], y_pred_logr)
f1_logr = f1_score(test_df['Label'], y_pred_logr, pos_label='TRUE', average='binary')
print(f"Logistic Regression Test Accuracy: {acc_logr:.4f}, F1: {f1_logr:.4f}", flush=True)
print(classification_report(test_df['Label'], y_pred_logr), flush=True)

# Save final_model.sav
final_model_path = os.path.join(base_dir, 'final_model.sav')
with open(final_model_path, 'wb') as f:
    pickle.dump(logr_pipeline, f)
print(f"Saved {final_model_path}", flush=True)

# Build PassiveAggressive / Enhanced Pipeline (model.pkl)
print("\n--- Training PassiveAggressive / Classifier Pipeline (model.pkl) ---", flush=True)
pac_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_df=0.85, sublinear_tf=True, max_features=35000)),
    ('clf', PassiveAggressiveClassifier(max_iter=100, random_state=42, C=0.5))
])

pac_pipeline.fit(train_df['Statement'], train_df['Label'])
y_pred_pac = pac_pipeline.predict(test_df['Statement'])
acc_pac = accuracy_score(test_df['Label'], y_pred_pac)
print(f"PassiveAggressive Test Accuracy: {acc_pac:.4f}", flush=True)
print(classification_report(test_df['Label'], y_pred_pac), flush=True)

# Save model.pkl
model_pkl_path = os.path.join(base_dir, 'model.pkl')
with open(model_pkl_path, 'wb') as f:
    pickle.dump(pac_pipeline, f)
print(f"Saved {model_pkl_path}", flush=True)

print("\nModel training & saving completed successfully!", flush=True)
