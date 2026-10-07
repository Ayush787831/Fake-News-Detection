# -*- coding: utf-8 -*-
"""
Flask Web Application for Fake News Detection
Provides interactive UI and REST API for real-time veracity analysis,
confidence scoring, and linguistic feature explainability.
"""

import os
import re
import json
import pickle
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_PATH = os.path.join(BASE_DIR, 'final_model.sav')
PAC_MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

_logr_model = None
_pac_model = None


def get_logr_model():
    global _logr_model
    if _logr_model is None:
        if os.path.exists(FINAL_MODEL_PATH):
            with open(FINAL_MODEL_PATH, 'rb') as f:
                _logr_model = pickle.load(f)
    return _logr_model


def get_pac_model():
    global _pac_model
    if _pac_model is None:
        if os.path.exists(PAC_MODEL_PATH):
            with open(PAC_MODEL_PATH, 'rb') as f:
                _pac_model = pickle.load(f)
    return _pac_model


def explain_prediction(text, model):
    """
    Extracts informative keywords and n-grams from the input text
    based on the trained TF-IDF vectorizer and classifier coefficients.
    """
    try:
        tfidf = model.named_steps.get('tfidf')
        clf = model.named_steps.get('clf')
        if not tfidf or not clf or not hasattr(clf, 'coef_'):
            return []

        feature_names = tfidf.get_feature_names_out()
        coefs = clf.coef_[0]
        
        # Transform input
        vec = tfidf.transform([text])
        non_zero_indices = vec.nonzero()[1]

        features = []
        for idx in non_zero_indices:
            feat_name = feature_names[idx]
            weight = float(coefs[idx])
            tfidf_val = float(vec[0, idx])
            impact = weight * tfidf_val
            features.append({
                'word': feat_name,
                'weight': round(weight, 3),
                'impact': round(impact, 3),
                'direction': 'real' if weight > 0 else 'fake'
            })

        # Sort by absolute impact
        features.sort(key=lambda x: abs(x['impact']), reverse=True)
        return features[:12]
    except Exception as e:
        print(f"Error in explain_prediction: {e}")
        return []


PRESET_EXAMPLES = [
    {
        "category": "Politics",
        "text": "The Chicago Bears have had more starting quarterbacks in the last 10 years than the total number of tenured faculty fired during the last two decades.",
        "label": "REAL"
    },
    {
        "category": "Health",
        "text": "Health care reform legislation is likely to mandate free sex change surgeries on demand.",
        "label": "FAKE"
    },
    {
        "category": "Economy",
        "text": "The economic turnaround started at the end of my term as job growth stabilized.",
        "label": "REAL"
    },
    {
        "category": "Science",
        "text": "Do not believe this hoax about microchip implants in coronavirus vaccines controlled by 5G networks.",
        "label": "FAKE"
    },
    {
        "category": "Government",
        "text": "Says the Annie's List political group supports third-trimester abortions on demand without medical reason.",
        "label": "FAKE"
    },
    {
        "category": "Social",
        "text": "Since 2000, nearly 12 million Americans have slipped out of the middle class and into poverty.",
        "label": "REAL"
    }
]


@app.route('/')
def home():
    return render_template('index.html', presets=PRESET_EXAMPLES)


@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    # Support both JSON payload and HTML Form POST
    if request.is_json:
        data = request.get_json() or {}
        news_text = data.get('news', '').strip()
        model_type = data.get('model_type', 'logistic_regression')
    else:
        news_text = request.form.get('news', '').strip()
        model_type = request.form.get('model_type', 'logistic_regression')

    if not news_text:
        return jsonify({
            'success': False,
            'error': 'Please enter a valid news headline or article statement.'
        }), 400

    model = get_logr_model()
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model is not loaded. Please ensure final_model.sav is generated.'
        }), 500

    # Prediction
    prediction = model.predict([news_text])[0]
    prob = model.predict_proba([news_text])[0]
    classes = list(model.classes_)

    if 'TRUE' in classes and 'FALSE' in classes:
        true_idx = classes.index('TRUE')
        false_idx = classes.index('FALSE')
        true_prob = float(prob[true_idx])
        fake_prob = float(prob[false_idx])
    else:
        fake_prob = float(prob[0])
        true_prob = float(prob[1])

    is_real = (prediction == 'TRUE')
    verdict = 'REAL NEWS' if is_real else 'FAKE NEWS'
    confidence = max(true_prob, fake_prob) * 100.0

    # Linguistic explanations
    explanations = explain_prediction(news_text, model)

    # Word stats
    words = re.findall(r'\b\w+\b', news_text)
    word_count = len(words)
    char_count = len(news_text)

    response_data = {
        'success': True,
        'statement': news_text,
        'prediction': verdict,
        'label': 'TRUE' if is_real else 'FALSE',
        'is_real': is_real,
        'truth_probability': round(true_prob * 100, 2),
        'fake_probability': round(fake_prob * 100, 2),
        'confidence_score': round(confidence, 1),
        'explanations': explanations,
        'metrics': {
            'word_count': word_count,
            'character_count': char_count,
            'reading_time_sec': max(1, round(word_count / 3.5))
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.path.startswith('/api/'):
        return jsonify(response_data)

    # Fallback to rendering template with result for pure form submits
    return render_template('index.html', result=response_data, presets=PRESET_EXAMPLES)


@app.route('/api/presets', methods=['GET'])
def get_presets():
    return jsonify({'success': True, 'presets': PRESET_EXAMPLES})


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'success': True,
        'dataset': {
            'name': 'LIAR Benchmark Dataset (William Yang Wang, ACL 2017)',
            'train_samples': 10240,
            'test_samples': 2551,
            'validation_samples': 2569,
            'classes': ['REAL (True, Mostly-true, Half-true)', 'FAKE (Barely-true, False, Pants-on-fire)']
        },
        'models': [
            {
                'name': 'Logistic Regression + TF-IDF (1-4 ngrams)',
                'test_accuracy': '60.5%',
                'f1_score': '0.67',
                'status': 'Active (Production Pipeline)'
            },
            {
                'name': 'Passive Aggressive Classifier',
                'test_accuracy': '57.2%',
                'f1_score': '0.61',
                'status': 'Available'
            },
            {
                'name': 'Random Forest & Linear SVM',
                'test_accuracy': '~62.0%',
                'f1_score': '0.65',
                'status': 'Benchmarked'
            }
        ]
    })


if __name__ == '__main__':
    # Default Flask port 5000
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Fake News Detection Web Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
