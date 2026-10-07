# -*- coding: utf-8 -*-
"""
Fake News Detection CLI Prediction Tool
Predicts whether a news statement/headline is True or False using the trained Logistic Regression model.
"""

import os
import sys
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'final_model.sav')

_cached_model = None

def get_model():
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_and_save_model.py first.")
        with open(MODEL_PATH, 'rb') as f:
            _cached_model = pickle.load(f)
    return _cached_model


def detecting_fake_news(var):
    """
    Predicts veracity of input text.
    Returns a dictionary with prediction label ('TRUE' or 'FALSE'),
    truth_probability (float 0..1), and fake_probability (float 0..1).
    """
    load_model = get_model()
    prediction = load_model.predict([var])[0]
    prob = load_model.predict_proba([var])[0]

    # Model classes: ['FALSE', 'TRUE']
    classes = list(load_model.classes_)
    if 'TRUE' in classes and 'FALSE' in classes:
        true_idx = classes.index('TRUE')
        false_idx = classes.index('FALSE')
        true_prob = float(prob[true_idx])
        fake_prob = float(prob[false_idx])
    else:
        fake_prob = float(prob[0])
        true_prob = float(prob[1])

    result = {
        'statement': var,
        'prediction': prediction,
        'is_real': prediction == 'TRUE',
        'truth_probability': true_prob,
        'fake_probability': fake_prob
    }

    print(f"\n==========================================")
    print(f" Statement: \"{var}\"")
    print(f" Prediction: {'REAL NEWS (True)' if prediction == 'TRUE' else 'FAKE NEWS (False)'}")
    print(f" Truth Probability Score: {true_prob * 100:.2f}%")
    print(f" Fake Probability Score:  {fake_prob * 100:.2f}%")
    print(f"==========================================\n")

    return result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        var = " ".join(sys.argv[1:])
    else:
        var = input("Please enter the news text you want to verify: ")
    detecting_fake_news(var)