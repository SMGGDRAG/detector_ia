# backend/app.py
import os
import sys
import pickle
import numpy as np
import pandas as pd
import math
import string
from collections import Counter
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


# Configurar NLTK
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

setup_nltk()

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)

# ============================================================
# FUNCIONES DE EXTRACCIÓN DE FEATURES
# ============================================================

def extract_features(text, language='english'):
    if not text or len(text.strip()) < 30:
        return None
    
    try:
        sents = sent_tokenize(text, language=language)
        words_raw = word_tokenize(text, language=language)
        words = [w for w in words_raw if any(c.isalpha() for c in w)]
        
        if len(words) < 10 or len(sents) == 0:
            return None
        
        stops = set(stopwords.words(language))
        non_stop = [w for w in words if w.lower() not in stops]
        unique = set(w.lower() for w in words)
        
        lexical_density = len(non_stop) / len(words) if len(words) > 0 else 0
        ttr = len(unique) / len(words) if len(words) > 0 else 0
        
        sent_lengths = []
        for s in sents:
            s_words = [w for w in word_tokenize(s, language=language) if any(c.isalpha() for c in w)]
            sent_lengths.append(len(s_words))
        
        sentence_mean_len = np.mean(sent_lengths) if sent_lengths else 0
        sentence_std_len = np.std(sent_lengths) if sent_lengths else 0
        
        words_lower = [w.lower() for w in words if w.isalpha()]
        bigrams = list(zip(words_lower[:-1], words_lower[1:]))
        if bigrams:
            counts = Counter(bigrams)
            total = sum(counts.values())
            probs = [c/total for c in counts.values()]
            bigram_entropy = -sum(p * math.log2(p) for p in probs)
        else:
            bigram_entropy = 0
        
        punct_chars = sum(1 for ch in text if ch in string.punctuation)
        punct_ratio = punct_chars / len(words) if len(words) > 0 else 0
        
        long_words = [w for w in words if len(w) > 6]
        long_words_ratio = len(long_words) / len(words) if len(words) > 0 else 0
        total_words = len(words)
        
        paragraphs = text.count('\n\n') + 1 if text.strip() else 1
        paragraphs_per_sent = paragraphs / len(sents) if len(sents) > 0 else 0
        words_longer6_ratio = long_words_ratio
        
        return {
            'lexical_density': lexical_density,
            'ttr': ttr,
            'sentence_mean_len': sentence_mean_len,
            'sentence_std_len': sentence_std_len,
            'bigram_entropy': bigram_entropy,
            'punct_ratio': punct_ratio,
            'long_words_ratio': long_words_ratio,
            'total_words': total_words,
            'paragraphs_per_sent': paragraphs_per_sent,
            'words_longer6_ratio': words_longer6_ratio
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# ============================================================
# CLASIFICADOR
# ============================================================

class IAClassifier:
    def __init__(self, model_path="backend/ia_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.is_loaded = False
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"⚠️ No se encontró {self.model_path}")
            print("Ejecuta primero: python backend/train_model.py")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_cols = model_data['feature_cols']
            self.is_loaded = True
            print(f"✅ Modelo cargado: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def predict(self, text):
        if not self.is_loaded:
            if not self.load_model():
                return None, None, None
        
        features = extract_features(text)
        if features is None:
            return None, None, None
        
        features_df = pd.DataFrame([features])[self.feature_cols]
        features_scaled = self.scaler.transform(features_df)
        
        proba_ia = self.model.predict_proba(features_scaled)[0, 1]
        prediction = 1 if proba_ia > 0.5 else 0
        
        # Feature contributions (simplificado)
        feature_contributions = []
        importance = self.model.feature_importances_
        
        for i, col in enumerate(self.feature_cols):
            direction = 'IA' if prediction == 1 else 'Humano'
            feature_contributions.append({
                'name': col.replace('_', ' ').title(),
                'impact': float(importance[i]),
                'direction': direction
            })
        
        feature_contributions.sort(key=lambda x: x['impact'], reverse=True)
        
        return prediction, proba_ia, feature_contributions[:3]

# ============================================================
# INICIALIZAR
# ============================================================

classifier = IAClassifier()

def get_message(prediction, confidence):
    if prediction == 1 and confidence > 0.8:
        return "⚠️ Alta probabilidad de IA. Texto con estructura uniforme y vocabulario predecible."
    elif prediction == 0 and confidence > 0.8:
        return "✅ Muy probablemente humano. Texto con variabilidad natural."
    else:
        return "📝 Caso ambiguo. El texto tiene características mixtas."

# ============================================================
# ENDPOINTS
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': classifier.is_loaded})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if len(text) < 50:
            return jsonify({'success': False, 'error': 'El texto debe tener al menos 50 caracteres'}), 400
        
        prediction, confidence, features = classifier.predict(text)
        
        if prediction is None:
            return jsonify({'success': False, 'error': 'No se pudo procesar el texto'}), 400
        
        return jsonify({
            'success': True,
            'is_ia': bool(prediction),
            'veredicto': '🤖 IA GENERADO' if prediction == 1 else '👤 HUMANO',
            'confidence': float(confidence),
            'confidence_percent': f"{confidence*100:.1f}%",
            'top_features': features,
            'message': get_message(prediction, confidence)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("🔍 DETECTOR DE IA - SERVIDOR")
    print("="*50)
    print("🚀 Iniciando en http://localhost:5000")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)