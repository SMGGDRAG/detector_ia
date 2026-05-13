# backend/train_model_v2_improved.py
"""
TRAINER MEJORADO
- Cross-validation para detectar overfitting
- Guarda datos de entrenamiento para SHAP
- Matriz de confusión y análisis detallado
- Detecta automáticamente datos sintéticos
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             classification_report, confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, data_path="data/train_v2_drcat_02.csv"):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_path = "backend/ia_model.pkl"
        self.X_train_scaled = None
        self.y_train = None
        self.X_test_scaled = None
        self.y_test = None
        self.is_synthetic = False
        
    def load_and_prepare_data(self):
        """Carga el CSV y prepara las features estilométricas"""
        print("="*60)
        print(" CARGANDO DATASET")
        print("="*60)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"""
             No se encontró el archivo: {self.data_path}
            
            Por favor, asegúrate de que:
            1. El archivo esté en: {os.path.abspath(self.data_path)}
            2. Tenga columnas 'text' (o similar) y 'label' (o similar)
            """)
        
        # Cargar CSV
        df = pd.read_csv(self.data_path)
        print(f"Archivo cargado: {len(df)} filas")
        print(f"Columnas disponibles: {list(df.columns)}")
        
        # Identificar columnas
        text_col = None
        label_col = None
        
        for col in ['text', 'content', 'essay', 'full_text', 'response']:
            if col in df.columns:
                text_col = col
                break
        
        for col in ['label', 'is_ia', 'generated', 'source', 'class']:
            if col in df.columns:
                label_col = col
                break
        
        if text_col is None or label_col is None:
            raise ValueError(f"Columnas no encontradas. Disponibles: {list(df.columns)}")
        
        print(f"Usando texto de: '{text_col}'")
        print(f"Usando etiqueta de: '{label_col}'")
        
        # Preparar datos
        df_clean = pd.DataFrame()
        df_clean['text'] = df[text_col].astype(str)
        df_clean['is_ia'] = df[label_col].astype(int)
        
        # Filtrar textos muy cortos
        df_clean['word_count'] = df_clean['text'].apply(lambda x: len(str(x).split()))
        print(f"\nDistribución de longitud:")
        print(f"   Media: {df_clean['word_count'].mean():.0f} palabras")
        print(f"   Min: {df_clean['word_count'].min()} palabras")
        print(f"   Max: {df_clean['word_count'].max()} palabras")
        
        min_words = 50
        df_filtered = df_clean[df_clean['word_count'] >= min_words].copy()
        print(f"\nFiltrados textos < {min_words} palabras:")
        print(f"   Antes: {len(df_clean)} textos")
        print(f"   Después: {len(df_filtered)} textos")
        print(f"   Eliminados: {len(df_clean) - len(df_filtered)} textos")
        
        # Balance de clases
        print(f"\nBalance de clases:")
        n_human = sum(df_filtered['is_ia'] == 0)
        n_ia = sum(df_filtered['is_ia'] == 1)
        print(f"   Humano (0): {n_human} ({n_human/len(df_filtered)*100:.1f}%)")
        print(f"   IA (1): {n_ia} ({n_ia/len(df_filtered)*100:.1f}%)")
        
        if abs(n_human - n_ia) / len(df_filtered) > 0.2:
            print("\nBalanceando clases...")
            min_samples = min(n_human, n_ia)
            df_human = df_filtered[df_filtered['is_ia'] == 0].sample(n=min_samples, random_state=42)
            df_ia = df_filtered[df_filtered['is_ia'] == 1].sample(n=min_samples, random_state=42)
            df_filtered = pd.concat([df_human, df_ia]).sample(frac=1, random_state=42)
            print(f"Balanceado: {len(df_filtered)} muestras (50% cada clase)")
        
        return df_filtered
    
    def extract_features_from_texts(self, df):
        """Extrae features estilométricas"""
        print("\n" + "="*60)
        print("🔧 EXTRACCIÓN DE FEATURES ESTILOMÉTRICAS")
        print("="*60)
        print("Esto puede tomar unos minutos...")
        
        import math
        import string
        from collections import Counter
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize, word_tokenize
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        def extract_single_text_features(text):
            try:
                sents = sent_tokenize(text, language='english')
                words = [w for w in word_tokenize(text, language='english') if any(c.isalpha() for c in w)]
                
                if len(words) < 10 or len(sents) == 0:
                    return None
                
                stops = set(stopwords.words('english'))
                non_stop = [w for w in words if w.lower() not in stops]
                unique = set(w.lower() for w in words)
                
                lexical_density = len(non_stop) / len(words) if len(words) > 0 else 0
                ttr = len(unique) / len(words) if len(words) > 0 else 0
                
                sent_lengths = [len(word_tokenize(s, language='english')) for s in sents]
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
                    'words_longer6_ratio': long_words_ratio
                }
            except:
                return None
        
        features_list = []
        self.feature_cols = ['lexical_density', 'ttr', 'sentence_mean_len', 'sentence_std_len',
                              'bigram_entropy', 'punct_ratio', 'long_words_ratio', 'total_words',
                              'paragraphs_per_sent', 'words_longer6_ratio']
        
        for idx, row in df.iterrows():
            feats = extract_single_text_features(row['text'])
            if feats:
                feats['is_ia'] = row['is_ia']
                features_list.append(feats)
            
            if (idx + 1) % 500 == 0:
                print(f"   Procesados {idx + 1} textos...")
        
        features_df = pd.DataFrame(features_list)
        print(f"\nFeatures extraídas: {len(features_df)} textos válidos")
        print(f"Eliminados: {len(df) - len(features_df)} textos")
        
        features_df.to_csv('data/phase2_features.csv', index=False)
        print(f"Features guardadas en: data/phase2_features.csv")
        
        return features_df
    
    def train(self):
        """Entrena el modelo con validación cruzada"""
        df = self.load_and_prepare_data()
        features_df = self.extract_features_from_texts(df)
        
        if len(features_df) < 100:
            print("\nPocos datos válidos. Usando datos sintéticos...")
            return self.train_synthetic()
        
        X = features_df[self.feature_cols]
        y = features_df['is_ia']
        
        print("\n" + "="*60)
        print(" ENTRENANDO MODELO")
        print("="*60)
        
        # Escalar
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Cross-validation
        print("\nVALIDACIÓN CRUZADA (5-Folds):")
        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        print(f"   Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"   Media: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # ¿Overfitting?
        train_acc = self.model.score(self.X_train_scaled, self.y_train)
        test_acc = self.model.score(self.X_test_scaled, self.y_test)
        diff = train_acc - test_acc
        
        print(f"\nANÁLISIS DE OVERFITTING:")
        print(f"   Accuracy en TRAIN: {train_acc:.4f}")
        print(f"   Accuracy en TEST:  {test_acc:.4f}")
        print(f"   Diferencia:        {diff:.4f}")
        
        if diff > 0.15:
            print("    ALERTA: Posible overfitting detectado!")
        elif diff < 0.02:
            print("   Buena generalización")
        else:
            print("   Overfitting moderado")
        
        # Evaluación en test
        y_pred = self.model.predict(self.X_test_scaled)
        y_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        print(f"\nRESULTADOS EN TEST:")
        print(f"   Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"   F1-Score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"   AUC-ROC:  {roc_auc_score(self.y_test, y_proba):.4f}")
        
        print(f"\nMATRIZ DE CONFUSIÓN:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"   True Neg:  {cm[0,0]:5d} | False Pos: {cm[0,1]:5d}")
        print(f"   False Neg: {cm[1,0]:5d} | True Pos:  {cm[1,1]:5d}")
        
        print(f"\nCLASSIFICATION REPORT:")
        print(classification_report(self.y_test, y_pred, target_names=['Humano', 'IA']))
        
        print(f"\nTOP 5 FEATURES:")
        importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {feat:25s}: {imp:.4f}")
        
        return self.model
    
    def train_synthetic(self):
        """Entrena con datos sintéticos como fallback"""
        print("\nGenerando datos sintéticos...")
        self.is_synthetic = True
        
        np.random.seed(42)
        n_samples = 2000
        
        data = []
        self.feature_cols = [
            'lexical_density', 'ttr', 'sentence_mean_len', 'sentence_std_len',
            'bigram_entropy', 'punct_ratio', 'long_words_ratio', 'total_words',
            'paragraphs_per_sent', 'words_longer6_ratio'
        ]
        
        for _ in range(n_samples):
            is_ia = np.random.choice([0, 1])
            
            if is_ia == 0:
                features = {
                    'lexical_density': np.random.uniform(0.5, 0.8),
                    'ttr': np.random.uniform(0.4, 0.7),
                    'sentence_mean_len': np.random.uniform(12, 25),
                    'sentence_std_len': np.random.uniform(5, 15),
                    'bigram_entropy': np.random.uniform(5, 10),
                    'punct_ratio': np.random.uniform(0.05, 0.15),
                    'long_words_ratio': np.random.uniform(0.15, 0.35),
                    'total_words': np.random.randint(100, 500),
                    'paragraphs_per_sent': np.random.uniform(0.1, 0.4),
                    'words_longer6_ratio': np.random.uniform(0.15, 0.35)
                }
            else:
                features = {
                    'lexical_density': np.random.uniform(0.3, 0.55),
                    'ttr': np.random.uniform(0.2, 0.45),
                    'sentence_mean_len': np.random.uniform(15, 22),
                    'sentence_std_len': np.random.uniform(2, 6),
                    'bigram_entropy': np.random.uniform(3, 6),
                    'punct_ratio': np.random.uniform(0.02, 0.08),
                    'long_words_ratio': np.random.uniform(0.05, 0.2),
                    'total_words': np.random.randint(150, 400),
                    'paragraphs_per_sent': np.random.uniform(0.05, 0.2),
                    'words_longer6_ratio': np.random.uniform(0.05, 0.2)
                }
            features['is_ia'] = is_ia
            data.append(features)
        
        df = pd.DataFrame(data)
        X = df[self.feature_cols]
        y = df['is_ia']
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print(f"Modelo entrenado con DATOS SINTÉTICOS")
        print(f"NOTA: Los resultados NO serán precisos en datos reales!")
        
        return self.model
    
    def save_model(self):
        """Guarda el modelo y datos auxiliares"""
        if self.model is None:
            self.train()
        
        os.makedirs('backend', exist_ok=True)
        
        # Calcular estadísticas de entrenamiento
        X_train_mean = self.X_train_scaled.mean(axis=0) if self.X_train_scaled is not None else None
        X_train_std = self.X_train_scaled.std(axis=0) if self.X_train_scaled is not None else None
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'feature_importance': self.model.feature_importances_,
            'X_train_scaled': self.X_train_scaled,
            'X_train_mean': X_train_mean,
            'X_train_std': X_train_std,
            'is_synthetic': self.is_synthetic
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModelo guardado en: {self.model_path}")
        print(f"   Datos sintéticos: {'SÍ' if self.is_synthetic else 'NO'}")
        
        return self.model_path

def main():
    print("="*60)
    print(" ENTRENAMIENTO DE MODELO v2 MEJORADO")
    print("="*60)
    print("\nBuscando archivo en: data/train_v2_drcat_02.csv\n")
    
    trainer = ModelTrainer(data_path="data/train_v2_drcat_02.csv")
    
    try:
        trainer.train()
        trainer.save_model()
        print("\n" + "="*60)
        print(" ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\nAhora ejecuta: python backend/app_v2_explainable.py")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nSoluciones posibles:")
        print("1. Verifica que el archivo esté en: data/train_v2_drcat_02.csv")
        print("2. Revisa que las columnas se llamen 'text' y 'label'")
        print("3. Si las columnas tienen otros nombres, modifica el código")

if __name__ == "__main__":
    main()
