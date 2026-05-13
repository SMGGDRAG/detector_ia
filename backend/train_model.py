# backend/train_model.py
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, data_path="data/train_v2_drcat_02.csv"):  # ← NOMBRE DE TU ARCHIVO
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_path = "backend/ia_model.pkl"
        
    def load_and_prepare_data(self):
        """
        Carga el CSV de DAIGT-v2 y prepara las features estilométricas
        """
        print("="*60)
        print(" CARGANDO DATASET DAIGT-v2")
        print("="*60)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"""
             No se encontró el archivo: {self.data_path}
            
            Por favor, asegúrate de que:
            1. El archivo train_v2_drcat_02.csv esté en la carpeta 'data/'
            2. La ruta sea: {os.path.abspath(self.data_path)}
            """)
        
        # Cargar CSV
        df = pd.read_csv(self.data_path)
        print(f" Archivo cargado: {len(df)} filas")
        print(f" Columnas disponibles: {list(df.columns)}")
        
        # Identificar columnas de texto y etiqueta
        text_col = None
        label_col = None
        
        # Buscar columna de texto
        for col in ['text', 'content', 'essay', 'full_text', 'response']:
            if col in df.columns:
                text_col = col
                break
        
        # Buscar columna de etiqueta
        for col in ['label', 'is_ia', 'generated', 'source', 'class']:
            if col in df.columns:
                label_col = col
                break
        
        if text_col is None:
            print(f" Columnas disponibles: {list(df.columns)}")
            raise ValueError("No se encontró una columna de texto. Usa una de estas: 'text', 'content', 'essay'")
        
        if label_col is None:
            print(f" Columnas disponibles: {list(df.columns)}")
            raise ValueError("No se encontró una columna de etiqueta. Usa una de estas: 'label', 'is_ia', 'generated'")
        
        print(f" Usando texto de: '{text_col}'")
        print(f" Usando etiqueta de: '{label_col}'")
        
        # Crear DataFrame con columnas estandarizadas
        df_clean = pd.DataFrame()
        df_clean['text'] = df[text_col].astype(str)
        
        # Convertir etiqueta a 0/1 (humano/IA)
        if label_col == 'label':
            # DAIGT-v2: label 0 = humano, 1 = IA
            df_clean['is_ia'] = df[label_col].astype(int)
        elif label_col == 'generated':
            # Algunos datasets: generated 0 = humano, 1 = IA
            df_clean['is_ia'] = df[label_col].astype(int)
        else:
            # Intentar inferir
            unique_vals = df[label_col].unique()
            print(f" Valores únicos en etiqueta: {unique_vals}")
            
            if len(unique_vals) == 2:
                # Mapear los valores a 0 y 1
                val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                df_clean['is_ia'] = df[label_col].map(val_map)
            else:
                raise ValueError(f"No se pueden mapear etiquetas: {unique_vals}")
        
        # Filtrar textos demasiado cortos
        df_clean['word_count'] = df_clean['text'].apply(lambda x: len(str(x).split()))
        print(f" Distribución de longitud de textos:")
        print(f"   Media: {df_clean['word_count'].mean():.0f} palabras")
        print(f"   Min: {df_clean['word_count'].min()} palabras")
        print(f"   Max: {df_clean['word_count'].max()} palabras")
        
        # Eliminar textos muy cortos (menos de 50 palabras)
        min_words = 50
        df_filtered = df_clean[df_clean['word_count'] >= min_words].copy()
        print(f"\n Filtrados textos con menos de {min_words} palabras:")
        print(f"   Antes: {len(df_clean)} textos")
        print(f"   Después: {len(df_filtered)} textos")
        print(f"   Eliminados: {len(df_clean) - len(df_filtered)} textos")
        
        # Ver balance de clases
        print(f"\n Balance de clases después del filtrado:")
        n_human = sum(df_filtered['is_ia'] == 0)
        n_ia = sum(df_filtered['is_ia'] == 1)
        print(f"   Humano (0): {n_human} ({n_human/len(df_filtered)*100:.1f}%)")
        print(f"   IA (1): {n_ia} ({n_ia/len(df_filtered)*100:.1f}%)")
        
        # Balancear si es necesario
        if abs(n_human - n_ia) / len(df_filtered) > 0.2:
            print("\n Balanceando clases...")
            min_samples = min(n_human, n_ia)
            df_human = df_filtered[df_filtered['is_ia'] == 0].sample(n=min_samples, random_state=42)
            df_ia = df_filtered[df_filtered['is_ia'] == 1].sample(n=min_samples, random_state=42)
            df_filtered = pd.concat([df_human, df_ia]).sample(frac=1, random_state=42)
            print(f"   Balanceado: {len(df_filtered)} muestras (50% cada clase)")
        
        return df_filtered
    
    def extract_features_from_texts(self, df):
        """
        Extrae features estilométricas de los textos
        """
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
        
        # Descargar recursos NLTK si no están
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        def extract_single_text_features(text):
            """Extrae features para un solo texto"""
            try:
                # Tokenización
                sents = sent_tokenize(text, language='english')
                words = [w for w in word_tokenize(text, language='english') if any(c.isalpha() for c in w)]
                
                if len(words) < 10 or len(sents) == 0:
                    return None
                
                # Stopwords
                stops = set(stopwords.words('english'))
                non_stop = [w for w in words if w.lower() not in stops]
                unique = set(w.lower() for w in words)
                #======================================================================
                #En un modelo de clasificación (por ejemplo, Random Forest, XGBoost, etc.) 
                # que distingue entre texto escrito por humano vs. IA, el feature_importance 
                # indica cuánto contribuye cada característica a la decisión final.
                #=====================================================================

                # Feature 1: Densidad léxica
                lexical_density = len(non_stop) / len(words) if len(words) > 0 else 0
                
                # Feature 2: Type-Token Ratio
                ttr = len(unique) / len(words) if len(words) > 0 else 0
                
                # Feature 3 y 4: Longitud de oraciones
                sent_lengths = [len(word_tokenize(s, language='english')) for s in sents]
                sentence_mean_len = np.mean(sent_lengths) if sent_lengths else 0
                sentence_std_len = np.std(sent_lengths) if sent_lengths else 0
                
                # Feature 5: Entropía de bigramas
                words_lower = [w.lower() for w in words if w.isalpha()]
                bigrams = list(zip(words_lower[:-1], words_lower[1:]))
                if bigrams:
                    counts = Counter(bigrams)
                    total = sum(counts.values())
                    probs = [c/total for c in counts.values()]
                    bigram_entropy = -sum(p * math.log2(p) for p in probs)
                else:
                    bigram_entropy = 0
                
                # Feature 6: Ratio de puntuación
                punct_chars = sum(1 for ch in text if ch in string.punctuation)
                punct_ratio = punct_chars / len(words) if len(words) > 0 else 0
                
                # Feature 7: Palabras largas (>6)
                long_words = [w for w in words if len(w) > 6]
                long_words_ratio = len(long_words) / len(words) if len(words) > 0 else 0
                
                # Feature 8: Total de palabras
                total_words = len(words)
                
                # Feature 9: Párrafos por oración
                paragraphs = text.count('\n\n') + 1 if text.strip() else 1
                paragraphs_per_sent = paragraphs / len(sents) if len(sents) > 0 else 0
                
                # Feature 10: Words longer 6 ratio
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
            except:
                return None
        
        # Extraer features de todos los textos
        features_list = []
        self.feature_cols = ['lexical_density', 'ttr', 'sentence_mean_len', 'sentence_std_len',
                              'bigram_entropy', 'punct_ratio', 'long_words_ratio', 'total_words',
                              'paragraphs_per_sent', 'words_longer6_ratio']
        
        for idx, row in df.iterrows():
            feats = extract_single_text_features(row['text'])
            if feats:
                feats['is_ia'] = row['is_ia']
                features_list.append(feats)
            
            # Mostrar progreso
            if (idx + 1) % 500 == 0:
                print(f"   Procesados {idx + 1} textos...")
        
        features_df = pd.DataFrame(features_list)
        print(f"\n Features extraídas: {len(features_df)} textos válidos")
        print(f" Eliminados: {len(df) - len(features_df)} textos (muy cortos o error)")
        
        # Guardar features para uso futuro
        features_df.to_csv('data/phase2_features.csv', index=False)
        print(f" Features guardadas en: data/phase2_features.csv")
        
        return features_df
    
    def train(self):
        """Entrena el modelo con los datos reales"""
        # Cargar y preparar datos
        df = self.load_and_prepare_data()
        
        # Extraer features
        features_df = self.extract_features_from_texts(df)
        
        if len(features_df) < 100:
            print(" Pocos datos válidos. Usando datos sintéticos...")
            return self.train_synthetic()
        
        # Preparar X y y
        X = features_df[self.feature_cols]
        y = features_df['is_ia']
        
        print("\n" + "="*60)
        print(" ENTRENANDO MODELO")
        print("="*60)
        
        # Escalar features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Dividir para evaluación
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        print(f"\n Resultados en conjunto de prueba:")
        print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"   F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"   AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
        
        print(f"\n Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Humano', 'IA']))
        
        # Feature importance
        print(f"\n Top 5 features más importantes:")
        importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {feat}: {imp:.4f}")
        
        return self.model
    
    def train_synthetic(self):
        """Fallback: entrenar con datos sintéticos"""
        print("\n Generando datos sintéticos...")
        
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
        
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.model.fit(X_scaled, y)
        
        print(f" Modelo entrenado con datos sintéticos")
        return self.model
    
    def save_model(self):
        """Guarda el modelo entrenado"""
        if self.model is None:
            self.train()
        
        os.makedirs('backend', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'feature_importance': self.model.feature_importances_
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n Modelo guardado en: {self.model_path}")
        return self.model_path

def main():
    print("="*60)
    print(" ENTRENAMIENTO DE MODELO DETECTOR DE IA")
    print("="*60)
    print("\n Buscando archivo en: data/train_v2_drcat_02.csv")
    
    trainer = ModelTrainer(data_path="data/train_v2_drcat_02.csv")
    
    try:
        trainer.train()
        trainer.save_model()
        print("\n" + "="*60)
        print(" ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\n Ahora ejecuta: python backend/app.py")
    except Exception as e:
        print(f"\n Error: {e}")
        print("\n Soluciones posibles:")
        print("1. Verifica que el archivo esté en: data/train_v2_drcat_02.csv")
        print("2. Revisa que las columnas del CSV se llamen 'text' y 'label'")
        print("3. Si las columnas tienen otros nombres, modifica el código")

if __name__ == "__main__":
    main()