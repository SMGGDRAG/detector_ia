# backend/save_model.py
# Script separado para guardar el modelo después de entrenar

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def save_trained_model(features_csv="../data/phase2_features.csv", model_path="ia_model.pkl"):
    """
    Guarda el modelo entrenado y el scaler para usarlos después
    """
    print(" Guardando modelo entrenado...")
    
    # Verificar que existe el archivo
    import os
    if not os.path.exists(features_csv):
        print(f" Error: No se encontró {features_csv}")
        print("   Ejecuta primero las Fases 1 y 2 para generar este archivo")
        return None
    
    # Cargar datos
    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c != 'is_ia']
    X = df[feature_cols]
    y = df['is_ia']
    
    print(f" Datos: {len(X)} muestras, {len(feature_cols)} features")
    print(f" Balance: Humano={sum(y==0)}, IA={sum(y==1)}")
    
    # Entrenar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # Guardar todo en un archivo
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_importance': model.feature_importances_,
        'accuracy': model.score(X_scaled, y)
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f" Modelo guardado en {model_path}")
    print(f" Precisión en entrenamiento: {model.score(X_scaled, y):.4f}")
    
    # Mostrar feature importance
    print("\n Importancia de features:")
    importance = dict(zip(feature_cols, model.feature_importances_))
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {feat}: {imp:.4f}")
    
    return model_data

if __name__ == "__main__":
    save_trained_model()