"""
API Flask - Predicción de Desapariciones
Endpoint único: POST /predecir
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle

app = Flask(__name__)
CORS(app)

print("🔄 Cargando modelo...")
model = keras.models.load_model('modelo_lstm_desapariciones.h5')
print("✅ Modelo cargado")

print("🔄 Cargando scaler...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("✅ Scaler cargado")

print("🔄 Cargando configuración...")
with open('model_config.pkl', 'rb') as f:
    config = pickle.load(f)
print("✅ Configuración cargada")

SEQUENCE_LENGTH = config['sequence_length']
FEATURE_COLUMNS = config['feature_columns']

print(f"\n{'='*70}")
print(f"✅ API LISTA PARA RECIBIR PETICIONES")
print(f"{'='*70}\n")

# ============================================================================
# ENDPOINT: /predecir
# ============================================================================

@app.route('/predecir', methods=['POST'])
def predecir():
    """
    Predice el riesgo de desapariciones para una fecha y ubicación futura
    
    INPUT:
    {
        "fecha": "2026-06-15",
        "provincia": "GUAYAS",
        "latitud": -2.19,
        "longitud": -79.90
    }
    
    OUTPUT:
    {
        "success": true,
        "fecha": "2026-06-15",
        "provincia": "GUAYAS",
        "ubicacion": {"latitud": -2.19, "longitud": -79.90},
        "nivel_riesgo": 2,
        "riesgo": "Alto",
        "confianza": 0.85,
        "probabilidades": {"bajo": 0.05, "medio": 0.10, "alto": 0.85}
    }
    """
    try:
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No se recibieron datos"
            }), 400
        
        # Validar campos requeridos
        required = ['fecha', 'provincia', 'latitud', 'longitud']
        missing = [f for f in required if f not in data]
        
        if missing:
            return jsonify({
                "success": False,
                "error": f"Campos faltantes: {', '.join(missing)}",
                "ejemplo": {
                    "fecha": "2026-06-15",
                    "provincia": "GUAYAS",
                    "latitud": -2.19,
                    "longitud": -79.90
                }
            }), 400
        
        # Extraer datos
        fecha_input = data['fecha']
        provincia = data['provincia'].upper().strip()
        latitud = float(data['latitud'])
        longitud = float(data['longitud'])
        
        # Parsear fecha
        try:
            fecha_dt = pd.to_datetime(fecha_input, format='%Y-%m-%d')
        except:
            try:
                fecha_dt = pd.to_datetime(fecha_input, format='%d/%m/%Y')
            except:
                return jsonify({
                    "success": False,
                    "error": "Formato de fecha inválido. Use YYYY-MM-DD o DD/MM/YYYY"
                }), 400
        
        mes = fecha_dt.month
        dia = fecha_dt.day
        dia_semana = fecha_dt.dayofweek + 1
        
        # Crear features (valores promedio/dummy para predicción)
        features = {
            'n_desapariciones': 0,  # Dummy, el modelo lo aprenderá del contexto
            'nivel_riesgo': 0,      # Dummy
            'latitud_desaparicion': latitud,
            'longitud_desaparicion': longitud,
            'mes': mes,
            'dia_semana': dia_semana,
            'dia': dia,
            'rango_edad_sexo_numerico': 0  # Dummy
        }
        
        # Crear array en el orden de FEATURE_COLUMNS
        feature_array = np.array([[features[col] for col in FEATURE_COLUMNS]])
        
        # Normalizar
        feature_scaled = scaler.transform(feature_array)
        
        # Crear secuencia (repetir el patrón SEQUENCE_LENGTH veces)
        sequence = np.array([np.repeat(feature_scaled, SEQUENCE_LENGTH, axis=0)])
        
        # Predecir
        prediction = model.predict(sequence, verbose=0)
        
        # Procesar resultado
        predicted_class = int(np.argmax(prediction[0]))
        prob_bajo = float(prediction[0][0])
        prob_medio = float(prediction[0][1])
        prob_alto = float(prediction[0][2])
        confianza = float(np.max(prediction[0]))
        
        # Etiqueta del riesgo
        riesgo_labels = {0: 'Bajo', 1: 'Medio', 2: 'Alto'}
        etiqueta = riesgo_labels[predicted_class]
        
        # Respuesta
        return jsonify({
            "success": True,
            "fecha": fecha_dt.strftime('%Y-%m-%d'),
            "provincia": provincia,
            "ubicacion": {
                "latitud": latitud,
                "longitud": longitud
            },
            "nivel_riesgo": predicted_class,
            "riesgo": etiqueta,
            "confianza": round(confianza, 4),
            "probabilidades": {
                "bajo": round(prob_bajo, 4),
                "medio": round(prob_medio, 4),
                "alto": round(prob_alto, 4)
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("    🚀 API DE PREDICCIÓN DE DESAPARICIONES")
    print("="*70)
    print("\n📍 Servidor: http://localhost:5000")
    print("\n📋 Endpoint:")
    print("   POST /predecir")
    print("\n💡 Ejemplo:")
    print("   curl -X POST http://localhost:5000/predecir \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{")
    print('       "fecha": "2026-06-15",')
    print('       "provincia": "GUAYAS",')
    print('       "latitud": -2.19,')
    print('       "longitud": -79.90')
    print("     }'")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)