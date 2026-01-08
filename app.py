from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde el frontend

# =============================================================================
# CARGAR MODELO Y SCALER
# =============================================================================

print("Cargando modelo...")
modelo = load_model('modelo_lstm_desapariciones.h5')

print("Cargando scaler...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Cargando modelo simple...")
modelo_simple = load_model('modelo_lstm_simple.h5')

print("Cargando scaler simple...")
with open('scaler_simple.pkl', 'rb') as f:
    scaler_simple = pickle.load(f)


print("✅ Sistema listo")

# =============================================================================
# FUNCIÓN DE PREDICCIÓN
# =============================================================================

def predecir_riesgo_punto(lat, long, fecha, rango_edad_cod, sexo_numerico=0.5):
    """
    Predice el nivel de riesgo para un punto geográfico.
    
    Args:
        lat (float): Latitud
        long (float): Longitud
        fecha (str): Fecha en formato 'YYYY-MM-DD'
        rango_edad_cod (int): 0=ADOLESCENTES, 1=ADULTO, 2=ADULTO MAYOR, 3=NIÑOS
        sexo_numerico (float): 0=MUJER, 1=HOMBRE, 0.5=DESCONOCIDO
    
    Returns:
        dict: Predicción con nivel de riesgo y probabilidades
    """
    try:
        # Parsear fecha
        fecha_dt = pd.to_datetime(fecha)
        
        # Crear vector de características
        entrada = np.array([[
            float(lat),
            float(long),
            int(fecha_dt.month),
            int(fecha_dt.dayofweek),
            int(fecha_dt.day),
            int(fecha_dt.year),
            int(rango_edad_cod),
            float(sexo_numerico)
        ]])
        
        # Normalizar
        entrada_scaled = scaler.transform(entrada)
        
        # Crear secuencia temporal (simular ventana de 12 pasos)
        entrada_seq = np.tile(entrada_scaled, (1, 12, 1))
        
        # Predecir
        prediccion = modelo.predict(entrada_seq, verbose=0)
        
        # Extraer resultados
        nivel_riesgo = int(np.argmax(prediccion[0]))
        probabilidades = prediccion[0].tolist()
        
        return {
            'nivel_riesgo': nivel_riesgo,
            'nivel_texto': ['Bajo', 'Medio', 'Alto'][nivel_riesgo],
            'probabilidades': {
                'bajo': round(probabilidades[0], 4),
                'medio': round(probabilidades[1], 4),
                'alto': round(probabilidades[2], 4)
            },
            'confianza': round(max(probabilidades), 4)
        }
    
    except Exception as e:
        raise Exception(f"Error en predicción: {str(e)}")

def predecir_riesgo_simple(lat, long, fecha):
    """
    Predice el nivel de riesgo solo con fecha y coordenadas (modelo simple).
    """
    try:
        fecha_dt = pd.to_datetime(fecha)
        
        entrada = np.array([[
            float(lat),
            float(long),
            int(fecha_dt.month),
            int(fecha_dt.dayofweek),
            int(fecha_dt.day),
            int(fecha_dt.year)
        ]])
        
        entrada_scaled = scaler_simple.transform(entrada)
        entrada_seq = np.tile(entrada_scaled, (1, 12, 1))
        prediccion = modelo_simple.predict(entrada_seq, verbose=0)
        
        nivel_riesgo = int(np.argmax(prediccion[0]))
        probabilidades = prediccion[0].tolist()
        
        return {
            'nivel_riesgo': nivel_riesgo,
            'nivel_texto': ['Bajo', 'Medio', 'Alto'][nivel_riesgo],
            'probabilidades': {
                'bajo': round(probabilidades[0], 4),
                'medio': round(probabilidades[1], 4),
                'alto': round(probabilidades[2], 4)
            },
            'confianza': round(max(probabilidades), 4)
        }
    
    except Exception as e:
        raise Exception(f"Error en predicción: {str(e)}")
# =============================================================================
# ENDPOINTS DE LA API
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'mensaje': 'API de Predicción de Riesgo de Desapariciones - Ecuador',
        'version': '1.0',
        'endpoints': {
            '/predecir': 'POST - Predecir un solo punto con filtro como el rango de edad y el sexo',
            '/predecir_lote': 'POST - Predecir múltiples puntos (para mapa de calor)'
        }
    })

@app.route('/predecir-filter', methods=['POST'])
def predecir_punto_endpoint():
    """
    Endpoint para predecir un solo punto.
    
    Body (JSON):
    {
        "lat": -2.125138401,
        "long": -79.9209116,
        "fecha": "2026-01-15",
        "rango_edad_cod": 0,
        "sexo_numerico": 1  (opcional, default=0.5)
    }
    """
    try:
        data = request.get_json()
        
        # Validar campos requeridos
        campos_requeridos = ['lat', 'long', 'fecha',"rango_edad_cod","sexo_numerico"]
        for campo in campos_requeridos:
            if campo not in data:
                return jsonify({'error': f'Campo requerido faltante: {campo}'}), 400
        
        # Extraer parámetros
        lat = data['lat']
        long = data['long']
        fecha = data['fecha']
        rango_edad_cod = data.get('rango_edad_cod',1)
        sexo_numerico = data.get('sexo_numerico', 1)
        
        # Validar rangos
        if not (-5 <= lat <= 2):  # Rango aproximado de Ecuador
            return jsonify({'error': 'Latitud fuera del rango de Ecuador'}), 400
        
        if not (-82 <= long <= -75):  # Rango aproximado de Ecuador
            return jsonify({'error': 'Longitud fuera del rango de Ecuador'}), 400
        
        if rango_edad_cod not in [0, 1, 2, 3]:
            return jsonify({'error': 'rango_edad_cod debe ser 0, 1, 2 o 3'}), 400
        
        # Realizar predicción
        resultado = predecir_riesgo_punto(lat, long, fecha, rango_edad_cod, sexo_numerico)
        
        return jsonify({
            'exito': True,
            'punto': {
                'lat': lat,
                'long': long,
                'fecha': fecha
            },
            'prediccion': resultado
        })
    
    except Exception as e:
        return jsonify({
            'exito': False,
            'error': str(e)
        }), 500


@app.route('/predecir-riesgo', methods=['POST'])
def predecir_riesgo_endpoint():
    """
    Endpoint para predecir solo con fecha y coordenadas.
    
    Body (JSON):
    {
        "lat": -2.125138401,
        "long": -79.9209116,
        "fecha": "2026-01-15"
    }
    """
    try:
        data = request.get_json()
        
        # Validar campos requeridos
        campos_requeridos = ['lat', 'long', 'fecha']
        for campo in campos_requeridos:
            if campo not in data:
                return jsonify({'error': f'Campo requerido faltante: {campo}'}), 400
        
        lat = data['lat']
        long = data['long']
        fecha = data['fecha']
        
        # Validar rangos
        if not (-5 <= lat <= 2):
            return jsonify({'error': 'Latitud fuera del rango de Ecuador'}), 400
        
        if not (-82 <= long <= -75):
            return jsonify({'error': 'Longitud fuera del rango de Ecuador'}), 400
        
        # Realizar predicción
        resultado = predecir_riesgo_simple(lat, long, fecha)
        
        return jsonify({
            'exito': True,
            'punto': {
                'lat': lat,
                'long': long,
                'fecha': fecha
            },
            'prediccion': resultado
        })
    
    except Exception as e:
        return jsonify({
            'exito': False,
            'error': str(e)
        }), 500


@app.route('/predecir_lote', methods=['POST'])
def predecir_lote():
    """
    Endpoint para predecir múltiples puntos (usado para generar mapa de calor).
    
    Body (JSON):
    {
        "puntos": [
            {
                "lat": -2.125138401,
                "long": -79.9209116,
                "fecha": "2026-01-15",
                "rango_edad_cod": 0,
                "sexo_numerico": 1
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'puntos' not in data:
            return jsonify({'error': 'Campo "puntos" requerido'}), 400
        
        puntos = data['puntos']
        
        if not isinstance(puntos, list):
            return jsonify({'error': '"puntos" debe ser una lista'}), 400
        
        resultados = []
        
        for i, punto in enumerate(puntos):
            try:
                # Extraer parámetros
                lat = punto['lat']
                long = punto['long']
                fecha = punto['fecha']
                rango_edad_cod = punto['rango_edad_cod']
                sexo_numerico = punto.get('sexo_numerico', 0.5)
                
                # Predecir
                prediccion = predecir_riesgo_punto(lat, long, fecha, rango_edad_cod, sexo_numerico)
                
                resultados.append({
                    'index': i,
                    'lat': lat,
                    'long': long,
                    'prediccion': prediccion
                })
            
            except Exception as e:
                resultados.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'exito': True,
            'total_puntos': len(puntos),
            'resultados': resultados
        })
    
    except Exception as e:
        return jsonify({
            'exito': False,
            'error': str(e)
        }), 500

@app.route('/info', methods=['GET'])
def info():
    """
    Información sobre los códigos de entrada.
    """
    return jsonify({
        'rango_edad_cod': {
            0: 'ADOLESCENTES',
            1: 'ADULTO',
            2: 'ADULTO MAYOR',
            3: 'NIÑOS(AS)'
        },
        'sexo_numerico': {
            0: 'MUJER',
            1: 'HOMBRE',
         
        },
        'nivel_riesgo': {
            0: 'Bajo (< 1 caso)',
            1: 'Medio (1-3 casos)',
            2: 'Alto (> 3 casos)'
        },
        'formato_fecha': 'YYYY-MM-DD',
        'rango_ecuador': {
            'latitud': '(-5, 2)',
            'longitud': '(-82, -75)'
        }
    })

# =============================================================================
# EJECUTAR SERVIDOR
# =============================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)