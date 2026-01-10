import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import unicodedata
import math
# =========================
# Inicializar Flask
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Cargar modelos (.h5)
# =========================
modelo_riesgo = load_model("modelo_lstm_riesgo.h5", compile=False)
modelo_geo = load_model("modelo_geo.h5", compile=False)

# =========================
# Provincias y códigos (OFICIALES)
# =========================
PROVINCIAS = {
    "PICHINCHA": 17,
    "GUAYAS": 9,
    "TUNGURAHUA": 18,
    "COTOPAXI": 5,
    "SUCUMBIOS": 21,
    "ESMERALDAS": 8,
    "EL ORO": 7,
    "CARCHI": 4,
    "STO DGO DE LOS TSACHILAS": 23,
    "MANABI": 13,
    "IMBABURA": 10,
    "LOS RIOS": 12,
    "AZUAY": 1,
    "ZAMORA CHINCHIPE": 19,
    "NAPO": 15,
    "CHIMBORAZO": 6,
    "ORELLANA": 22,
    "CAÑAR": 3,
    "MORONA SANTIAGO": 14,
    "LOJA": 11,
    "SANTA ELENA": 24,
    "PASTAZA": 16,
    "GALAPAGOS": 20,
    "BOLIVAR": 2
}

LAT_MIN = -5.0
LAT_MAX = 1.5
LON_MIN = -81.0
LON_MAX = -75.0


# =========================
# Normalizar texto (quitar tildes)
# =========================
def normalizar(texto):
    texto = texto.upper().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    return texto


def denormalizar(valor_norm, min_val, max_val):
    return valor_norm * (max_val - min_val) + min_val
def generar_batch_input(fecha_target):
    # Asumimos códigos de provincia del 1 al 24
    codigos_provincias = list(range(1, 25)) 
    batch_input_list = []
    
    for cod_prov in codigos_provincias:
        secuencia_provincia = []
        for i in range(7, 0, -1):
            fecha_pasada = fecha_target - timedelta(days=i)
            # Normalización manual (Ajusta según tu entrenamiento real)
            row = [
                fecha_pasada.month / 12.0,
                fecha_pasada.day / 31.0,
                (fecha_pasada.year - 2000) / 30.0,
                fecha_pasada.weekday() / 6.0,
                cod_prov / 24.0 
            ]
            secuencia_provincia.append(row)
        batch_input_list.append(secuencia_provincia)
    
    return np.array(batch_input_list), codigos_provincias

# =========================
# Crear secuencia LSTM (7 días)
# =========================
def crear_secuencia(fecha_target, cod_prov):
    secuencia = []

    for i in range(7, 0, -1):
        f = fecha_target - timedelta(days=i)

        fila = [
            f.month / 12.0,
            f.day / 31.0,
            (f.year - 2000) / 30.0,
            f.weekday() / 6.0,
            cod_prov / 24.0
        ]
        secuencia.append(fila)

    return np.array(secuencia).reshape(1, 7, 5)

# =========================================================
# ENDPOINT 1 — CONTEXTO (Modelo 1)
# =========================================================
@app.route("/api/prediccion/contexto", methods=["POST"])
def prediccion_contexto():
    data = request.get_json()

    try:
        fecha = datetime.strptime(data["fecha"], "%Y-%m-%d")
        provincia = normalizar(data["provincia"])

        if provincia not in PROVINCIAS:
            return jsonify({
                "error": "Provincia no válida",
                "provincias_validas": list(PROVINCIAS.keys())
            }), 400

        cod_prov = PROVINCIAS[provincia]

        input_seq = crear_secuencia(fecha, cod_prov)

        riesgo_pred, casos_pred = modelo_riesgo.predict(input_seq)

        nivel_riesgo = int(np.argmax(riesgo_pred[0]))
        casos = float(casos_pred[0][0])

        return jsonify({
            "fecha": data["fecha"],
            "provincia": provincia,
            "riesgo": nivel_riesgo,
            "riesgo_label": ["BAJO", "MEDIO", "ALTO"][nivel_riesgo],
            "desapariciones_estimadas": round(casos, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prediccion/localizacion", methods=["POST"])
def prediccion_localizacion():
    data = request.get_json()

    try:
        fecha = datetime.strptime(data["fecha"], "%Y-%m-%d")
        provincia = normalizar(data["provincia"])
        riesgo = int(data["riesgo"])
        casos = float(data["desapariciones_estimadas"])

        if provincia not in PROVINCIAS:
            return jsonify({"error": "Provincia no válida"}), 400

        cod_prov = PROVINCIAS[provincia]

        base_seq = crear_secuencia(fecha, cod_prov)[0]

        secuencia_geo = []
        for fila in base_seq:
            fila_geo = np.concatenate([
                fila,
                [1],        # bias
                [riesgo],
                [casos]
            ])
            secuencia_geo.append(fila_geo)

        input_geo = np.array(secuencia_geo).reshape(1, 7, 8)

        latlon = modelo_geo.predict(input_geo)[0]
        

        lat_real = denormalizar(latlon[0], LAT_MIN, LAT_MAX)
        lon_real = denormalizar(latlon[1], LON_MIN, LON_MAX)

        return jsonify({
            "puntos": [
                {
                    "lat": float(latlon[0]),
                    "lng": float(latlon[1]),
                    "lat_real": float(lat_real),
                    "lng_real": float(lon_real),
                    "peso": casos
                }
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_point_risk', methods=['POST'])
def predict_point_risk():
    try:
        data = request.get_json()
        fecha_str = data.get('fecha')
        lat_usuario = float(data.get('lat'))
        lng_usuario = float(data.get('lng'))    
        
        fecha_target = datetime.strptime(fecha_str, '%Y-%m-%d')

        # ESTRATEGIA:
        # 1. Predecimos dónde estará el peligro en todo el país (igual que el heatmap).
        # 2. Calculamos la distancia de TU punto a los puntos de peligro predichos.
        # 3. Si estás cerca de una zona de peligro, te asignamos ese riesgo.

        # PASO A: Predecir todo el escenario nacional (Reutilizamos lógica)
        input_batch, _ = generar_batch_input(fecha_target)
        riesgos_pred, n_des_pred = modelo_riesgo.predict(input_batch)
        
        # Preparar para geo (Simplificado)
        batch_geo_list = []
        lista_niveles_riesgo = []
        
        for i in range(len(input_batch)):
            nivel = np.argmax(riesgos_pred[i])
            lista_niveles_riesgo.append(nivel)
            n_c = n_des_pred[i][0]
            
            seq_geo = []
            for paso in range(7):
                row = np.concatenate([input_batch[i][paso], [1], [nivel], [n_c]])
                seq_geo.append(row)
            batch_geo_list.append(seq_geo)
            
        latlon_preds = modelo_geo.predict(np.array(batch_geo_list))

        # PASO B: Buscar el punto de riesgo más cercano al usuario
        distancia_minima = float('inf')
        riesgo_encontrado = 0
        provincia_cercana = -1
        
        # Umbral de cercanía (en grados). 0.1 grados son aprox 11km. 
        # Si está a menos de 10-15km de una zona predicha, asume ese riesgo.
        UMBRAL_DISTANCIA = 0.15 

        for i in range(len(latlon_preds)):
            pred_lat = latlon_preds[i][0]
            pred_lng = latlon_preds[i][1]
            
            # Distancia Euclidiana simple (Pitágoras)
            dist = math.sqrt((pred_lat - lat_usuario)**2 + (pred_lng - lng_usuario)**2)
            
            if dist < distancia_minima:
                distancia_minima = dist
                # Si está "cerca" de la predicción, tomamos ese riesgo
                if dist < UMBRAL_DISTANCIA:
                    riesgo_encontrado = int(lista_niveles_riesgo[i])
                    provincia_cercana = i + 1 # Código de provincia (i+1)
                else:
                    # Si el punto más cercano está muy lejos, el riesgo es bajo/desconocido
                    riesgo_encontrado = 0 

        # Respuesta
        mensaje = "Zona Segura"
        if riesgo_encontrado == 1: mensaje = "Zona de Riesgo Medio"
        if riesgo_encontrado == 2: mensaje = "Zona de Alto Riesgo"

        return jsonify({
            "ubicacion_usuario": {"lat": lat_usuario, "lng": lng_usuario},
            "distancia_al_foco_mas_cercano": float(distancia_minima),
            "nivel_riesgo_estimado": riesgo_encontrado,
            "mensaje": mensaje,
            "provincia_referencia": provincia_cercana
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
# =========================
# Ejecutar servidor
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
