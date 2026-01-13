import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from pydantic import BaseModel
import unicodedata
import math
import joblib
app = Flask(__name__)
CORS(app)


modelo_riesgo = load_model("modelo_lstm_riesgo.h5", compile=False)
modelo_geo = load_model("modelo_geo.h5", compile=False)

modelo_riesgo_punto = load_model("modelo_riesgo_punto_v2.h5", compile=False)
scaler = joblib.load("scaler_modelo3.pkl")
ETIQUETAS_RIESGO = ["BAJO", "MEDIO", "ALTO"]
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
def normalizar(texto):
    texto = texto.upper().strip()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    return texto

def normalizar_texto(texto):
    texto = texto.upper().strip()
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
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


def crear_secuencia(fecha_target, cod_prov):
    secuencia = []

    for i in range(7, 0, -1):
        f = fecha_target - timedelta(days=i)
        
        mes = math.sin(2 * np.pi * (f.month - 1) / 12)
        dia_semana = math.sin(2 * np.pi * (f.weekday()) / 7)
        fila = [
            f.day,
            mes,
            dia_semana,
            (f.year - 2017),
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
# =========================================================
# ENDPOINT 2 — LOCALIZACION (Modelo 2)
# =========================================================

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

        # valores neutros (no disponibles en la API)
        rango_edad_cod = 0
        sexo_numerico = 0

        secuencia_geo = []

        for i in range(7, 0, -1):
            f = fecha - timedelta(days=i)
            mes = math.sin(2 * np.pi * (f.month - 1) / 12)
            dia_semana = math.sin(2 * np.pi * (f.weekday()) / 7)

            fila = [
                f.day,
                mes,
                (f.year - 2017),
                dia_semana,
                (cod_prov/24),
                rango_edad_cod,
                sexo_numerico,
                riesgo,
                casos
            ]

            secuencia_geo.append(fila)

        input_geo = np.array(secuencia_geo, dtype=np.float32).reshape(1, 7, 9)

        latlon = modelo_geo.predict(input_geo, verbose=0)[0]

        return jsonify({
            "puntos": [
                {
                    "lat": float(latlon[0]),
                    "lng": float(latlon[1]),
                    "peso": round(casos, 2)
                }
            ]
        })

    except Exception as e:
        return jsonify({
            "error": "Error interno en predicción de localización",
            "detalle": str(e)
        }), 500

# =========================================================
# ENDPOINT 1 — PREDICCION DE RIESGO (Modelo 3)
# =========================================================


@app.route("/api/prediccion/punto", methods=["POST"])
@app.route("/api/prediccion/punto", methods=["POST"])
def predict_riesgo():
    try:
        data = request.get_json()

        # Validaciones existentes...
        fecha_dt = datetime.strptime(data["fecha"], "%Y-%m-%d")
        lat = float(data["lat"])
        lng = float(data["lng"])
        provincia = PROVINCIAS[data["provincia"]]
        
        mes = fecha_dt.month
        dia = fecha_dt.day
        dia_semana = fecha_dt.weekday()

        # ✅ CREAR SECUENCIA DE 7 PASOS
        secuencia = []
        for i in range(7):
            # Puedes usar la misma fecha o generar fechas pasadas
            secuencia.append([
                lat,
                lng,
                mes,
                dia,
                dia_semana,
                provincia
            ])
        
        # Convertir a numpy array con shape (7, 6)
        sample = np.array(secuencia, dtype=np.float32)
        
        # Escalar toda la secuencia
        sample_scaled = scaler.transform(sample)
        
        # ✅ Reshape a (1, 7, 6) para el modelo
        sample_scaled = sample_scaled.reshape(1, 7, 6)

        # Predicción
        riesgo_pred, n_pred = modelo_riesgo_punto.predict(sample_scaled, verbose=0)

        codigo_riesgo = int(np.argmax(riesgo_pred[0]))
        nivel_riesgo = ETIQUETAS_RIESGO[codigo_riesgo]
        n_desapariciones = max(0, float(n_pred[0][0]))

        return jsonify({
            "fecha": data["fecha"],
            "riesgo": {
                "codigo": codigo_riesgo,
                "nivel": nivel_riesgo,
            },
            "n_desapariciones": round(n_desapariciones, 2),
            "ubicacion": {
                "lat": lat,
                "lng": lng,
                "provincia": data["provincia"]
            }
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": "Error interno del servidor",
            "detalle": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
