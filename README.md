
📌 API de Predicción de Riesgo y Localización

API desarrollada en Flask que utiliza modelos de Machine Learning (TensorFlow / Keras) para predecir:

Nivel de riesgo por provincia y fecha

Número estimado de desapariciones

Posibles puntos geográficos asociados

Riesgo en un punto geográfico específico

🚀 Tecnologías utilizadas

Python 3.9+

Flask + Flask-CORS

TensorFlow / Keras

NumPy

Joblib

Scikit-learn

📂 Estructura esperada del proyecto
/
├── app.py
├── modelo_lstm_riesgo.h5
├── modelo_geo.h5
├── modelo_riesgo_punto_v2.h5
├── scaler_modelo3.pkl
├── requirements.txt
└── README.md

🔧 Instalación y ejecución del servidor
1️⃣ Crear entorno virtual (opcional pero recomendado)
python -m venv venv


Activar:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

2️⃣ Instalar dependencias

Crea el archivo requirements.txt con:

flask
flask-cors
numpy
tensorflow
joblib
scikit-learn
pydantic


Luego ejecuta:

pip install -r requirements.txt

3️⃣ Ejecutar el servidor Flask
python app.py


El servidor quedará disponible en:

http://localhost:5000

📡 ENDPOINTS
🔹 1. Predicción de contexto (Modelo 1)

Predice el nivel de riesgo y el número estimado de desapariciones para una provincia y fecha.

URL

POST /api/prediccion/contexto

📥 Request (JSON)
{
  "fecha": "2025-01-15",
  "provincia": "PICHINCHA"
}

📤 Response (JSON)
{
  "fecha": "2025-01-15",
  "provincia": "PICHINCHA",
  "riesgo": 2,
  "riesgo_label": "ALTO",
  "desapariciones_estimadas": 12.47
}

🔹 2. Predicción de localización (Modelo 2)

Genera puntos geográficos probables basados en el riesgo y número de casos.

URL

POST /api/prediccion/localizacion

📥 Request (JSON)
{
  "fecha": "2025-01-15",
  "provincia": "PICHINCHA",
  "riesgo": 2,
  "desapariciones_estimadas": 12.47
}

📤 Response (JSON)
{
  "puntos": [
    {
      "lat": 0.42,
      "lng": -0.61,
      "peso": 12.47
    }
  ]
}


📌 Las coordenadas están normalizadas según el entrenamiento del modelo.

🔹 3. Predicción de riesgo por punto geográfico (Modelo 3)

Evalúa el riesgo en una ubicación exacta (lat/lng).

URL

POST /api/prediccion/punto

📥 Request (JSON)
{
  "fecha": "2025-01-15",
  "lat": -0.18,
  "lng": -78.48,
  "provincia": "PICHINCHA"
}

📤 Response (JSON)
{
  "fecha": "2025-01-15",
  "riesgo": {
    "codigo": 1,
    "nivel": "MEDIO"
  },
  "n_desapariciones": 3.21,
  "ubicacion": {
    "lat": -0.18,
    "lng": -78.48,
    "provincia": 17
  }
}

⚠️ Manejo de errores

La API devuelve errores claros en formato JSON:

{
  "error": "Faltan campos requeridos",
  "campos_requeridos": ["fecha", "lat", "lng", "provincia"]
}
