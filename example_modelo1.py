
/// MODELO 1 
scaler = MinMaxScaler()

columnas_numericas = [
    'latitud_desaparicion',
    'longitud_desaparicion',
    'mes',
    'dia_semana_cod',
    'dia',
    'anio',
    'rango_edad_cod',
    'sexo_numerico',
    'codigo_provincia'
]

dftest[columnas_numericas] = scaler.fit_transform(dftest[columnas_numericas])


ef crear_secuencias(X, y, pasos=7):
    Xs, ys = [], []
    for i in range(len(X) - pasos):
        Xs.append(X[i:i+pasos])
        ys.append(y[i+pasos])
    return np.array(Xs), np.array(ys)


dfmodelo1 = dftest[[
    'mes', 'dia', 'anio',
    'dia_semana_cod', 'codigo_provincia',
    'nivel_riesgo', 'n_desapariciones'
]].copy()


features_modelo1 = [
    'mes', 'dia', 'anio',
    'dia_semana_cod',
    'codigo_provincia'
]

X1 = dfmodelo1[features_modelo1].values
y_riesgo = dfmodelo1['nivel_riesgo'].values
y_n = dfmodelo1['n_desapariciones'].values


X1_seq, y_riesgo_seq = crear_secuencias(X1, y_riesgo)
_, y_n_seq = crear_secuencias(X1, y_n)


input_lstm = Input(shape=(X1_seq.shape[1], X1_seq.shape[2]))

x = LSTM(64, return_sequences=False)(input_lstm)
x = Dense(32, activation='relu')(x)

salida_riesgo = Dense(3, activation='softmax', name='riesgo')(x)
salida_n = Dense(1, activation='linear', name='desapariciones')(x)

modelo_1 = Model(inputs=input_lstm, outputs=[salida_riesgo, salida_n])


modelo_1.compile(
    optimizer='adam',
    loss={
        'riesgo': 'sparse_categorical_crossentropy',
        'desapariciones': 'mse'
    },
    metrics={
        'riesgo': 'accuracy',
        'desapariciones': 'mae'
    }
)


modelo_1.fit(
    X1_seq,
    {'riesgo': y_riesgo_seq, 'desapariciones': y_n_seq},
    epochs=30,
    batch_size=32,
    validation_split=0.2
)
modelo_1.save('modelo_lstm_riesgo.h5')
print("Modelo guardado exitosamente.")

# 2. Guardar el escalador (el objeto 'scaler' de tu imagen 545849.png)
# Es vital guardar el mismo objeto que hizo el .fit() con tus datos de 2017-2024.
joblib.dump(scaler, 'scaler_desapariciones.pkl')
print("Escalador guardado exitosamente.")



[308]
0 s
riesgo_pred, n_pred = modelo_1.predict(fecha_seq)

nivel_riesgo_pred = np.argmax(riesgo_pred)
n_des_pred = n_pred[0][0]

print("Nivel riesgo:", nivel_riesgo_pred)
print("N desapariciones:", n_des_pred)

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 243ms/step
Nivel riesgo: 0
N desapariciones: 0.27364945

//MODELO 3 


df['fecha_desaparicion'] = pd.to_datetime(df['fecha_desaparicion'])

df['mes'] = df['fecha_desaparicion'].dt.month
df['dia'] = df['fecha_desaparicion'].dt.day
df['dia_semana'] = df['fecha_desaparicion'].dt.weekday
features = [
    'latitud_desaparicion',
    'longitud_desaparicion',
    'mes',
    'dia',
    'dia_semana',
    'codigo_provincia'
]

X = df[features].values
y_risk = df['nivel_riesgo'].values            # clasificación (0,1,2)
y_n = df['n_desapariciones'].values.astype(float)  # regresión (puede ser decimal)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler_modelo3.pkl')
X_train, X_test, y_r_train, y_r_test, y_n_train, y_n_test = train_test_split(
    X_scaled, y_risk, y_n, test_size=0.2, random_state=42
)
input_layer = Input(shape=(X_scaled.shape[1],))

h = Dense(64, activation='relu')(input_layer)
h = Dropout(0.3)(h)
h = Dense(32, activation='relu')(h)

# salida clasificación
out_riesgo = Dense(3, activation='softmax', name='riesgo')(h)
# salida regresión
out_n = Dense(1, activation='linear', name='desapariciones')(h)

modelo_3 = Model(inputs=input_layer, outputs=[out_riesgo, out_n])
modelo_3.compile(
    optimizer='adam',
    loss={
        'riesgo': 'sparse_categorical_crossentropy',
        'desapariciones': 'mse'
    },
    metrics={
        'riesgo': 'accuracy',
        'desapariciones': 'mae'
    }
)
history = modelo_3.fit(
    X_train,
    {'riesgo': y_r_train, 'desapariciones': y_n_train},
    validation_data=(X_test, {'riesgo': y_r_test, 'desapariciones': y_n_test}),
    epochs=40,
    batch_size=32
)

modelo_3.save("modelo_riesgo_punto_v2.h5")
print("Modelo 3 guardado: modelo_riesgo_punto_v2.h5")

# Punto de prueba (Guayaquil)
sample = np.array([[ 
    -2.19,    # lat
    -79.88,   # lon
    10,       # mes
    15,       # dia
    4,        # viernes
    9         # GUAYAS
]], dtype=np.float32)

# Escalar con el MISMO scaler del entrenamiento
sample_scaled = scaler.transform(sample)

# Predicción
riesgo_pred, n_pred = modelo_3.predict(sample_scaled)

# Resultados
codigo_riesgo = int(np.argmax(riesgo_pred[0]))
n_des = float(n_pred[0][0])

etiquetas = ["BAJO", "MEDIO", "ALTO"]

print("Riesgo:", codigo_riesgo, etiquetas[codigo_riesgo])
print("Probabilidades:", riesgo_pred[0])
print("N desapariciones:", n_des)


me sale es ta mrd 

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 93ms/step
Riesgo: 0 BAJO
Probabilidades: [nan nan nan]
N desapariciones: nan