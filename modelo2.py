# 'dia', 'mes', 'anio', 'dia_semana_cod', 'codigo_provincia', 'rango_edad_cod', 'sexo_numerico', 'nivel_riesgo', 'n_desapariciones'
features_modelo2 = [
    'dia', 'mes', 'anio',
    'dia_semana_cod',
    'codigo_provincia',
    'rango_edad_cod',
    'sexo_numerico',
    'nivel_riesgo',
    'n_desapariciones',
]

X2 = df_geo[features_modelo2].values
y_geo = df_geo[['latitud_desaparicion', 'longitud_desaparicion']].values

X2_seq, y_geo_seq = crear_secuencias(X2, y_geo)\\
input_geo = Input(shape=(X2_seq.shape[1], X2_seq.shape[2]))

x = LSTM(64)(input_geo)
x = Dense(32, activation='relu')(x)

salida_geo = Dense(2, activation='linear')(x)

modelo_2 = Model(input_geo, salida_geo)
modelo_2.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
modelo_2.fit(
    X2_seq,
    y_geo_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)