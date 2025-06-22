import joblib
import pandas as pd

# Cargar el modelo
modelo = joblib.load('bkt_model.joblib')

# Crear un DataFrame de prueba
df = pd.DataFrame([{
    # "student_id": 1,
    "topic_id": 1,
    "correct": 0,
    "PL": 0.4
}])

# Usar el modelo
pred = modelo.predict(df)
print(pred) 
print(pred[0])
