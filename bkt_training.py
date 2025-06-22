# Importar bibliotecas
import pandas as pd
from pyBKT.models import Model
import joblib

if __name__ == '__main__':
    # Leer datos desde CSV
    data = pd.read_csv('students_performance.csv')  # Reemplaza 'tu_archivo.csv' con la ruta de tu archivo

    # Mostrar información básica del dataset
    print("Primeras 5 filas del dataset:")
    print(data.head())
    print(f"\nTamaño del dataset: {data.shape}")
    print(f"Estudiantes únicos: {data['student_id'].nunique()}")
    print(f"Temas únicos: {data['topic_id'].nunique()}")

    # Renombrar columnas para que coincidan con el formato esperado por pyBKT
    data_bkt = data.rename(columns={
        'student_id': 'user_id',
        'topic_id': 'skill_name'
    })

    # Convertir topic_id a string si es necesario (pyBKT puede preferir strings para skill_name)
    data_bkt['skill_name'] = data_bkt['skill_name'].astype(str)

    # Verificar que los valores en 'correct' sean 0 o 1
    print(f"\nValores únicos en 'correct': {data_bkt['correct'].unique()}")

    # Mostrar el dataframe preparado para BKT
    print("\nDatos preparados para BKT:")
    print(data_bkt.head())

    # Inicializar y entrenar el modelo BKT
    print("\nEntrenando modelo BKT...")
    model = Model()
    model.fit(data=data_bkt)

    # Guardar el modelo entrenado
    #joblib.dump(model, 'bkt_model.joblib') 

    # Obtener los parámetros estimados para cada habilidad
    params = model.params()
    print(f"\nParámetros estimados para cada tema:")
    print(params)
    
    # Obtener las habilidades únicas (temas)
    skills = params.index.get_level_values('skill').unique()
    
    for skill in skills:
        print(f"\nTema {skill}:")
        try:
            # Acceder a cada parámetro usando el MultiIndex
            prior = params.loc[(skill, 'prior', 'default'), 'value']
            learns = params.loc[(skill, 'learns', 'default'), 'value']
            guesses = params.loc[(skill, 'guesses', 'default'), 'value']
            slips = params.loc[(skill, 'slips', 'default'), 'value']
            forgets = params.loc[(skill, 'forgets', 'default'), 'value']
            
            print(f"  - Probabilidad inicial de conocimiento (P(L0)): {prior:.3f}")
            print(f"  - Probabilidad de transición/aprendizaje (P(T)): {learns:.3f}")
            print(f"  - Probabilidad de adivinar (P(G)): {guesses:.3f}")
            print(f"  - Probabilidad de deslizamiento (P(S)): {slips:.3f}")
            print(f"  - Probabilidad de olvido (P(F)): {forgets:.3f}")
            
        except KeyError as e:
            print(f"  Error accediendo a parámetros: {e}")
            print(f"  Estructura disponible para tema {skill}:")
            skill_data = params.loc[skill]
            print(skill_data)

    # Ejemplo de predicción para un estudiante nuevo
    # Crear datos de ejemplo para predicción
    ejemplo_estudiante = pd.DataFrame({
        'user_id': [999],  # ID de estudiante nuevo
        'skill_name': ['1'],  # Tema específico (ajusta según tus datos)
        'correct': [1]  # Respuesta hipotética
    })

    # Realizar predicción
    try:
        predicciones = model.predict(data=ejemplo_estudiante)
        print(f"\nTipo de predicciones: {type(predicciones)}")
        
        # Inspeccionar la estructura de las predicciones
        if hasattr(predicciones, 'columns'):
            print(f"Columnas de predicciones: {list(predicciones.columns)}")
            print(f"Predicciones completas:")
            print(predicciones)
            
            # Intentar acceder a diferentes nombres posibles para mastery
            mastery_cols = [col for col in predicciones.columns if 'mastery' in col.lower() or 'knowledge' in col.lower() or 'probability' in col.lower()]
            if mastery_cols:
                print(f"\nPredicción de dominio para estudiante 999 en tema 1: {predicciones[mastery_cols[0]].values}")
            else:
                print(f"\nNo se encontró columna de mastery. Columnas disponibles: {list(predicciones.columns)}")
        else:
            print(f"Predicciones: {predicciones}")
            
    except Exception as e:
        print(f"\nError en predicción: {e}")
        print("Verifica que el tema usado en la predicción existe en los datos de entrenamiento")
        
        # Intentar con un tema que sabemos que existe
        try:
            print("\nIntentando predicción con todos los temas disponibles...")
            for skill in skills:
                ejemplo_skill = pd.DataFrame({
                    'user_id': [999],
                    'skill_name': [str(skill)],  # Convertir a string
                    'correct': [1]
                })
                pred_skill = model.predict(data=ejemplo_skill)
                print(f"Predicción para tema {skill}: {pred_skill}")
                break  # Solo mostrar el primero que funcione
        except Exception as e2:
            print(f"Error en predicción alternativa: {e2}")

    # Estadísticas adicionales del modelo
    print(f"\nResumen del entrenamiento:")
    print(f"- Temas procesados: {list(skills)}")
    print(f"- Número total de observaciones: {len(data_bkt)}")