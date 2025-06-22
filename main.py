from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BKT Model API",
    description="API para predicción de conocimiento estudiantil usando modelo BKT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model_path = Path("bkt_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
    modelo = joblib.load(model_path)
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    raise

class InputData(BaseModel):
    topic_id: int = Field(..., description="ID del tópico")
    correct: int = Field(..., ge=0, le=1, description="Respuesta correcta (0 o 1)")
    PL: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de aprender")

class PredictionResponse(BaseModel):
    prediccion: float
    topic_id: int
    status: str = "success"

@app.get("/")
async def root():
    return {"message": "BKT Model API está funcionando", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": modelo is not None}

@app.post("/predict-student-knowledge")
async def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        logger.info(f"Realizando predicción para topic_id: {data.topic_id}")
        
        pred = modelo.predict(df)
        
        if len(pred) == 0:
            raise ValueError("El modelo no devolvió predicción")
            
        resultado = float(pred[0])
        
        logger.info(f"Predicción exitosa: {resultado}")
        
        return PredictionResponse(
            prediccion=resultado,
            topic_id=data.topic_id
        )
        
    except Exception as e:
        logger.error(f"Error al predecir: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)