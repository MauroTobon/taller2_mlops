from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import nest_asyncio

# Aplicar parche para correr async event loop en Colab
nest_asyncio.apply()

# Cargar modelo
modelo = joblib.load('xgb_pipeline_final.pkl')

# Crear app
app = FastAPI()

# Clase de entrada (ajústala a tus columnas)
class InputData(BaseModel):
    Age: float
    StudyTimeWeekly: float
    Absences: int
    ParentalEducation: int
    ParentalSupport: int
    Gender: int
    Tutoring: int
    Extracurricular: int
    Sports: int
    Music: int
    Volunteering: int
    Ethnicity: int

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = modelo.predict(df)
    return {"prediction": int(pred[0])}