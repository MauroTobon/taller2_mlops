from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from pandas import DataFrame

# Cargar modelo
modelo = load('xgb_pipeline_final.pkl')

# Crear app
app = FastAPI()

# Clase de entrada (ajustala a tus columnas)
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
    df = DataFrame([data.dict()])
    pred = modelo.predict(df)
    return {"prediction": int(pred[0])}