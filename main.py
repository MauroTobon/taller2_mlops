!pip install fastapi uvicorn nest-asyncio pyngrok

from pyngrok import ngrok

# Conectar ngrok con tu cuenta
ngrok.set_auth_token("2z7fXSKiBmQzjRsVF4NwbkEPzfz_3gp8C1ZDPHUcX1U5BLoRM")

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import nest_asyncio
from pyngrok import ngrok
import uvicorn

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

# Abrir túnel público a Colab
public_url = ngrok.connect(8000)
print("Tu API está disponible en:", public_url)

# Ejecutar servidor
uvicorn.run(app, host="0.0.0.0", port=8000)