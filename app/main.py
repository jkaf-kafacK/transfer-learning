from fastapi import FastAPI

from datetime import datetime
import uvicorn
import numpy as np
from pydantic import BaseModel

from model import model_train,predicte,val_accu
app = FastAPI()

x_data = [4.4, 2.9, 1.4, 0.2]


#y_data = [0,1,2,3]

class InputIrisData(BaseModel):
    """
    Input data for iris species predictions, sepal length/width, petal length/width, with defaults as the median.
    """

    sl: float = 6.9   
    sw: float = 3.1
    pl: float = 5.4
    pw: float = 2.1

class OutputIrisData(BaseModel):
    """
    Output data for iris species predictions, ordinal number of species.
    """
    species: int



# BASIC ROUTES
@app.get("/")
async def root():
    """
    Simple message for homepage
    """
    return {"message": "Welcome!"}


@app.get("/hello")
async def hello():
    """
    Returns heartbeat with timestamp
    """
    return {"message": f"{datetime.now():%Y-%m-%d %H:%M:%S.%f}"}

@app.post("/training_iris")
async def training_iris():
    #model_train()
    Model_train = model_train()
    return {"training ": str(Model_train)}

@app.post("/predict_iris", response_model=OutputIrisData)
async def predict_iris_(data: InputIrisData):
    """
    Processes numbers from input and predicts iris species.
    """
    feat = np.array([v for k, v in data.dict().items()])
    model_input = feat.reshape(1, -1)
    result =predicte(model_input)

    return {"species": result}


@app.post("/accuracy_iris")
async def accuracy_iris():
    #predicte(x_data)
    return {"val data":val_accu()}


@app.get("/iris_report")
async def iris_report():
    """
    report
    """
    return {"report": "no report"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000)