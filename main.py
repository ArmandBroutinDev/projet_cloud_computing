import joblib
from fastapi import FastAPI

app = FastAPI()

@app.get('/predict/{tweet}')
def predict(tweet:str):
    #load model 
    model_filename = "./model/hatespeech.joblib.z"
    clf = joblib.load(model_filename)
    # Receives the input query from form
    probas = clf.predict_proba([(tweet)])[0]

    return {"langage haineux ": probas[0], "langage offensif ": probas[1], "aucun des deux ": probas[2]}






