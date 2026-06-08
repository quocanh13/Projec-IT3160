import pandas as pd
import json

def create_symptoms_json():
    df = pd.read_csv("./disease_predictor/dataset/dataset_1/data.csv")

    symptoms = df.columns.to_list()[1:]
    res = {}
    for i, s in enumerate(symptoms):
        res[s] = i
    with open("./disease_predictor/dataset/dataset_1/symptoms.json", "w") as symptom_file:
        json.dump(res, symptom_file, indent=2)

def create_disease_json():
    df = pd.read_csv("./disease_predictor/dataset/dataset_1/data.csv")

    diseases = df["diseases"].unique().tolist()
    res = {}
    for i, d in enumerate(diseases):
        res[d] = i
    

    with open("./disease_predictor/dataset/dataset_1/diseases.json", "w") as disease_file:
        json.dump(res, disease_file, indent=2)

create_symptoms_json()

