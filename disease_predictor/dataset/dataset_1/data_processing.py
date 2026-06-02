import pandas as pd
import json

def create_symptoms_json():
    df = pd.read_csv("./disease_predictor/dataset/dataset_1/data.csv")

    symptoms = df.columns.to_list()[1:]

    with open("./disease_predictor/dataset/dataset_1/symptoms.json", "w") as symptom_file:
        json.dump(symptoms, symptom_file, indent=2)

def create_disease_json():
    df = pd.read_csv("./disease_predictor/dataset/dataset_1/data.csv")

    diseases = df["diseases"].unique().tolist()

    with open("./disease_predictor/dataset/dataset_1/diseases.json", "w") as disease_file:
        json.dump(diseases, disease_file, indent=2)

create_disease_json()

