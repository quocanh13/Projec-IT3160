import pandas as pd
import numpy as np
import time
import pickle
from disease_predictor.models.naive_bayes.model import NaiveBayes

if __name__ == "__main__":
    CSV_TRAIN_PATH = './disease_predictor/dataset/dataset_1/train_data.csv'
    CSV_TEST_PATH = './disease_predictor/dataset/dataset_1/test_data.csv' 
    MODEL_PATH = 'naive_bayes_model.pkl'
    
    df_train = pd.read_csv(CSV_TRAIN_PATH).dropna(axis=1, how='all')
    df_test = pd.read_csv(CSV_TEST_PATH).dropna(axis=1, how='all')
    
    target_col = df_train.columns[0]
    
    X_train = df_train.drop(columns=[target_col]).values
    
    unique_diseases = list(df_train[target_col].unique())
    disease_to_id = {disease: i for i, disease in enumerate(unique_diseases)}
    id_to_disease = {i: disease for disease, i in disease_to_id.items()}
    
    y_train = df_train[target_col].map(disease_to_id).values
    
    X_test = df_test.drop(columns=[target_col]).values
    y_test = df_test[target_col].map(disease_to_id)
    
    valid_idx = y_test.notna()
    X_test = X_test[valid_idx]
    y_test = y_test[valid_idx].astype(int).values

    start_time = time.time()
    
    model = NaiveBayes()
    model.fit(X_train, y_train)

    y_pred = model.predict_id(X_test)
    acc = np.sum(y_pred == y_test) / len(y_test)
    print(f"Độ chính xác trên tập Test: {acc * 100:.2f}%\n")

    export_package = {
        'model': model,
        'id_to_disease': id_to_disease
    }
    
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(export_package, file)
