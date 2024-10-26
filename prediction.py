# predict.py
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(X):
    return model.predict(np.array(X).reshape(-1, 1))

if __name__ == "__main__":
    X_test = [[10], [20]]
    predictions = predict(X_test)
    print(f"Predictions: {predictions}")
