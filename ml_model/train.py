import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
import random

def generate_training_data(samples=1000):
    x = []
    y = []

    for _ in range(samples):
        load1 = random.randint(0, 20)
        load2 = random.randint(0, 20)
        load3 = random.randint(0, 20)
        request_rate = random.randint(1, 50)

        loads = [load1, load2, load3]
        best_server = loads.index(min(loads))

        x.append([load1, load2, load3, request_rate])
        y.append(best_server)

    return np.array(x), np.array(y)

def train_model():
    x, y = generate_training_data()
    model = DecisionTreeClassifier()
    model.fit(x, y)

    os.makedirs("ml_model", exist_ok=True)
    with open("ml_model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model trained and saved.")
    return model

if __name__ == "__main__":
    train_model()