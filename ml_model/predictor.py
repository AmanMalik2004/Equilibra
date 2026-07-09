import pickle
import numpy as np

class AILoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        with open("ml_model/model.pkl", "rb") as f:
            self.model = pickle.load(f)
    
    def get_server(self):
        loads = [s.current_load for s in self.servers]
        request_rate = sum(loads) + 1
        features = np.array([loads + [request_rate]])
        predicted_index = self.model.predict(features)[0]
        return self.servers[predicted_index]