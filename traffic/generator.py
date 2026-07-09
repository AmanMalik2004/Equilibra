import random

def generate_requests(n=300):
    requests = []
    for i in range(n):
        requests.append({
            "id": i + 1,
            "arrival time": i * random.uniform(0.01, 0.05),
            "duration": random.uniform(0.1, 0.5)
        })
    return requests