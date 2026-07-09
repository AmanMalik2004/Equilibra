import random

class Server:
    def __init__(self, name):
        self.name = name
        self.current_load = 0
        self.total_requests = 0
        self.total_response_time = 0

    def handle_request(self, request_id):
        self.current_load += 1
        self.total_requests += 1
# 
        base_time = random.uniform(0.1, 0.5)
        load_penalty = self.current_load * 0.05
        response_time = base_time + load_penalty

        self.total_response_time += response_time

        # Load drops gradually instead of instantly
        drop = random.randint(0, 1)  # 50% chance load stays elevated
        if drop:
            self.current_load = max(0, self.current_load - 1)

        return response_time

    def avg_response_time(self):
        if self.total_requests == 0:
            return 0
        return self.total_response_time / self.total_requests

    def utilization(self):
        return self.total_requests

    def reset(self):
        self.current_load = 0
        self.total_requests = 0
        self.total_response_time = 0