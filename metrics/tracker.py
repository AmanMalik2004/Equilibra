class MetricTracker:
    def __init__(self):
        self.response_times = []

    def record(self, response_time):
        self.response_times.append(response_time)
    
    def average_response_time(self):
        if not self.response_times:
            return 0
        return sum(self.response_times) / len(self.response_times)
    
    def total_requests(self):
        return len(self.response_times)