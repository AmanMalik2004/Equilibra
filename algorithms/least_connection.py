class LeastConnections:
    def __init__(self, servers):
        self.servers = servers
    
    def get_server(self):
        return min(self.servers, key=lambda s: s.current_load)