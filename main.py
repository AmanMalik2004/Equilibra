from servers.server import Server
from algorithms.round_robin import RoundRobin
from algorithms.least_connection import LeastConnections
from ml_model.predictor import AILoadBalancer
from ml_model.train import train_model
from traffic.generator import generate_requests
from metrics.tracker import MetricTracker
from visualization.charts import(
    plot_avg_response_time,
    plot_server_utilization,
    plot_response_time_over_time
)
import os

def run_simulatuion(balancer, servers, requests, tracker):
    response_times = []
    for r in requests:
        server = balancer.get_server()
        rt = server.handle_request(r["id"])
        tracker.record(rt)
        response_times.append(rt)
    return response_times

def reset_servers(servers):
    for s in servers:
        s.reset()

def main():
    if not os.path.exists("ml_model/model.pkl"):
        print("Training ML model...")
        train_model()
    
    requests = generate_requests(300)
    results = {}

    servers_rr = [Server("Server1"), Server("Server2"), Server("Server3")]
    rr = RoundRobin(servers_rr)
    tracker_rr = MetricTracker()
    rr_times = run_simulatuion(rr, servers_rr, requests, tracker_rr)
    results["Round Robin"] = {
        "avg_response_time": tracker_rr.average_reponse_time(),
        "utilization": [s.utilization() for s in servers_rr]
    }
    print(f"Round Robin    | Avg Response Time: {tracker_rr.average_reponse_time():.4f}s | Total Requests: {tracker_rr.total_requests()}")

    servers_lc = [Server("Server1"), Server("Server2"), Server("Server3")]
    lc = LeastConnections(servers_lc)
    tracker_lc = MetricTracker()
    lc_times = run_simulatuion(lc, servers_lc, requests, tracker_lc)
    results["Least Connections"] = {
        "avg_response_time": tracker_lc.average_reponse_time(),
        "utilization": [s.utilization() for s in servers_lc]
    }
    print(f"Least Connections    | Avg Response Time: {tracker_lc.average_reponse_time():.4f}s | Total Requests: {tracker_lc.total_requests()}")

    servers_ai = [Server("Server1"), Server("Server2"), Server("Server3")]
    ai = AILoadBalancer(servers_ai)
    tracker_ai = MetricTracker()
    ai_times = run_simulatuion(ai, servers_ai, requests, tracker_ai)
    results["AI Balancer"] = {
        "avg_response_time": tracker_ai.average_reponse_time(),
        "utilization": [s.utilization() for s in servers_ai]
    }
    print(f"AI Balancer    | Avg Response Time: {tracker_ai.average_reponse_time():.4f}s | Total Requests: {tracker_ai.total_requests()}")

    os.makedirs("visualization", exist_ok=True)
    plot_avg_response_time(results)
    plot_server_utilization(results)
    plot_response_time_over_time(rr_times, lc_times, ai_times)

if __name__ == "__main__":
    main()