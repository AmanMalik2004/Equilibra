from flask import Flask, jsonify, render_template
from servers.server import Server
from algorithms.round_robin import RoundRobin
from algorithms.least_connection import LeastConnections
from ml_model.predictor import AILoadBalancer
from ml_model.train import train_model
from traffic.generator import generate_requests
from metrics.tracker import MetricTracker
import os

app = Flask(__name__)

def run_algo(balancer, servers, requests):
    tracker = MetricTracker()
    times = []
    log = []
    for req in requests:
        server = balancer.get_server()
        rt = server.handle_request(req["id"])
        tracker.record(rt)
        times.append(round(rt, 4))
        log.append({"req": req["id"], "server": server.name, "rt": round(rt, 4)})
    return {
        "avg_response_time": round(tracker.average_response_time(), 4),
        "total_requests": tracker.total_requests(),
        "utilization": [{"name": s.name, "total": s.utilization(), "avg_rt": round(s.avg_response_time(), 4)} for s in servers],
        "times": times,
        "log": log
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/simulate/<int:n>")
def simulate(n):
    if not os.path.exists("ml_model/model.pkl"):
        train_model()

    requests = generate_requests(n)

    servers_rr = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
    servers_lc = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]
    servers_ai = [Server("SRV-1"), Server("SRV-2"), Server("SRV-3")]

    rr_result = run_algo(RoundRobin(servers_rr), servers_rr, requests)
    lc_result = run_algo(LeastConnections(servers_lc), servers_lc, requests)
    ai_result = run_algo(AILoadBalancer(servers_ai), servers_ai, requests)

    return jsonify({
        "round_robin": rr_result,
        "least_connections": lc_result,
        "ai_balancer": ai_result
    })

if __name__ == "__main__":
    app.run(debug=True)