# AI-Based Smart Load Balancer

A simulation system that distributes synthetic client requests across three backend 
servers using three different strategies — Round Robin, Least Connections, and an 
AI-assisted routing algorithm powered by a Decision Tree Classifier — then compares 
their performance.

All server behavior is simulated in memory; no real network communication is involved.

---

## Features

- **Synthetic traffic generation** — configurable number of requests (default: 300) 
  with randomized arrival times and processing durations
- **Round Robin** — cyclic request distribution across servers
- **Least Connections** — routes each request to the server with the fewest active connections
- **AI Load Balancer** — uses a trained `DecisionTreeClassifier` (scikit-learn) to predict 
  the optimal server based on current load
- **Metrics tracking** — records per-request response times and per-server utilization
- **Visualization** — generates comparative bar and line charts (Matplotlib)

---

## Architecture

| Layer | Component(s) | Responsibility |
|---|---|---|
| Simulation Entry | `main.py` | Orchestrates the full pipeline |
| Traffic Generation | `traffic/generator.py` | Produces synthetic request objects |
| Load Balancing | `algorithms/round_robin.py`, `algorithms/least_connection.py`, `ml_model/predictor.py` | Selects target server per request |
| Server Simulation | `servers/server.py` | Simulates request processing, tracks server state |
| ML Pipeline | `ml_model/train.py`, `ml_model/predictor.py` | Trains and applies the Decision Tree Classifier |
| Metrics | `metrics/tracker.py` | Aggregates response-time data |
| Visualization | `visualization/charts.py` | Generates comparison charts |

---

## Requirements

- Python 3.10+
- scikit-learn >= 1.0
- NumPy >= 1.20
- Matplotlib >= 3.4

Install dependencies:
```bash
pip install scikit-learn numpy matplotlib
```

---

## How to Run

From the project root:
```bash
python main.py
```

This will:
1. Train the Decision Tree model (if `ml_model/model.pkl` doesn't already exist)
2. Generate 300 synthetic requests
3. Run all three load-balancing strategies against the same workload
4. Print a comparison summary to the console
5. Save three PNG charts to `visualization/`

To retrain the ML model directly:
```bash
python ml_model/train.py
```

---

## Response Time Model

Each server's simulated response time is computed as:
response_time = Uniform(0.1, 0.5) + current_load * 0.05

where `current_load` is the number of active connections on the server at the time 
the request begins processing.

---

## Output

After running `main.py`, you'll get:
- `visualization/avg_response_time.png` — average response time per algorithm
- `visualization/server_utilzation.png` — requests handled per server per algorithm
- `visualization/response_over_time.png` — per-request response time across the full run

---

## Project Structure
SE Project/
├── main.py
├── algorithms/
│   ├── round_robin.py
│   └── least_connection.py
├── ml_model/
│   ├── train.py
│   ├── predictor.py
│   └── model.pkl
├── servers/
│   └── server.py
├── traffic/
│   └── generator.py
├── metrics/
│   └── tracker.py
└── visualization/
└── charts.py

---

## Notes

- This is a course project (Software Engineering) built as a simulation prototype — 
  it does not implement real network communication.
- Full SRS (IEEE Std 830-1998 compliant) available in `/docs` for implementation details.
