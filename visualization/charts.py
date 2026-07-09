import matplotlib.pyplot as plt

def plot_avg_response_time(results):
    algorithms = list(results.keys())
    avg_times = [results[a]["avg_response_time"] for a in algorithms]

    plt.figure(figsize=(8,5))
    plt.bar(algorithms, avg_times, color=["steelblue", "tomato", "seagreen"])
    plt.title("Average Response Time by Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Response Time(s)")
    plt.tight_layout()
    plt.savefig("visualization/avg_response_time.png")
    plt.show()

def plot_server_utilization(results):
    algorithms = list(results.keys())
    server_names = ["Server1", "Server2", "Server3"]
    x = range(len(server_names))
    width = 0.25

    plt.figure(figsize=(10,6))
    for i, a in enumerate(algorithms):
        utils = results[a]["utilization"]
        bars = [v + i * width for v in x]
        plt.bar(bars, utils, width=width, label=a)
    
    plt.title("Server Utilization by Algorithm")
    plt.xlabel("Server")
    plt.ylabel("Total Requests Handled")
    plt.xticks([v + width for v in x], server_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization/server_utilzation.png")
    plt.show()

def plot_response_time_over_time(rr_times, lc_times, ai_times):
    plt.figure(figsize=(12, 5))
    plt.plot(rr_times, label="Round Robin", alpha=0.7)
    plt.plot(lc_times, label="Least Connections", alpha=0.7)
    plt.plot(ai_times, label="AI Balancer", alpha=0.7)
    plt.title("Response Time per Request")
    plt.xlabel("Request #")
    plt.ylabel("Response Time(s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization/response_over_time.png")
    plt.show()
